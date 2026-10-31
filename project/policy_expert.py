"""Published relocation offers + observed contract components, independent of labels.

The schedules are offers, not observed test transactions. Publication must precede
each query date; this is NOT a model frozen at the competition train cutoff.
"""
import json
import re
import shutil
import unicodedata

import numpy as np
import pandas as pd

from features import ID, TARGET, normalize, roc_date
from policy_tables import ROOT, OUT, SOURCES, main as extract_tables, unit_key
from train_mae import checksum, dump, export_submission

POLICY_PUBLISHED = '2024-08-23'
BASE = ROOT / 'submission1017_v4.csv'
BASE_SHA = '723ce5f0c04740fdcdc4ff9cfc3976deddb644a0462fa5ab3e6a8b28aff7e058'


def contract_discount(note):
    """Only subtract an explicitly observed component difference, never the total."""
    text = normalize(note).replace(',', '').replace('，', '')
    if '減收' not in text:
        return 0, 0, 0
    match = re.search(r'原房屋價款(\d+)元.*?差額減收後之房屋價款為(\d+)元', text)
    if not match:
        raise ValueError(f'Unparsed discount: {note}')
    original, reduced = map(int, match.groups())
    if not 0 <= reduced <= original:
        raise ValueError('Invalid original/reduced house component')
    return original, reduced, original - reduced


def parking_count(value):
    text = unicodedata.normalize('NFKC', str(value))
    match = re.fullmatch(r'土地\d+建物1車位(\d+)', text)
    if not match:
        raise ValueError(f'Unsupported transaction counts: {value}')
    count = int(match[1])
    if count not in (0, 1):
        raise ValueError('Relocation expert supports zero/one parking space only')
    return count


def predict_policy(raw, houses, parking):
    """Return an ID-indexed audit for eligible rows; no target/row order dependence."""
    if raw[ID].astype(str).duplicated().any():
        raise ValueError('Duplicate query IDs')
    if houses.duplicated(['project', 'unit']).any():
        raise ValueError('Ambiguous schedule')
    prices = parking.groupby('project').price.agg(['median', 'min', 'max'])
    if not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any().any():
        raise ValueError('Invalid parking prices')
    table = houses.set_index(['project', 'unit'])
    projects = set(houses.project)
    notes = raw['備註'].fillna('').map(normalize)
    selected = raw.loc[notes.str.contains('安置住宅') & raw['建案名稱'].map(normalize).isin(projects)].copy()
    dates = roc_date(selected['交易年月日'])
    records = []
    for idx, row in selected.iterrows():
        project, unit = normalize(row['建案名稱']), unit_key(row['棟及號'])
        if (project, unit) not in table.index:
            raise ValueError(f'No unique public offer for {project}/{unit}')
        offer = table.loc[(project, unit)]
        available = max(pd.Timestamp(POLICY_PUBLISHED), pd.Timestamp(offer.source_published))
        if pd.isna(dates.loc[idx]) or dates.loc[idx] < available:
            raise ValueError('Source/policy was not available before the query transaction')
        cars = parking_count(row['交易筆棟數'])
        gross, park_area = float(row['建物移轉總面積平方公尺']), float(row['車位移轉總面積平方公尺'])
        if not np.isfinite([gross, park_area]).all() or park_area < 0 or gross <= park_area:
            raise ValueError('Invalid net area')
        if (cars == 0) != (park_area == 0):
            raise ValueError('Parking count/area conflict')
        net = gross - park_area
        error = net - offer.area_sqm
        # Published terms provide area reconciliation. Proportional adjustment is
        # an approximation for small revisions, explicitly audited, not exact truth.
        if abs(error) / offer.area_sqm > .05:
            raise ValueError('Area mismatch exceeds the predeclared 5% limit')
        ratio = net / offer.area_sqm if abs(error) > .05 else 1.0
        original, reduced, discount = contract_discount(row['備註'])
        # 2023 nearby book precedes the universal cut; 2024 airport book includes it.
        cut = offer.area_ping * offer.policy_cut_per_ping
        house = (offer.published_house_land_price - cut) * ratio
        park = prices.loc[project]
        estimate = house + cars * park['median'] - discount
        if not np.isfinite(estimate) or estimate <= 0:
            raise ValueError('Invalid policy estimate')
        unscaled_house = offer.published_house_land_price - cut
        records.append({
            ID: str(row[ID]), 'project': project, 'unit': unit,
            'transaction_date': str(dates.loc[idx].date()),
            'source': offer.source, 'source_page': int(offer.source_page),
            'source_published': offer.source_published, 'policy_published': POLICY_PUBLISHED,
            'net_area_sqm': net, 'published_area_sqm': offer.area_sqm,
            'area_error_sqm': error, 'area_scale': ratio,
            'published_house_land_price': offer.published_house_land_price,
            'universal_cut_ntd': cut, 'adjusted_house_land_price': house,
            'parking_count': cars, 'parking_estimate': cars * park['median'],
            'note_original_house_component': original,
            'note_reduced_house_component': reduced, 'explicit_discount': discount,
            'policy_prediction': estimate,
            # Scenario bounds, not statistical prediction intervals; actual contract
            # prices may also differ from the published offer.
            'scenario_low': min(house, unscaled_house) + cars * park['min'] - discount,
            'scenario_high': max(house, unscaled_house) + cars * park['max'] - discount,
        })
    return pd.DataFrame(records)


def main():
    if checksum(BASE) != BASE_SHA:
        raise ValueError('Frozen v4 changed')
    # Verify the downloaded PDFs before extracting their archived text.
    for source in SOURCES:
        manifest = json.loads((OUT/'sources'/f'{source}.json').read_text())
        if checksum(OUT/'sources'/f'{source}.pdf') != manifest['sha256']:
            raise ValueError('External PDF changed')
    source_names = ['policy_expert.py', 'policy_tables.py', 'features.py', 'train_mae.py', 'test_policy.py']
    input_files = [BASE, ROOT/'data/test.parquet', ROOT/'data/test-reindex-test-sample-submission-v2.csv']
    input_files += sorted((OUT/'sources').glob('*.pdf')) + sorted((OUT/'sources').glob('*.txt'))
    input_files += sorted((OUT/'sources').glob('*.html')) + sorted((OUT/'sources').glob('*.json'))
    config = {
        'version': 'policy_v1', 'base_sha256': BASE_SHA,
        'sources': SOURCES, 'policy_published': POLICY_PUBLISHED,
        'gate': 'note contains 安置住宅 AND exact project in official offer schedules',
        'matching': 'exact normalized project/unit; net area tolerance 0.05 sqm',
        'area_revision': 'net/published proportional correction if mismatch <=5%; otherwise refuse',
        'parking': 'median over published inventory of that project; zero if no car',
        'discount': 'subtract only original-minus-reduced component explicitly stated in input notes',
        'source_availability': 'before each affected transaction, NOT all before 2024-08-10 train cutoff',
        'selection': 'No coefficient fitted to leaderboard/test labels; policy subgroup absent in train',
        'source_sha256': {f: checksum(ROOT/f) for f in source_names},
        'input_sha256': {str(f.relative_to(ROOT)): checksum(f) for f in input_files},
    }
    config = json.loads(json.dumps(config))
    if (OUT/'config.json').exists() and json.loads((OUT/'config.json').read_text()) != config:
        raise ValueError('Frozen policy experiment changed; use a new version')
    dump(OUT/'config.json', config)
    extract_tables()
    raw = pd.read_parquet(ROOT/'data/test.parquet')
    old = pd.read_csv(BASE, dtype={ID: str}).set_index(ID)[TARGET]
    ids = raw[ID].astype(str)
    if old.index.duplicated().any() or set(ids) != set(old.index):
        raise ValueError('Base IDs mismatch')
    audit = predict_policy(raw, pd.read_csv(OUT/'house_schedules.csv'), pd.read_csv(OUT/'parking_schedules.csv'))
    if len(audit) != 1408 or raw['備註'].fillna('').str.contains('安置住宅').sum() != len(audit):
        raise ValueError('Unexpected relocation coverage; review rather than silently fall back')
    audit['v4_prediction'] = audit[ID].map(old)
    audit['change_ntd'] = np.rint(audit.policy_prediction) - audit.v4_prediction
    prediction = ids.map(old).to_numpy(float)
    changed = ids.isin(audit[ID]).to_numpy()
    lookup = audit.set_index(ID).policy_prediction
    prediction[changed] = ids[changed].map(lookup)
    submission = export_submission(raw, prediction, ROOT/'submission1017_v5.csv')
    np.testing.assert_array_equal(submission.loc[~changed, TARGET], ids[~changed].map(old))
    audit.to_csv(OUT/'row_audit.csv', index=False, encoding='utf-8-sig')
    audit.groupby('project').agg(rows=(ID,'size'), v4_mean=('v4_prediction','mean'),
        v5_mean=('policy_prediction','mean'), explicit_discount_rows=('explicit_discount',lambda s: (s>0).sum()),
        area_revision_rows=('area_scale',lambda s: (s!=1).sum())).to_csv(OUT/'project_audit.csv', encoding='utf-8-sig')
    summary = {
        'rows': len(raw), 'changed_rows': int(changed.sum()), 'unchanged_rows': int((~changed).sum()),
        'matched_house_offers': len(audit), 'area_exact_within_005sqm': int((audit.area_error_sqm.abs()<=.05).sum()),
        'area_adjusted_ids': audit.loc[audit.area_scale!=1, ID].tolist(),
        'discount_rows': int((audit.explicit_discount>0).sum()),
        'no_parking_rows': int((audit.parking_count==0).sum()),
        'relocation_v4_mean': float(audit.v4_prediction.mean()),
        'relocation_v5_mean': float(audit.policy_prediction.mean()),
        'relocation_v5_min': float(audit.policy_prediction.min()),
        'relocation_v5_max': float(audit.policy_prediction.max()),
        'whole_set_mean_absolute_prediction_change': float(audit.change_ntd.abs().sum()/len(raw)),
        'scenario_width_mean': float((audit.scenario_high-audit.scenario_low).mean()),
        'submission_sha256': checksum(ROOT/'submission1017_v5.csv'),
        'public_mae': None, 'evaluation': 'unlabelled policy experiment; matching is not MAE validation',
    }
    dump(OUT/'verification.json', summary)
    (OUT/'source').mkdir(exist_ok=True)
    for name in source_names:
        shutil.copy2(ROOT/name, OUT/'source'/name)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
