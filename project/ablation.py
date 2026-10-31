"""Fixed matched CatBoost ablation; never selects or changes the main ensemble."""
import json
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd

from features import TARGET, roc_date
from train import fit_model, predict, scores, dump, subset


def main():
    root = Path(__file__).resolve().parent
    out = root / 'artifacts/run_v2'
    if (out / 'ablation.json').exists():
        print((out / 'ablation.json').read_text())
        return
    # Same rows, parameters, target, seed, and number of trees: remove only market features.
    reference = joblib.load(out / 'dev_cat_total.joblib')
    config = json.loads((out / 'config.json').read_text())
    dump(out / 'ablation_protocol.json', {'fixed_before_evaluation': True,
        'comparison': 'CatBoost log-total, same rounds; with versus without causal market features',
        'iterations': reference['iterations'], 'changes_main_model': False})
    args = SimpleNamespace(iterations=reference['iterations'], device=config['device'], threads=config['threads'])
    raw = pd.read_parquet(root / 'data/train.parquet')
    y0 = pd.to_numeric(raw[TARGET], errors='coerce')
    dates = roc_date(raw['交易年月日'])
    raw = raw.loc[y0.notna() & (y0 > 0) & dates.between('2000-01-01', '2024-08-10')].copy()
    raw['_date'] = dates.loc[raw.index]
    raw = raw.sort_values('_date', kind='stable').reset_index(drop=True)
    dates = raw['_date']
    y = pd.to_numeric(raw[TARGET]).to_numpy(float)
    x, cats, groups, offset = joblib.load(out / 'features.joblib')
    assert len(raw) == len(x)
    pre_ix = np.where(dates <= '2023-12-31')[0]
    path = out / 'ablation_no_market.joblib'
    if path.exists():
        baseline = joblib.load(path)
    else:
        baseline = fit_model(reference['spec'], subset(x, pre_ix), y[pre_ix], offset[pre_ix], cats,
                             args, iterations=reference['iterations'])
        joblib.dump(baseline, path, compress=3)
    market = joblib.load(out / 'market_dev.joblib')
    report = {'comparison': 'Same log-total CatBoost and rounds, remove causal market features only',
              'fixed_model_not_used_for_selection': True, 'evaluations': {}}
    for label, mask in [('calibration', (dates > '2023-12-31') & (dates <= '2024-04-30')),
                        ('temporal_audit', dates > '2024-04-30')]:
        ix = np.where(mask)[0]
        xx = subset(x, ix)
        base = predict(baseline, xx, offset[ix])
        enhanced = predict(reference, market.transform(xx), offset[ix])
        report['evaluations'][label] = {'without_market': scores(y[ix], base),
                                        'with_market': scores(y[ix], enhanced)}
    dump(out / 'ablation.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
