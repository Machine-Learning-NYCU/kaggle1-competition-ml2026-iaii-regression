"""Cache the supplied workbooks and audit features, without inspecting holdout labels."""
import hashlib
import json
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent / 'data'


def main():
    audit = {}
    for name, source in [('train', 'train-v2.xlsx'), ('valid', 'valid-v2.xlsx'),
                         ('test', 'test-reindex-test-v2.2.xlsx')]:
        path = DATA / source
        cache = DATA / f'{name}.parquet'
        stamp = DATA / f'{name}.source.json'
        checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        cached_source = json.loads(stamp.read_text()) if stamp.exists() else {}
        if not cache.exists() or cached_source.get('sha256') != checksum:
            frame = pd.read_excel(path, engine='calamine').astype('string')
            frame.to_parquet(cache, index=False)
            stamp.write_text(json.dumps({'source': source, 'sha256': checksum}))
        frame = pd.read_parquet(cache)
        feature = frame.drop(columns=['總價元'], errors='ignore')
        audit[name] = {
            'rows': len(frame), 'columns': list(frame),
            'sha256': checksum,
            'missing_fraction': feature.isna().mean().round(4).to_dict(),
            'unique': feature.nunique().to_dict(),
            'year_counts': {str(k): int(v) for k, v in pd.to_numeric(feature['交易年月日'], errors='coerce').floordiv(10000).value_counts().sort_index().items()},
            'district_counts': feature['鄉鎮市區'].value_counts().head(15).to_dict(),
        }
        if name == 'train':
            y = pd.to_numeric(frame['總價元'], errors='coerce')
            audit[name]['target_quantiles'] = y.quantile([0,.001,.01,.1,.5,.9,.99,.999,1]).to_dict()
            audit[name]['target_invalid'] = int((y.isna() | (y <= 0)).sum())
        print(name, json.dumps(audit[name], ensure_ascii=False), flush=True)
    out = DATA.parent / 'artifacts'
    out.mkdir(exist_ok=True)
    (out / 'data_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
