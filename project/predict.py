"""Regenerate predictions from saved models; no fitting or label access."""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from features import ID, TARGET, make_features
from train import blend, predict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', default='artifacts/run_v2')
    parser.add_argument('--input', default='data/test.parquet')
    parser.add_argument('--template', default='data/test-reindex-test-sample-submission-v2.csv')
    parser.add_argument('--metric', choices=['rmsle', 'rmse'], default='rmsle')
    parser.add_argument('--output', default='artifacts/submission_reproduced.csv')
    args = parser.parse_args()
    source = Path(args.input)
    if source.suffix == '.parquet':
        frame = pd.read_parquet(source)
    elif source.suffix == '.xlsx':
        frame = pd.read_excel(source, engine='calamine')
    else:
        frame = pd.read_csv(source)
    model_dir = Path(args.model_dir)
    config = json.loads((model_dir / 'config.json').read_text())
    root = Path(__file__).resolve().parent
    for filename, checksum in config['code_sha256'].items():
        if hashlib.sha256((root / filename).read_bytes()).hexdigest() != checksum:
            raise ValueError(f'Training code mismatch for {filename}; use the matching run/model version')
    frozen = json.loads((model_dir / 'frozen_blend.json').read_text())
    x, _, _, offset = make_features(frame)
    x = joblib.load(model_dir / 'market_final.joblib').transform(x)
    matrix = np.column_stack([predict(joblib.load(model_dir / f'final_{name}.joblib'), x, offset)
                              for name in frozen['models']])
    values = blend(matrix, frozen['weights'][args.metric], args.metric)
    template = pd.read_csv(args.template)
    if list(template) != [ID, TARGET] or template[ID].astype(str).tolist() != frame[ID].astype(str).tolist():
        raise ValueError('Template schema or ordered identifiers do not match input')
    if not (np.isfinite(values).all() and (values > 0).all()):
        raise ValueError('Invalid predictions')
    template[TARGET] = values
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(path, index=False, encoding='utf-8-sig')
    print(f'Saved {len(template)} rows to {path}')


if __name__ == '__main__':
    main()
