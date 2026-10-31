"""Regenerate the MAE submission from saved models, without training labels."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from features import make_features
from mae_models import predict
from train import predict as old_predict, blend as old_blend
from train_mae import ROOT, LEGACY, checksum, export_submission


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', default='artifacts/run_mae_v1')
    parser.add_argument('--input', default='data/test.parquet')
    parser.add_argument('--output', default='artifacts/submission_mae_reproduced.csv')
    args = parser.parse_args()
    model_dir = ROOT/args.model_dir
    config = json.loads((model_dir/'config.json').read_text())
    for name, sha in config['source_sha256'].items():
        if checksum(ROOT/name) != sha:
            raise ValueError(f'Incompatible source version: {name}')
    frozen = json.loads((model_dir/'frozen_blend.json').read_text())
    source = ROOT/args.input
    raw = pd.read_parquet(source) if source.suffix == '.parquet' else pd.read_excel(source, engine='calamine')
    x, _, _, offset = make_features(raw)
    x = joblib.load(model_dir/'market_final.joblib').transform(x)
    matrix = []
    for name in frozen['models']:
        if name == 'legacy_rmsle':
            old_frozen = json.loads((LEGACY/'frozen_blend.json').read_text())
            old_matrix = np.column_stack([old_predict(joblib.load(LEGACY/f'final_{n}.joblib'), x, offset)
                                          for n in old_frozen['models']])
            matrix.append(old_blend(old_matrix, old_frozen['weights']['rmsle'], 'rmsle'))
        else:
            matrix.append(predict(joblib.load(model_dir/f'final_{name}.joblib'), x, offset))
    prediction = np.column_stack(matrix)@np.asarray(frozen['weights'])
    output = ROOT/args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    export_submission(raw, prediction, output)
    print(f'Saved {len(raw)} rows: {output}')


if __name__ == '__main__':
    main()
