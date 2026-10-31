"""Validate submission contract and saved-model inference reproducibility."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    root = Path(__file__).resolve().parent
    out = root / 'artifacts/run_v2'
    template = pd.read_csv(root / 'data/test-reindex-test-sample-submission-v2.csv')
    manifest = json.loads((out / 'submission_manifest.json').read_text())
    checks = {}
    for name, checksum in manifest['files'].items():
        path = out / name
        frame = pd.read_csv(path)
        assert list(frame) == list(template), 'Wrong columns'
        assert len(frame) == 47073, 'Wrong row count'
        assert frame['編號'].equals(template['編號']), 'Identifier order mismatch'
        assert frame['編號'].is_unique, 'Duplicated identifiers'
        assert np.isfinite(frame['總價元']).all() and (frame['總價元'] > 0).all(), 'Invalid predictions'
        assert frame['總價元'].nunique() > 1, 'Constant placeholder predictions'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == checksum, 'File modified after export'
        checks[name] = {'rows': len(frame), 'positive_finite': True, 'ordered_ids_match': True,
                        'sha256_match': True, 'min_prediction': float(frame['總價元'].min()),
                        'max_prediction': float(frame['總價元'].max())}
    regenerated = pd.read_csv(root / 'artifacts/submission_reproduced.csv')
    original = pd.read_csv(out / 'submission_rmsle.csv')
    assert regenerated['編號'].equals(original['編號'])
    np.testing.assert_allclose(regenerated['總價元'], original['總價元'], rtol=1e-12, atol=1e-6)
    checks['saved_model_reproduction'] = {'passed': True,
        'max_absolute_difference_ntd': float(np.max(np.abs(regenerated['總價元'] - original['總價元'])))}
    assert not manifest['official_valid_used_for_training']
    checks['official_valid_excluded_from_training'] = True
    (out / 'verification.json').write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
