import json
from pathlib import Path

import numpy as np
import pandas as pd

from train_mae import ROOT, checksum, dump


def main():
    out = ROOT/'artifacts/run_mae_v1'
    original = pd.read_csv(out/'submission_mae.csv')
    template = pd.read_csv(ROOT/'data/test-reindex-test-sample-submission-v2.csv')
    reproduced = pd.read_csv(ROOT/'artifacts/submission_mae_reproduced.csv')
    manifest = json.loads((out/'submission_manifest.json').read_text())
    assert list(original) == ['編號', '總價元']
    assert original.columns.tolist() == template.columns.tolist()
    assert len(original) == 47073
    assert original['編號'].equals(template['編號']) and original['編號'].is_unique
    assert np.isfinite(original['總價元']).all() and (original['總價元'] > 0).all()
    assert pd.api.types.is_integer_dtype(original['總價元'])
    pd.testing.assert_frame_equal(original, reproduced)
    assert checksum(out/'submission_mae.csv') == manifest['sha256'] == checksum(ROOT/'submission_mae.csv')
    # Read bytes as well as pandas, checking the literal header and absence of an index column.
    assert (out/'submission_mae.csv').read_text(encoding='utf-8').splitlines()[0] == '編號,總價元'
    result = {'rows': len(original), 'schema': list(original), 'ordered_ids_match': True,
              'positive_finite_integer_prices': True, 'saved_model_reproduction_exact': True,
              'root_submission_matches': True, 'sha256': manifest['sha256']}
    dump(out/'verification.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
