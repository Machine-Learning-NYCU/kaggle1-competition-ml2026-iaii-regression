"""Audit saved artifacts without fitting or changing the frozen experiment.

The explicit ProjectIdentityMasker import also resolves models saved by the
training command under __main__; no training is run while loading the model.
"""
import json

import joblib
import numpy as np

from train_generalization import (ROOT, OUT, SEED, SPECS, WINDOWS,
    ProjectIdentityMasker, project_keys, cold_mask, inputs, baseline, predict,
    pred_path, model_path, choose, metrics)
from train_mae import checksum, dump


def main():
    config = json.loads((OUT/'config.json').read_text())
    for source, expected in config['source_sha256'].items():
        assert checksum(ROOT/source) == expected, source
        assert checksum(OUT/'source'/source) == expected, source
    for name, expected in config['input_sha256'].items():
        assert checksum(ROOT/name) == expected, name
    selection = json.loads((OUT/'selection.json').read_text())
    assert checksum(OUT/'config.json') == selection['config_sha256']
    assert choose(selection['grid']) == selection['selected']
    checks = []
    for fold in WINDOWS:
        d = inputs(fold)
        cold = cold_mask(d)
        fitting_groups = set(project_keys(d['x']))
        validation_groups = set(project_keys(d['vx'].loc[cold]))
        assert fitting_groups.isdisjoint(validation_groups)
        assert not d['seen'][cold].any()
        assert (d['vx'].loc[cold, 'sem_relocation'] == 0).all()
        rng = np.random.default_rng(SEED)
        ix = rng.choice(len(d['vx']), min(512, len(d['vx'])), replace=False)
        for name in SPECS:
            model = joblib.load(model_path(fold, name, SEED))
            cached = np.load(pred_path(fold, name, SEED))
            repeated = predict(model, d['vx'].iloc[ix], d['vbase'][ix])
            np.testing.assert_allclose(repeated, cached[ix], rtol=1e-12, atol=1e-7)
            order = rng.permutation(len(ix))
            shuffled = predict(model, d['vx'].iloc[ix[order]], d['vbase'][ix[order]])
            np.testing.assert_allclose(shuffled, repeated[order], rtol=1e-12, atol=1e-7)
            saved = json.loads((OUT/fold/f'{name}_{SEED}.json').read_text())
            measured = metrics(d['vy'], baseline(fold), cached, cold, ~d['tune'])
            assert measured == saved['standalone']
            checks.append(dict(fold=fold, name=name, sample_rows=len(ix),
                max_prediction_difference=float(abs(repeated-cached[ix]).max()),
                row_order_invariant=True,
                model_sha256=checksum(model_path(fold, name, SEED)),
                predictions_sha256=checksum(pred_path(fold, name, SEED))))
    dump(OUT/'audit_verification.json', dict(
        source_and_input_hashes_unchanged=True,
        fitting_and_cold_validation_project_keys_disjoint=True,
        frozen_selection_recomputed_identically=True,
        no_model_refitting=True, no_submission_changes=True,
        checks=checks,
        limitation='Saved-model sample inference checked against cached historical features; this is not a new independent validation dataset.'))
    print(json.dumps(dict(verified_models=len(checks),
        v6_sha256=checksum(ROOT/'submission1017_v6.csv'),
        result=str(OUT/'audit_verification.json')), ensure_ascii=False))


if __name__ == '__main__':
    main()
