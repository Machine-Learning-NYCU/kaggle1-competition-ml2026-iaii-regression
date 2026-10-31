"""Selected submission: existing experts, price-MAE weights, one scale parameter.

All numeric calibration uses Jan-Apr 2024 only. The pipeline choice follows
reused historical validation, so this is not an untouched final-test estimate.
"""
import json
from pathlib import Path
import shutil

import joblib
import numpy as np
import pandas as pd

from features import make_features
from mae_models import learn_mae_blend
from train import scores, predict, blend
from train_mae import ROOT, LEGACY, checksum, dump, clustered_comparison, export_submission

OUT = ROOT/'artifacts/mae_final'


def optimal_price_scale(y, prediction, bounds=(.95, 1.05)):
    """Minimize sum |y-a*p|: a is the p-weighted median of y/p."""
    y, p = np.asarray(y, float), np.asarray(prediction, float)
    if not (np.isfinite(y).all() and np.isfinite(p).all() and (p > 0).all()):
        raise ValueError('Scale calibration requires positive finite predictions')
    ratio = y/p
    order = np.argsort(ratio, kind='stable')
    index = np.searchsorted(np.cumsum(p[order]), .5*p.sum())
    return float(np.clip(ratio[order[index]], *bounds))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    old_frozen = json.loads((LEGACY/'frozen_blend.json').read_text())
    config = {'official_metric': 'MAE in NTD', 'calibration_period': '2024-01 through 2024-04',
              'penalty': .02, 'scale_bounds': [.95, 1.05], 'models': old_frozen['models'],
              'selection_evaluation_is_reused': True,
              'choice': 'MAE-calibrated previous experts; new L1 fits failed temporal generalization.',
              'source_sha256': {f: checksum(ROOT/f) for f in ['features.py','market.py','train.py','mae_models.py','train_mae.py','finalize_mae.py']},
              'calibration_sha256': checksum(LEGACY/'calibration_predictions.npz'),
              'model_sha256': {name: checksum(LEGACY/f'final_{name}.joblib') for name in old_frozen['models']},
              'market_sha256': checksum(LEGACY/'market_final.joblib')}
    if (OUT/'config.json').exists() and json.loads((OUT/'config.json').read_text()) != config:
        raise ValueError('Selected pipeline changed; use a new output directory')
    dump(OUT/'config.json', config)
    if not (OUT/'calibration.json').exists():
        cal = np.load(LEGACY/'calibration_predictions.npz')
        w, optimization = learn_mae_blend(cal['predictions'], cal['y'], penalty=.02)
        p = cal['predictions']@w
        scale = optimal_price_scale(cal['y'], p)
        dump(OUT/'calibration.json', {'weights': w.tolist(), 'scale': scale, 'optimization': optimization,
             'calibration': scores(cal['y'], scale*p), 'labels_used': '2024-01 through 2024-04 only'})
    fitted = json.loads((OUT/'calibration.json').read_text())
    w, scale = np.array(fitted['weights']), fitted['scale']
    report = {'metric': 'MAE', 'model_selection_on_reused_evaluation': True, 'evaluations': {}}
    for name in ['future_2024_05_to_08', 'official_valid']:
        archive = np.load(LEGACY/f'{name}_predictions.npz', allow_pickle=True)  # trusted, locally generated artifact
        y, matrix, groups = archive['y'], archive['predictions'], archive['groups']
        old = blend(matrix, old_frozen['weights']['rmsle'], 'rmsle')
        selected = scale*(matrix@w)
        row = {'previous_rmsle_ensemble': scores(y, old),
               'previous_rmse_ensemble': scores(y, blend(matrix, old_frozen['weights']['rmse'], 'rmse')),
               'mae_selected': scores(y, selected),
               'relative_improvement_vs_previous_main': float(1-np.mean(abs(y-selected))/np.mean(abs(y-old))),
               'uncertainty': clustered_comparison(y, selected, old, groups)}
        report['evaluations'][name] = row
        np.savez_compressed(OUT/f'{name}_predictions.npz', y=y, predictions=selected)
    dump(OUT/'evaluation.json', report)

    # Recompute predictions from actual fitted models, not just a cached submission.
    raw = pd.read_parquet(ROOT/'data/test.parquet')
    x, _, _, offset = make_features(raw)
    x = joblib.load(LEGACY/'market_final.joblib').transform(x)
    matrix = np.column_stack([predict(joblib.load(LEGACY/f'final_{name}.joblib'), x, offset)
                              for name in old_frozen['models']])
    cached = np.load(LEGACY/'test_base_predictions.npz')['predictions']
    np.testing.assert_allclose(matrix, cached, rtol=1e-12, atol=1e-6)
    prediction = scale*(matrix@w)
    saved = scale*(cached@w)
    np.testing.assert_array_equal(np.rint(prediction), np.rint(saved))
    frame = export_submission(raw, prediction, OUT/'submission.csv')
    shutil.copy2(OUT/'submission.csv', ROOT/'submission.csv')
    assert (ROOT/'submission.csv').read_text().splitlines()[0] == '編號,總價元'
    assert frame['編號'].is_unique and len(frame) == 47073
    verification = {'rows': len(frame), 'columns': list(frame), 'ordered_ids_match': True,
                    'positive_finite_integer_prices': bool((frame['總價元'] > 0).all()),
                    'recomputed_model_predictions_match_cache': True, 'rounded_predictions_match_exactly': True,
                    'sha256': checksum(OUT/'submission.csv'), 'submitted_to_kaggle': False}
    assert verification['positive_finite_integer_prices']
    assert checksum(ROOT/'submission.csv') == verification['sha256']
    dump(OUT/'verification.json', verification)

    experiment_path = ROOT/'artifacts/run_mae_v1/evaluation.json'
    experiment = json.loads(experiment_path.read_text()) if experiment_path.exists() else None
    lines = ['# 最終 MAE 提交候選', '',
        '正式提交檔：專案根目錄 `submission.csv`。評分為總價元的 MAE，檔案含 47,073 筆，標頭精確為 `編號,總價元`。', '',
        '## 為何選這個版本', '',
        '新訓練的 MAE 分支在 1–4 月校準資料上的集成 MAE 較低，但 5–8 月時間回測變差。因此最終保留較穩定的既有四個專家，重新以 MAE 求解非負集成權重，再加入一個受限的價格尺度校準參數。沒有因損失函數名稱改為 MAE 就自動採用泛化較差的模型。', '',
        '| 歷史評估資料 | 上版主模型 MAE（元） | 最終候選 MAE（元） | 下降比例 |', '|---|---:|---:|---:|']
    for name, row in report['evaluations'].items():
        lines.append(f"| {name} | {row['previous_rmsle_ensemble']['mae']:,.0f} | {row['mae_selected']['mae']:,.0f} | {row['relative_improvement_vs_previous_main']:.2%} |")
    if experiment:
        lines += ['', '新 L1 重訓集成（未採用）亦完整保留：']
        for name, row in experiment['evaluations'].items():
            lines.append(f"- {name}：MAE {row['mae_ensemble']['mae']:,.0f} 元。")
    lines += ['', '先前 RMSE 候選的時間回測 MAE 為 ' + f"{report['evaluations']['future_2024_05_to_08']['previous_rmse_ensemble']['mae']:,.0f}" + ' 元，也列入比較；上表基準明確指之前主要的 RMSLE 集成。', '',
        '## 校準與 overfitting 控制', '',
        '- 權重與尺度只使用 2024/01–04 的預測及標籤求解。各基礎模型當時只以截至 2023/12 的資料訓練。',
        '- 權重非負、總和為 1；MAE 線性規劃含 0.02 的等權先驗收縮懲罰，限制權重自由度。',
        '- 全域尺度 a 取 y/p 的 p 加權中位數，等價於最小化 Σ|y−a·p|；限制 a∈[0.95,1.05]。',
        f'- 校準尺度為 {scale:.10f}。沒有按測試 ID 修改個別價格。',
        '- 時間與官方驗證集已被前一輪使用，又參與本次候選比較，所以它們是歷史選模／確認結果，不能宣稱是完全獨立的最終泛化估計。',
        '- 官方 valid 未加入基礎模型訓練或數值權重校準；所有測試標籤仍未知。', '',
        '| 專家 | MAE 權重 |', '|---|---:|']
    for name, value in zip(old_frozen['models'], w):
        lines.append(f'| {name} | {value:.8f} |')
    for name, row in report['evaluations'].items():
        ci = row['uncertainty']['improvement_over_legacy_95pct']
        lines += ['', f"{name}：建案群組 bootstrap 的 MAE 改善量 95% 區間為 [{ci[0]:,.0f}, {ci[1]:,.0f}] 元。正值代表改善；區間不涵蓋模型選擇偏差與未來市場漂移。"]
    lines += ['', '## 重現', '',
        '```bash\n.venv/bin/python -m pytest -q\n.venv/bin/python finalize_mae.py\n```', '',
        '已從保存模型重新推論，與快取預測一致；整元提交檔逐列相同。各模型仍保存於 `artifacts/run_v2/`。重訓 L1 的實驗程式為 `train_mae.py`；它產生的 `submission_mae.csv` 是比較用實驗候選，正式候選是本程式產生的 `submission.csv`。', '',
        '尚未送出 Kaggle，沒有榜單成績，也不能保證第一名。']
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print((OUT/'RESULTS.md').read_text())


if __name__ == '__main__':
    main()
