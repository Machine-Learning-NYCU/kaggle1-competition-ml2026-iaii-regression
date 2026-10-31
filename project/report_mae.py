"""Report the measured MAE results, without treating reused data as pristine."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from train_mae import ROOT


def main():
    out = ROOT/'artifacts/run_mae_v1'
    frozen = json.loads((out/'frozen_blend.json').read_text())
    evaluation = json.loads((out/'evaluation.json').read_text())
    manifest = json.loads((out/'submission_manifest.json').read_text())
    verify = json.loads((out/'verification.json').read_text())
    lines = ['# 原價 L1 重訓試驗：未採用為最終候選', '',
             '最終選定提交檔為專案根目錄 `submission.csv`，比較與理由見 `artifacts/mae_final/RESULTS.md`。本報告保存校準改善、時間泛化反而變差的重訓試驗。', '',
             '官方評分經使用者提供確認為「總價元的 mean absolute error」。本版本的訓練損失、early stopping 與集成都針對原始總價尺度的 L1 誤差。', '',
             '## 實測比較', '', '| 資料 | 前版 RMSLE 集成的 MAE（元） | 新版 MAE（元） | MAE 相對下降 |',
             '|---|---:|---:|---:|']
    old_cal, new_cal = frozen['legacy_calibration']['mae'], frozen['calibration']['mae']
    lines.append(f'| 2024/01–04 校準／選模資料 | {old_cal:,.0f} | {new_cal:,.0f} | {1-new_cal/old_cal:.2%} |')
    for name, row in evaluation['evaluations'].items():
        lines.append(f"| {name} | {row['legacy']['mae']:,.0f} | {row['mae_ensemble']['mae']:,.0f} | {row['mae_improvement_fraction']:.2%} |")
    lines += ['', '**評估限制：這些歷史資料已在上一輪工作中使用過。** 本次修改由官方 MAE 規則驅動，集成權重僅在 1–4 月校準，沒有使用 5–8 月或官方 valid 重新調權重；仍不能把它們稱為新的完全獨立測試集。Kaggle 測試標籤未取得，尚未有榜單分數。', '',
              '## 各模型與權重', '', '| 模型 | 迭代數 | 凍結權重 | 時間回測 MAE（元） | 官方 valid MAE（元） |',
              '|---|---:|---:|---:|---:|']
    for name, weight in zip(frozen['models'], frozen['weights']):
        n = frozen['records'][name]['iterations'] if name in frozen['records'] else '沿用固定模型'
        temporal = evaluation['evaluations']['temporal_2024_05_to_08']['base_models'][name]['mae']
        official = evaluation['evaluations']['official_valid']['base_models'][name]['mae']
        lines.append(f'| {name} | {n} | {weight:.5f} | {temporal:,.0f} | {official:,.0f} |')
    pilot = ROOT/'artifacts/loglink_mae_experiment/calibration.json'
    if pilot.exists():
        pilot_record = json.loads(pilot.read_text())
        lines += ['', f"另做一次正值 log-link MAE 分支試驗，校準 MAE 為 {pilot_record['calibration']['mae']:,.0f} 元，高於舊版 {old_cal:,.0f} 元，因此未納入候選集成。該試驗未讀取後續時間回測結果。這也顯示損失函數對齊並不保證單模型立即優於舊版。"]
    lines += ['', '## 未見建案與不確定性', '', '| 評估 | 群組 | 筆數 | 前版 MAE（元） | 新版 MAE（元） |', '|---|---|---:|---:|---:|']
    for name, row in evaluation['evaluations'].items():
        for group, s in row['slices'].items():
            lines.append(f"| {name} | {group} | {s['rows']:,} | {s['legacy']['mae']:,.0f} | {s['mae_ensemble']['mae']:,.0f} |")
    for name, row in evaluation['evaluations'].items():
        ci = row['uncertainty']['mae_95pct']
        delta = row['uncertainty']['improvement_over_legacy_95pct']
        lines += ['', f"{name}：建案／建物群組 bootstrap 的 MAE 95% 區間為 [{ci[0]:,.0f}, {ci[1]:,.0f}] 元；相對前版 MAE 改善量的配對 95% 區間為 [{delta[0]:,.0f}, {delta[1]:,.0f}] 元。正值代表新版較好。這些區間不包含模型選擇偏差或未來市場漂移。"]
    lines += ['', '## 為何比例分支仍在優化總價 MAE', '',
              '令 B(x,t)>0 為以過去成交、面積及階層行情形成的基準總價，S=10,000,000。比例目標 r=y/B，樣本權重 w=B/S，預測總價為 B·f(x)。因此每一筆訓練資料都滿足：', '',
              '```text\nw · |r − f(x)| = (B/S) · |y/B − f(x)| = |y − B·f(x)| / S\n```', '',
              '這是代數恆等式，而非 log-MAE 近似。除正值輸出限制外，最小化此加權 L1 與最小化原總價 L1 等價。加法分支以 (y−B)/S 為目標，也保留相同等價關係。固定資料集的加權損失分母只造成常數倍，不改變 early-stopping 的排序。', '',
              '集成在原價空間求解：mean(|y−Pw|)/scale + 0.02·||w−uniform||₁，限制 w≥0、Σw=1。使用稀疏線性規劃 HiGHS 求全域最優解；沒有用 test 或官方 valid 的標籤求權重。', '',
              '模型依序以截至 2023/09 的資料訓練、2023/10–12 early stopping、截至 2023/12 固定迭代數重訓；2024/01–04 校準權重。正式模型再以全部 455,776 筆有效 train 重訓，官方 valid 未加入訓練。', '',
              '## 提交與重現', '',
              '- 本重訓試驗候選：專案根目錄 `submission_mae.csv`（與此資料夾的同名檔一致）；正式選定檔案為 `submission.csv`。',
              f"- {manifest['rows']:,} 筆，精確欄名 `編號,總價元`，ID 順序與官方範本一致，金額取最接近整元。",
              f"- 保存模型重現檢查：{verify['saved_model_reproduction_exact']}；原檔與重現檔完全一致。",
              f"- SHA-256：`{manifest['sha256']}`。",
              '- 金額整數化最多改變每筆 0.5 元；報告採整數化前的模型預測計算 MAE。',
              '- 沒有外部答案資料，也尚未提交 Kaggle；課程模型限制與帳號授權仍待確認。', '',
              '```bash\n.venv/bin/python train_mae.py\n.venv/bin/python predict_mae.py\n.venv/bin/python verify_mae.py\n.venv/bin/python report_mae.py\n```', '',
              '使用套件的損失定義見 [CatBoost 回歸目標文件](https://catboost.ai/docs/en/concepts/loss-functions-regression) 與 [LightGBM 參數文件](https://lightgbm.readthedocs.io/en/stable/Parameters.html)。']
    (out/'RESULTS_MAE.md').write_text('\n'.join(lines)+'\n')
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    entries = list(evaluation['evaluations'].values())
    pos = np.arange(2)
    ax[0].bar(pos-.18, [e['legacy']['mae']/1e6 for e in entries], .36, label='Previous ensemble', color='#9daeb8')
    ax[0].bar(pos+.18, [e['mae_ensemble']['mae']/1e6 for e in entries], .36, label='MAE ensemble', color='#117e91')
    ax[0].set(xticks=pos, xticklabels=['Temporal backtest', 'Official validation'], ylabel='MAE (million NTD)', title='Original-price absolute error')
    ax[0].legend()
    t = evaluation['evaluations']['temporal_2024_05_to_08']['slices']
    ax[1].bar(pos-.18, [t[k]['legacy']['mae']/1e6 for k in ['seen_group', 'unseen_group']], .36, color='#9daeb8')
    ax[1].bar(pos+.18, [t[k]['mae_ensemble']['mae']/1e6 for k in ['seen_group', 'unseen_group']], .36, color='#117e91')
    ax[1].set(xticks=pos, xticklabels=['Seen projects', 'Unseen projects'], ylabel='MAE (million NTD)', title='Temporal backtest by project coverage')
    fig.suptitle('MAE-aligned models | Historical evaluations reused', fontsize=14)
    fig.savefig(out/'mae_comparison.png', dpi=160)
    fig.savefig(out/'mae_comparison.pdf')
    print((out/'RESULTS_MAE.md').read_text())


if __name__ == '__main__':
    main()
