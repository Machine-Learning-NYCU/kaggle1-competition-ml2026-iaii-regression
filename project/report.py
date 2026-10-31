"""Export the actual locked evaluation, matched ablation, and a standalone figure."""
import json
import hashlib
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from train import blend


def main():
    root = Path(__file__).resolve().parent
    out = root / 'artifacts/run_v2'
    evaluation = json.loads((out / 'evaluation.json').read_text())
    frozen = json.loads((out / 'frozen_blend.json').read_text())
    manifest = json.loads((out / 'submission_manifest.json').read_text())
    ablation = json.loads((out / 'ablation.json').read_text())
    # Export the already-fitted experts without any additional fitting or selection.
    base_dir = out / 'base_submissions'
    base_dir.mkdir(exist_ok=True)
    base_values = np.load(out / 'test_base_predictions.npz')['predictions']
    template = pd.read_csv(root / 'data/test-reindex-test-sample-submission-v2.csv')
    base_manifest = {}
    for i, name in enumerate(frozen['models']):
        frame = template.copy()
        frame['總價元'] = base_values[:, i]
        assert len(frame) == 47073 and np.isfinite(frame['總價元']).all() and (frame['總價元'] > 0).all()
        path = base_dir / f'{name}.csv'
        frame.to_csv(path, index=False, encoding='utf-8-sig')
        base_manifest[name] = {'rows': len(frame), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    (base_dir / 'manifest.json').write_text(json.dumps(base_manifest, indent=2))
    model = joblib.load(out / 'final_cat_total.joblib')['model']
    importance = pd.DataFrame({'feature': model.feature_names_, 'importance': model.feature_importances_}).sort_values('importance', ascending=False)
    importance.to_csv(out / 'feature_importance.csv', index=False)
    lines = ['# 實測報告', '', '下列數字來自實際訓練及固定設定後的評估；不是 Kaggle 榜單分數。', '',
             '## 固定模型的結果', '', '| 評估資料 | 候選 | RMSLE | RMSE（元） | MAE（元） | MAPE | 誤差≤10%比例 |',
             '|---|---|---:|---:|---:|---:|---:|']
    for label, entry in evaluation['evaluations'].items():
        for metric, result in entry['blends'].items():
            s = result['all']
            lines.append(f"| {label} | {metric} | {s['rmsle']:.5f} | {s['rmse']:,.0f} | {s['mae']:,.0f} | {s['mape']:.2%} | {s['within_10pct']:.2%} |")
    lines += ['', '## 單模型比較', '', '| 模型 | 時間回測 RMSLE | 時間回測 RMSE（元） | 官方驗證 RMSLE |', '|---|---:|---:|---:|']
    for name in frozen['models']:
        temporal = evaluation['evaluations']['future_2024_05_to_08']['base_models'][name]
        official = evaluation['evaluations']['official_valid']['base_models'][name]
        lines.append(f"| {name} | {temporal['rmsle']:.5f} | {temporal['rmse']:,.0f} | {official['rmsle']:.5f} |")
    lines += ['', '集成並非每個指標都勝過單模型：在本次時間回測，LightGBM 行情殘差分支的 RMSE 低於凍結的 RMSE 集成。這反映大型交易與校準時段的分布差異，不能只展示有利指標。原來的集成權重維持不變。',
              '各單模型候選另存於 `base_submissions/`。若依目前回測挑選其中一個模型，這段回測即為模型選擇資料，不能再當作獨立最終評估。']
    lines += ['', '## 新建案與群組相依', '', '| 評估資料 | 已見群組 RMSLE | 未見群組 RMSLE | 未見比例 | 群組 bootstrap 95% RMSLE 區間 |',
              '|---|---:|---:|---:|---|']
    for label, entry in evaluation['evaluations'].items():
        s = entry['blends']['rmsle']
        ci = s['cluster_bootstrap_95pct']
        lines.append(f"| {label} | {s['slices']['seen_group']['rmsle']:.5f} | {s['slices']['unseen_group']['rmsle']:.5f} | {entry['unseen_group_fraction']:.2%} | [{ci[0]:.5f}, {ci[1]:.5f}] |")
    lines += ['', f"測試集相對全部訓練資料的未見建案／建物群組比例：**{manifest['test_new_group_fraction']:.2%}**。", '',
              '這裡的群組為行政區＋建案；沒有建案名稱時退回地址建物。群組 bootstrap 量化此回測資料上的抽樣變異，沒有涵蓋未來市場制度變動，亦非測試集保證區間。', '',
              '## 固定參數消融', '', '同樣的 CatBoost 總價分支、訓練交易、seed、深度與迭代數，僅移除行情特徵；此比較不改變已凍結的提交模型。', '',
              '| 資料 | 不含行情 RMSLE | 含行情 RMSLE | 相對下降 |', '|---|---:|---:|---:|']
    for label, result in ablation['evaluations'].items():
        old, new = result['without_market']['rmsle'], result['with_market']['rmsle']
        lines.append(f'| {label} | {old:.5f} | {new:.5f} | {1-new/old:.2%} |')
    lines += ['', '## 凍結的權重', '', '| 模型 | 迭代數 | RMSLE 權重 | RMSE 權重 |', '|---|---:|---:|---:|']
    for i, name in enumerate(frozen['models']):
        lines.append(f"| {name} | {frozen['records'][name]['iterations']} | {frozen['weights']['rmsle'][i]:.4f} | {frozen['weights']['rmse'][i]:.4f} |")
    lines += ['', '## 產物與限制', '',
              '- 兩份候選提交檔均為 47,073 列，欄位及 ID 順序與官方範本一致，預測均為有限正數。',
              '- 官方驗證標籤未加入訓練。時間回測的模型只用到 2023 年底；最終提交模型固定參數後以全部有效訓練資料重訓。',
              '- calibration 分數含權重選擇偏差，不能作為獨立泛化估計。',
              '- v1 消融曾評估同一時間回測區間；v2 僅修正道路正規表示式解析錯誤，模型架構、超參數及時間切分不變。因此此區間是修正後的確認回測，不能宣稱是從未看過的全新測試集。v1 已於官方 valid 評估前停止，官方驗證仍只在 v2 決策凍結後使用。',
              '- 官方驗證與時間回測代表不同分布；應優先參考時間回測，但它仍未覆蓋全部測試年份。',
              '- CatBoost GPU 重新訓練可能有微小差異；保存模型推論及 CSV 已可重現。',
              '- 官方評分公式、課程規則、Kaggle 登入與榜單尚未取得。尚未提交，不能聲稱第一名。', '',
              '完整架構、理論動機、驗證方式與重現命令見專案 README.md。']
    (out / 'RESULTS.md').write_text('\n'.join(lines) + '\n')

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    labels = ['Temporal holdout', 'Official validation']
    entries = list(evaluation['evaluations'].values())
    pos = np.arange(2)
    axes[0, 0].bar(pos-.17, [e['blends']['rmsle']['slices']['seen_group']['rmsle'] for e in entries], .34, label='Seen projects', color='#167d9a')
    axes[0, 0].bar(pos+.17, [e['blends']['rmsle']['slices']['unseen_group']['rmsle'] for e in entries], .34, label='Unseen projects', color='#e79638')
    axes[0, 0].set(xticks=pos, xticklabels=labels, ylabel='RMSLE (lower is better)', title='Generalization to new projects')
    axes[0, 0].legend()
    a = ablation['evaluations']
    axes[0, 1].bar(pos-.17, [a[k]['without_market']['rmsle'] for k in a], .34, label='No causal market', color='#90a4ae')
    axes[0, 1].bar(pos+.17, [a[k]['with_market']['rmsle'] for k in a], .34, label='Causal market', color='#167d9a')
    axes[0, 1].set(xticks=pos, xticklabels=['Calibration', 'Temporal holdout'], ylabel='RMSLE', title='Matched CatBoost ablation')
    axes[0, 1].legend()
    archive = np.load(out / 'future_2024_05_to_08_predictions.npz', allow_pickle=False)
    y = archive['y']
    p = blend(archive['predictions'], frozen['weights']['rmsle'], 'rmsle')
    squared = (p - y) ** 2
    tail = {'top_2_rows_share_of_squared_error': float(np.sort(squared)[-2:].sum() / squared.sum()),
            'transactions_above_100m_ntd': int((y > 1e8).sum()),
            'rows': len(y), 'note': 'Diagnostics only; all rows remain in the reported primary metrics.'}
    (out / 'tail_diagnostics.json').write_text(json.dumps(tail, indent=2))
    with (out / 'RESULTS.md').open('a') as f:
        f.write(f"\n時間回測中有 {tail['transactions_above_100m_ntd']} 筆總價超過一億元。RMSLE 候選最大的兩筆平方誤差占整體平方誤差 {tail['top_2_rows_share_of_squared_error']:.2%}；主要成績保留全部交易，沒有刪除這些案例來美化 RMSE。\n")
    axes[1, 0].hexbin(np.log10(y), np.log10(p), gridsize=60, bins='log', mincnt=1, cmap='viridis')
    limits = [min(np.log10(y).min(), np.log10(p).min()), max(np.log10(y).max(), np.log10(p).max())]
    axes[1, 0].plot(limits, limits, color='#d34b4b', ls='--')
    axes[1, 0].set(xlabel='Observed price (log10 NTD)', ylabel='Prediction (log10 NTD)', title='Forward temporal backtest')
    top = importance.head(12).iloc[::-1]
    axes[1, 1].barh(top.feature, top.importance, color='#167d9a')
    axes[1, 1].set(xlabel='CatBoost feature importance', title='Learned signals (not causal effects)')
    fig.suptitle('Taiwan housing: temporal validation and leakage controls', fontsize=16)
    fig.savefig(out / 'validation.png', dpi=160)
    fig.savefig(out / 'validation.pdf')
    print((out / 'RESULTS.md').read_text())


if __name__ == '__main__':
    main()
