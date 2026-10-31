"""Describe the frozen study; never train, select a model, or change predictions."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from train_generalization import OUT, ROOT, SPECS, WINDOWS, SEED


def read(path):
    return json.loads(path.read_text())


def main():
    selection = read(OUT / 'selection.json')
    config = read(OUT / 'config.json')
    scope = {r['fold']: r for r in read(OUT / 'scope.json')}
    rows = []
    ablations = []
    burden = []
    for fold in WINDOWS:
        audit = pd.read_parquet(OUT / f'{fold}_error_audit.parquet')
        cold = ~audit.seen.to_numpy(bool)
        burden.append(dict(fold=fold, rows=len(audit), unseen_rows=int(cold.sum()),
            unseen_mae=float(audit.loc[cold, 'ae'].mean()),
            seen_mae=float(audit.loc[~cold, 'ae'].mean()),
            unseen_error_share=float(audit.loc[cold, 'ae'].sum()/audit.ae.sum())))
        for name in SPECS:
            result = read(OUT / fold / f'{name}_{SEED}.json')
            ablations.append(dict(fold=fold, name=name,
                standalone_cold_mae=result['standalone']['cold']['candidate_mae']))
            for alpha, metrics in result['blends'].items():
                rows.append(dict(fold=fold, name=name, alpha=float(alpha),
                    baseline_cold_mae=metrics['cold']['baseline_mae'],
                    candidate_cold_mae=metrics['cold']['candidate_mae'],
                    cold_gain_pct=100*(1-metrics['cold']['ratio']),
                    late_cold_gain_pct=100*(1-metrics['cold_late']['ratio']),
                    all_mae=metrics['all']['candidate_mae']))
    table = pd.DataFrame(rows)
    table.to_csv(OUT / 'comparison.csv', index=False)
    pd.DataFrame(ablations).to_csv(OUT / 'ablation.csv', index=False)
    (OUT / 'error_burden.json').write_text(json.dumps(burden, ensure_ascii=False, indent=2)+'\n')

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout='constrained')
    labels = ['2021-08 to 2022-08', '2022-08 to 2023-08']
    shares = [r['unseen_error_share']*100 for r in burden]
    axes[0].bar(labels, shares, color=['#4c78a8', '#59a14f'])
    for i, value in enumerate(shares):
        axes[0].text(i, value+1, f'{value:.1f}%', ha='center')
    axes[0].set(ylim=(0, 100), ylabel='Share of total absolute error (%)',
                title='Baseline: projects absent from training')
    keys = [(name, alpha) for name in SPECS for alpha in [.25, .5]]
    xx = np.arange(len(keys))
    for j, fold in enumerate(WINDOWS):
        values = [table.loc[(table.fold == fold) & (table.name == name) &
                    (table.alpha == alpha), 'cold_gain_pct'].iloc[0] for name, alpha in keys]
        axes[1].bar(xx+(j-.5)*.37, values, width=.37, label=labels[j])
    axes[1].axhline(.5, color='black', linestyle='--', linewidth=1, label='Required in both years')
    axes[1].axhline(0, color='grey', linewidth=.8)
    axes[1].set_xticks(xx, [f'{n}\n{a:.0%}' for n, a in keys], rotation=45, ha='right')
    axes[1].set(ylabel='Cold-project MAE reduction (%)', title='Fixed candidates; higher is better')
    axes[1].legend(fontsize=8)
    fig.savefig(OUT / 'generalization_study.png', dpi=180)
    fig.savefig(OUT / 'generalization_study.pdf')
    plt.close(fig)

    lines = ['# 下一個研究目標：一般住宅的新建案泛化', '',
        '本研究以 v6 為目前提交基準。一般住宅沿用 v4；歷史窗口沒有安置住宅，因此以已保存的 v4 歷史預測比較。新分支只使用競賽 train 的標籤，沒有讀取外部逐戶價格，也沒有使用排行榜分數選擇參數。', '',
        '## 誤差來源與作用範圍', '',
        '| 歷史窗口 | 原始名稱未見建案筆數 | 未見建案 MAE（元） | 已見建案 MAE（元） | 未見建案占總絕對誤差 |',
        '|---|---:|---:|---:|---:|']
    for r in burden:
        lines.append(f"| {r['fold']} | {r['unseen_rows']:,} | {r['unseen_mae']:,.0f} | {r['seen_mae']:,.0f} | {r['unseen_error_share']:.2%} |")
    lines += ['',
        '上述誤差占比包含樣本數的影響，並非「模型有 82%–93% 的錯誤率」。實驗採用更嚴格的範圍：原始名稱與正規化建案鍵都必須未出現在 fitting，並排除安置住宅。', '',
        f"嚴格範圍在 stress2021 為 {scope['stress2021']['canonical_cold_nonpolicy']:,} 筆，development 為 {scope['development']['canonical_cold_nonpolicy']:,} 筆。test 可作用於 {scope['final']['canonical_cold_nonpolicy']:,} 筆、{scope['final']['cold_canonical_projects']:,} 個一般住宅新建案；1,408 筆政策分支不在本次改動範圍。", '',
        '## 可檢驗假設與實作', '',
        '假設：模型過度依賴建案身分時，對從未出現的新建案轉移效果較差。訓練時以固定雜湊選取約 35% 的建案，遮蔽該建案所有 fitting 列的 project、canonical_project、project_building_type、address_base、tower、house_number、parcel_number。推論遇到未保留的建案也做相同處理。面積、樓層、格局、建材、備註語義及區位繼續使用。', '',
        '這是部分識別特徵的正則化實驗，不保證建案完全不可被其他屬性辨識；例如建商名稱前綴、道路、地段及建築屬性仍可能形成代理。地址向量只來自城市／行政區／道路／地段，不包含門牌、戶號或建案名；其字典只由該折 fitting 建立。', '',
        '四組固定消融為保留／遮蔽身分 × 對數 RMSE／面積加權 L1。所有組別共用 3,000 輪、深度 8、學習率 0.045、L2 10；沒有在本輪依驗證成績調整樹深、輪數或追加搜尋。訓練及驗證仍以前向時間分割；驗證另限制整個建案未出現在 fitting。遮蔽訓練建案本身不是留出驗證，其標籤仍參與訓練。', '',
        '令 b = 有效面積 × 僅以 fitting 估計的時間指數，L1 分支學習 y/(100000b)，權重為 b/100。因此訓練加權絕對誤差與原總價絕對誤差相差固定尺度／正規化常數。推論沿用固定 0.5 的時間趨勢折減，最後仍以原始總價 MAE 評估。對數分支的目標與權重是既有近似，不宣稱等價 MAE。', '',
        '## 先凍結的採用門檻', '',
        '設定、程式碼與輸入雜湊保存於 config.json；四組各只評估 25% 與 50% 兩個混合權重，僅對嚴格新建案混合。兩個年度的新建案 MAE 均須改善至少 0.5%，且各自後八個月不得退步。採最差年度相對誤差較小者，差距在預定容忍範圍時偏好 25% 的較小權重。', '',
        '通過後才能固定候選進入第二 seed；兩個 seed 各自在兩年均須改善，兩 seed 平均仍須過原門檻。再用第三年度確認至少 0.5% 改善、後八個月不退步、建案群組 bootstrap 改善量 95% 下界大於零。任一步失敗就停止，沒有依確認結果換第二名候選。', '',
        '**這些歷史窗口已在前幾版使用過，並非全新獨立測試集。** 固定候選、跨年度與群組檢查降低過擬合風險，不能證明絕不 overfit，bootstrap 也未校正過去的模型搜尋。', '',
        '## 完整固定候選結果', '',
        '正數為改善，負數為退步；以下均是嚴格新建案，與前面原始名稱分組的 MAE 分母略有不同。', '',
        '| 模型 | 混合權重 | 2021 年窗口改善 | 2022 年窗口改善 | 2021 後八月改善 | 2022 後八月改善 |',
        '|---|---:|---:|---:|---:|---:|']
    for name, alpha in keys:
        a = table.loc[(table.name == name) & (table.alpha == alpha)].set_index('fold')
        gains = [a.loc[f, 'cold_gain_pct'] for f in WINDOWS]
        late = [a.loc[f, 'late_cold_gain_pct'] for f in WINDOWS]
        lines.append(f'| {name} | {alpha:.0%} | {gains[0]:+.3f}% | {gains[1]:+.3f}% | {late[0]:+.3f}% | {late[1]:+.3f}% |')
    lines += ['', '![研究結果](generalization_study.png)', '', '## 決策', '']
    if selection['selected'] is None:
        lines += ['**本輪全部候選未達跨年度採用門檻，不產生 v7，保留 v6。** 未啟動第二 seed、第三年度候選確認或 final 重訓。接近門檻也不修改原門檻，不將單年度最好的結果包裝為已證實改善。', '',
            '這個結果只否定本次固定設定已足以取代基準，不代表所有遮蔽方法或 L1 模型無效。下一個值得研究的方向仍是新建案，但應補充可跨建案轉移的非價格區位與建物特徵，而不是繼續在同一批驗證上擴大權重／樹深搜尋。候選例如可追溯版本的交通可達性、土地使用與建物結構屬性；目前尚未蒐集、訓練或宣稱有效。外部成交／配售資料仍須排除 test 同戶標籤，不能用地址接回答案。']
    else:
        chosen = selection['selected']
        lines.append(f"初選固定為 {chosen['name']}、權重 {chosen['alpha']:.0%}。")
        for name in ['seed_check', 'confirmation', 'verification']:
            path = OUT / f'{name}.json'
            if path.exists():
                lines += ['', f'### {name}', '', '```json', path.read_text().strip(), '```']
        if not (OUT / 'verification.json').exists():
            lines += ['', '**後續穩定性或確認門檻未全部通過，不產生新版提交；保留 v6。**']
        else:
            lines += ['', '已產生 v7；公開榜單結果未知。另見 reproduction.json 的保存模型推論重現檢查。']
    verification = OUT/'audit_verification.json'
    if verification.exists():
        audit = read(verification)
        lines += ['', f"已驗證 {len(audit['checks'])} 個保存模型，各抽查 512 筆歷史預測並檢查列順序不變性；凍結的來源與輸入雜湊一致。詳見 audit_verification.json。既有六份提交檔完整性另記錄於 legacy_integrity.json。"]
    lines += ['', '## 檔案與重現', '',
        '- `train_generalization.py`：訓練、固定門檻、seed 與確認流程。',
        '- `test_generalization.py`：群組遮蔽、未知建案、選擇門檻與 L1 尺度等價性測試。',
        '- `config.json`、`source/`：訓練前凍結設定與來源快照。',
        '- `comparison.csv`：兩個年度、四組模型、兩個權重的完整數據。',
        '- `ablation.csv`：未混合的四組模型新建案 MAE。',
        '- `selection.json`：所有候選與不可更改的選擇結果。',
        '- `generalization_study.png` / `.pdf`：可匯出的研究圖。', '',
        '```bash', 'OMP_NUM_THREADS=12 OPENBLAS_NUM_THREADS=1 .venv/bin/python train_generalization.py all',
        '.venv/bin/python report_generalization.py', '.venv/bin/python verify_generalization.py',
        '.venv/bin/python -m pytest -q', '```', '',
        'GPU 重訓不保證逐位元一致；保存模型的推論另行檢查。CatBoost 的損失與 GPU 支援依[官方回歸損失文件](https://catboost.ai/docs/en/concepts/loss-functions-regression)；方法是否符合主辦單位全部規則仍不能由本地實驗自行認定。', '']
    (OUT / 'RESULTS.md').write_text('\n'.join(lines))
    print(json.dumps(dict(status=selection['status'], report=str(OUT/'RESULTS.md'),
        candidate_rows=len(table), frozen_rounds=config['iterations']), ensure_ascii=False))


if __name__ == '__main__':
    main()
