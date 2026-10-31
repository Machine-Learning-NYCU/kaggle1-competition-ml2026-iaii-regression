"""Report the frozen geography experiment without changing its selection."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from geography_features import ROOT, OUT
from train_geography import SPECS, SEED, WINDOWS


def read(name):
    return json.loads((OUT/name).read_text())


def main():
    selection = read('selection.json')
    audit = read('geocoder_audit.json'); coverage = read('coverage.json')
    rows = []
    for fold in WINDOWS:
        for name in SPECS:
            result = read(f'{fold}/{name}_{SEED}.json')
            for alpha, m in result['blends'].items():
                rows.append(dict(fold=fold, name=name, alpha=float(alpha),
                    baseline_cold_mae=m['cold']['baseline_mae'], cold_mae=m['cold']['candidate_mae'],
                    cold_gain_pct=100*(1-m['cold']['ratio']),
                    late_cold_gain_pct=100*(1-m['cold_late']['ratio']),
                    all_mae=m['all']['candidate_mae']))
    df = pd.DataFrame(rows)
    df.to_csv(OUT/'comparison.csv', index=False)
    fig, ax = plt.subplots(figsize=(10, 5), layout='constrained')
    combinations = [(n, a) for n in SPECS for a in [.25, .5]]
    for j, fold in enumerate(WINDOWS):
        values = [df.loc[(df.fold==fold)&(df.name==name)&(df.alpha==alpha), 'cold_gain_pct'].iloc[0] for name,alpha in combinations]
        ax.bar(np.arange(len(values))+(j-.5)*.36, values, width=.36,
            label={'stress2021':'2021-08 to 2022-08', 'development':'2022-08 to 2023-08'}[fold])
    ax.axhline(.5, color='black', linestyle='--', linewidth=1, label='Required in both years')
    ax.axhline(0, color='grey', linewidth=.8)
    ax.set_xticks(range(len(combinations)), [f'{n}\n{a:.0%}' for n,a in combinations], rotation=45, ha='right')
    ax.set(ylabel='Cold-project MAE reduction (%)', title='Non-price geography: fixed models and matching controls')
    ax.legend(fontsize=8)
    fig.savefig(OUT/'geography_study.png', dpi=180); fig.savefig(OUT/'geography_study.pdf')
    plt.close(fig)
    lines = ['# 非價格地理特徵：道路位置輔助的新建案房價模型', '',
        '本輪已實作、建立特徵並訓練比較。提交基準為 v6；新分支只以競賽 train 總價學習，外部資料不包含房價、配售價、實價登錄或 test 同戶標籤。', '',
        '## 資料與資訊時間', '',
        '來源是警政署的[歷史交通事故公開資料](https://data.gov.tw/dataset/12197)，採用 2020 年 A2 的地址及經緯度建立位置參考，A1 只用作定位誤差檢查。每份來源的官方下載網址、下載時間與 SHA-256 保存在 sources/ 的 JSON；授權為政府資料開放授權條款第 1 版。', '',
        f"A2 共 {audit['input_rows']:,} 筆，排除無效座標與無法解析／區位矛盾後保留 {audit['retained_rows']:,} 筆。模型只讀取地址、座標及用來限制年份的日期；不讀傷亡、車種等欄位。", '',
        '三份 CSV 檔尾都記載「資料提供日期：110年07月01日」，早於最早的房價回測切點 2021-08-10。這是來源宣告的提供日期；目前下載的位元組未經獨立驗證為當年的原始快照。固定 2020 年位置參考對所有歷史交易一致套用，並非宣稱每筆更早的訓練交易當時已有這份參考資料。', '',
        '## 可泛化的模型架構', '',
        '```mermaid', 'flowchart LR',
        '  A[2020 公開道路地址與座標] --> G[道路／區段／路口位置及可信度]',
        '  X[競賽地址與建物屬性] --> F[地理與原有建物特徵]',
        '  G --> F', '  F --> M[CatBoost 或 LightGBM]',
        '  Y[僅 fitting 的競賽總價] --> M',
        '  M --> V[跨年度、整建案未見驗證]',
        '  V --> P[通過門檻才納入預測]', '```', '',
        '依路口、100 號門牌區段、道路、行政區、城市逐層取得非價格位置統計。所有層級的最少參考點與最大分散距離均在價格訓練前固定；無法定位時保留缺失與低可信度。新增 13 個數值特徵包括經緯度、定位層級、參考筆數、座標分散程度、距城市／行政區參考中心的距離，以及半徑 1／3 公里的歷史參考點密度。', '',
        '參考中心是歷史事故點的中位位置，並非官方行政區幾何中心、CBD 或房屋精確座標；參考點密度也不是事故風險或交通可達性的直接量測。模型需自行從競賽標籤學習是否有預測價值。地址僅連接位置特徵，不連接外部價格；建案名、棟號、編號及目標價格的改動不會影響位置特徵。', '',
        '## 涵蓋率與定位品質', '',
        f"train 的道路或更細層級涵蓋率為 {coverage['train']['road_or_better_fraction']:.2%}，test 為 {coverage['test']['road_or_better_fraction']:.2%}。", '',
        '| 留出 A1 事故點的定位層級 | 筆數 | 中位誤差（km） | P90 誤差（km） |',
        '|---|---:|---:|---:|']
    names = {1:'城市', 2:'行政區', 3:'道路', 4:'100 號區段', 5:'路口'}
    for r in audit['heldout_a1']:
        lines.append(f"| {names[r['level']]} | {r['rows']:,} | {r['median_error_km']:.3f} | {r['p90_error_km']:.3f} |")
    lines += ['', '這是事故位置的留出測試，不能直接當作房屋定位誤差或房價 MAE。房屋地址經常只有地段或整條道路，模型必須保留位置不確定性。', '',
        '## 凍結的實驗與採用門檻', '',
        'CatBoost 固定深度 8、3,000 輪、對數 RMSE；LightGBM 固定 63 葉、2,000 輪、面積加權 L1。每種各比較有／無地理特徵；第一 seed 的 CatBoost 對照重用上一輪相同設定與 seed 的凍結預測。新增特徵的價格係數、樹分裂及交互作用均由 fitting 資料學習；沒有人工調整特定建案價格。', '',
        '只考慮 25%／50% 混合權重，作用於原始名稱與正規化建案鍵均未見的非安置住宅。兩個年度冷啟動 MAE 均須改善至少 0.5%，後八個月均不退步，且各年都要勝過相同混合權重的無地理特徵對照。接近最佳時依預定規則偏好較小混合權重。', '',
        '通過後才固定候選，進入第二 seed 與其對照；均值仍須通過原門檻，每個 seed 各年均須改善。再以 2023-08 至 2024-08 確認，同樣要求至少 0.5% 改善、後八個月不退步，而且相對基準與相對對照的建案群組 bootstrap 改善量 95% 下界都須大於零。失敗即停止，不能換候選或放寬門檻。', '',
        '**歷史窗口已重複用於先前研究，不能稱為全新獨立測試，也不能保證零過擬合。** 本輪預先固定候選、來源與參數，完整保留失敗結果，沒有用 public leaderboard 回饋選模。', '',
        '## 第一 seed 的完整結果', '',
        '正數表示 MAE 改善，負數表示退步。', '',
        '| 模型 | 權重 | 2021 年窗口 | 2022 年窗口 | 2021 後八月 | 2022 後八月 |',
        '|---|---:|---:|---:|---:|---:|']
    for name, alpha in combinations:
        a = df.loc[(df.name==name)&(df.alpha==alpha)].set_index('fold')
        values = [a.loc[f, metric] for metric in ['cold_gain_pct', 'late_cold_gain_pct'] for f in WINDOWS]
        lines.append(f'| {name} | {alpha:.0%} | '+ ' | '.join(f'{v:+.3f}%' for v in values)+' |')
    lines += ['', '![跨年度地理特徵結果](geography_study.png)']
    if (OUT/'development_location_diagnostic.json').exists():
        diagnostic = read('development_location_diagnostic.json')
        lines += ['', '### 定位層級診斷（不參與選模或改變作用範圍）', '',
            'development 年度的 cat_geo、固定 25% 混合結果如下。定位更精確不代表房價必然更準；沒有據此挑選有利群組、改權重或改路由。', '',
            '| 定位層級 | 新建案筆數 | MAE 改善 |', '|---|---:|---:|']
        for r in diagnostic['rows']:
            lines.append(f"| {names.get(r['location_level'], '無法定位')} | {r['rows']:,} | {r['gain_pct']:+.3f}% |")
    lines += ['', '## 決策與後續檢查', '']
    if selection['selected'] is None:
        lines += ['**未通過跨年度初選，不產生 v7，保留 v6。** 沒有進行第二 seed 或第三年度確認，也沒有在看到結果後追加候選。']
    else:
        s = selection['selected']
        lines.append(f"初選凍結為 **{s['name']}，混合權重 {s['alpha']:.0%}**。")
        if (OUT/'seed_check.json').exists() and read('seed_check.json')['accepted']:
            final_rows = dict(read('seed_check.json')['folds'])
            if (OUT/'confirmation.json').exists():
                final_rows['confirmation'] = read('confirmation.json')['metrics']
            labels = {'stress2021':'2021-08 → 2022-08', 'development':'2022-08 → 2023-08',
                'confirmation':'2023-08 → 2024-08'}
            lines += ['', '以下為凍結權重、兩 seed 平均後的結果；整體 MAE 包含沿用基準的其他交易。', '',
                '| 歷史窗口 | 基準整體 MAE | 候選整體 MAE | 整體改善 | 新建案改善 |',
                '|---|---:|---:|---:|---:|']
            for fold, m in final_rows.items():
                lines.append(f"| {labels[fold]} | {m['all']['baseline_mae']:,.0f} | {m['all']['candidate_mae']:,.0f} | {100*(1-m['all']['ratio']):.2f}% | {100*(1-m['cold']['ratio']):.2f}% |")
        for stage in ['seed_check', 'confirmation', 'verification', 'reproduction']:
            if (OUT/f'{stage}.json').exists():
                lines += ['', f'### {stage}', '', '```json', (OUT/f'{stage}.json').read_text().strip(), '```']
        if (OUT/'verification.json').exists():
            lines += ['', '**已產生 submission1017_v7.csv；公開 MAE 尚未知。** 受保護的已見建案與安置住宅沿用 v6。']
        else:
            lines += ['', '**後續穩定性或確認門檻未全數通過，保留 v6，不產生 v7。**']
    if (OUT/'confirmation_location_diagnostic.json').exists():
        lines += ['', '### 凍結候選在確認年度的定位層級診斷', '',
            '此表使用實際採用的 50% 權重、兩 seed 平均，僅供檢查，不改變選擇或作用範圍。', '',
            '| 定位層級 | 新建案筆數 | MAE 改善 |', '|---|---:|---:|']
        for r in read('confirmation_location_diagnostic.json')['rows']:
            lines.append(f"| {names.get(r['location_level'], '無法定位')} | {r['rows']:,} | {r['gain_pct']:+.3f}% |")
    if (OUT/'audit_verification.json').exists():
        verification = read('audit_verification.json')
        lines += ['', f"已檢查 {len(verification['model_checks'])} 個保存模型的抽樣推論重現與列順序不變性；原始 test 重新產生的地理特徵與快取一致。價格、ID、建案名、棟號的擾動不改變位置特徵。來源與設定雜湊通過檢查。"]
    lines += ['', '## 重現', '', '```bash',
        '# 重放本次已凍結的實驗，沿用 sources/、geocoder.joblib 與 features.joblib。',
        'OMP_NUM_THREADS=12 OPENBLAS_NUM_THREADS=1 .venv/bin/python train_geography.py all',
        '.venv/bin/python verify_geography.py', '.venv/bin/python report_geography.py',
        '.venv/bin/python -m pytest -q', '```', '',
        '資料準備實作位於 geography_features.py。若要更換來源、修改地址解析或重建特徵，應另開版本目錄，保留本次凍結檔；不要直接覆蓋來源與快取。訓練會核對凍結雜湊。GPU 重訓不保證位元組一致，保存模型推論另行驗證。', '']
    (OUT/'RESULTS.md').write_text('\n'.join(lines))
    print(json.dumps(dict(report=str(OUT/'RESULTS.md'), status=selection['status']), ensure_ascii=False))


if __name__ == '__main__':
    main()
