"""Document public feedback separately from reused historical backtests."""
import json
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from train_hedonic import ROOT, OUT, V2, SPATIAL
from select_hedonic import selected_names, SPECS
from train_mae import checksum, dump


def main():
    sel=json.loads((OUT/'selection.json').read_text())
    ev=json.loads((OUT/'confirmation/evaluation.json').read_text())
    verify=json.loads((OUT/'verification.json').read_text())
    rows=[]
    for fold in ['stress2021','development']:
        r=sel[f'{fold}_mae']
        rows.append(dict(window=fold,v2_mae=r['v2'],v3_mae=r['candidate']))
    rows.append(dict(window='confirmation',v2_mae=ev['v2']['mae'],v3_mae=ev['candidate']['mae']))
    for r in rows:r['relative_improvement']=1-r['v3_mae']/r['v2_mae']
    pd.DataFrame(rows).to_csv(OUT/'comparison.csv',index=False)
    old_submission=pd.read_csv(ROOT/'submission1017_v2.csv')
    new_submission=pd.read_csv(ROOT/'submission1017_v3.csv')
    delta=new_submission['總價元']/old_submission['總價元']-1
    dump(OUT/'submission_changes.json',dict(
        rows_changed=int((old_submission['總價元']!=new_submission['總價元']).sum()),
        fractional_change_quantiles={str(q):float(delta.quantile(q)) for q in [0,.01,.25,.5,.75,.99,1]},
        mean_absolute_change_ntd=float(abs(new_submission['總價元']-old_submission['總價元']).mean())))
    names=selected_names(sel)
    ci=ev['uncertainty']['improvement_over_legacy_95pct']
    lines=['# v3：跨市場期間選模與房屋結構分支','',
        '交付檔案：`submission1017_v3.csv`。公開榜單回饋由使用者提供：原版 **2,393,304.6257**，v2 **2,188,394.3386**，下降 **8.56%**。v3 尚未提交、沒有公開榜單成績，不能宣稱已超越第一名。','',
        '## 本輪結論','',
        '本輪實作並訓練新的結構／區位專家，將符合兩段歷史回測改善條件的分支加入 v2。每個已見／未見建案分組至少保留 25% 的 v2 權重。所有最終選擇只依歷史總價 MAE，沒有把公開榜單分數轉成價格倍率。','',
        '| 回測用途與期間 | v2 MAE（元） | v3 MAE（元） | 下降比例 |',
        '|---|---:|---:|---:|']
    labels={'stress2021':'壓力選模：2021-08-11 → 2022-08-10',
            'development':'開發選模：2022-08-11 → 2023-08-10',
            'confirmation':'凍結後確認：2023-08-11 → 2024-08-10'}
    for r in rows:
        lines.append(f"| {labels[r['window']]} | {r['v2_mae']:,.0f} | {r['v3_mae']:,.0f} | {r['relative_improvement']:.2%} |")
    lines+=['','上述比較全部是**重用歷史資料的回測**，並非三個完全未接觸的測試集。2021 壓力窗口的模型只用截止日前資料訓練，但迭代數、v2 集成權重與方法設計已受較晚歷史資料影響；它提供跨時期壓力檢查，不能冒充當年可部署的無前視績效。2023–2024 標籤也曾參與之前的模型工作。新的分支與混合比例在本輪確認預測產生前已凍結。','',
        '| 確認年度分組 | 筆數 | v2 MAE | v3 MAE |','|---|---:|---:|---:|']
    for name,r in ev['groups'].items():
        lines.append(f"| {name} | {r['rows']:,} | {r['v2']['mae']:,.0f} | {r['candidate']['mae']:,.0f} |")
    lines+=['',f'以建案／建物群組進行 500 次配對 bootstrap，確認年度 MAE 改善量的 95% 區間為 **[{ci[0]:,.0f}, {ci[1]:,.0f}] 元**，正值代表改善。此區間不包含反覆開發的選擇偏差，也不能量化未來市場轉折。','',
        '## 已實作的方法','',
        '原有 v2 包含地址字元表徵、道路與地段、同建案固定效果的近期時間指數，以及 CatBoost／LightGBM 專家。本輪保留這些來源，新增以下受控比較。不同模型也使用不同隨機種子與學習率，故不能將整體改善全部歸因於單一特徵。','',
        '```mermaid\nflowchart LR\n  X[競賽交易欄位] --> V[v2 時間與區位集成]\n  X --> F[地址／房型／樓層／面積]\n  F --> N[扣除部分車位面積的單價 CatBoost]\n  F --> D[原建物面積的深度 10 CatBoost]\n  N --> T[受限時間趨勢外推]\n  D --> T\n  V --> G[已見／未見建案分組混合]\n  T --> G\n  G --> P[總價預測]\n```','',
        '| 新分支 | 面積尺度 | 額外區位統計 | 深度 | 訓練損失 | 開發凍結輪數 |',
        '|---|---|---|---:|---|---:|']
    for name,spec in SPECS.items():
        n=json.loads((OUT/'development'/f'{name}.json').read_text())['iterations']
        lines.append(f"| {name} | {spec['area']} | {spec['spatial']} | {spec['depth']} | {spec['loss']} | {n:,} |")
    lines+=['',
        '令 A 為面積，I(t) 為僅由 fitting 資料估計的城市時間指數，B=A×I(t)。新模型擬合 `log(y/B)`，輸出 `B×exp(f(x))`。RMSE 分支權重為截斷的 `sqrt(y/10⁷)`；Huber 分支權重為截斷的 `y/10⁷`，delta=0.10。這些對數損失不是原價 MAE 的精確等價物；是否採用仍以原價 MAE 判斷。舊 v2 的加權單價 L1 分支保有精確 MAE 等價性。','',
        '- `total` 使用建物面積，缺失時以土地面積回退；`net` 在車位面積有效時使用 `建物面積 − 0.75×車位面積`。0.75 是預先固定的結構假設，不是估出的車位價格，也不是法律上的坪數定義。',
        '- 區位統計使用全國、城市、行政區、道路、道路×型態、地段、交叉口、建案名稱前綴的分層收縮。兩個道路欄位共用字典；每層都扣除查詢建案的全部交易，包括全國回退，避免直接讀取同建案價格。',
        '- 區位統計使用 fitting partition 的標籤；它們不是「每筆訓練列僅見過去」的時間因果編碼。確認與 test 標籤完全不參與統計。',
        '- 模型輪數只用開發年度前四個月 early stopping 決定；之後在壓力、確認與全量資料中固定。新增模型沒有使用最近交易加權，該加權與行情殘差組合在先導實驗中未達標。','',
        '## 防止只適合單一年度的選模規則','',
        '區位統計與兩種 Huber 實驗未通過開發門檻，先排除。其餘分別對已見與未見建案選擇 net_log、deep_log 或兩者等權平均。比較時間趨勢延續比例 0、0.5、1，以及新分支權重 0.25、0.5、0.75。候選必須在 2021 壓力年與 2022 開發年都改善至少 0.5%；否則保持 v2。合格者最小化兩個年度中較差的 `候選MAE/v2MAE`；在最低值 0.5% 以內優先選較小的新分支權重。0 趨勢選項是在開發模型的分組診斷後、壓力模型及確認預測之前納入。','',
        '| 分組 | 新分支 | 新分支權重 | v2 權重 |','|---|---|---:|---:|']
    for key,choice in sel['branches'].items():
        lines.append(f"| {key} | {choice['candidate'] or '未新增'} | {choice['alpha']:.2f} | {1-choice['alpha']:.2f} |")
    lines+=['','這個有限網格與最差年度準則降低單年調參的風險，但沒有消除選模過擬合。完整候選成績保存在 `selection.json`，可檢查未採用分支。沒有調整全域價格倍率。','',
        '## 未採用的先導實驗','',
        '先導實驗使用同樣的跨建案區位先驗，直接擬合其對數殘差，並加入近期樣本加權及移除建案識別的分支。開發表現不佳，沒有加入提交。這一輪的區位特徵只作為可由樹選擇的輸入，不強迫作為加法錨點。','',
        '直接加入區位特徵的 context_log 開發 MAE 約 239 萬元，亦未採用。原始 `net_huber` 在未平移的 log 單價上出現數值訓練失敗（開發 MAE 約 44 億元）。修正版 `centered_huber` 將目標固定減去 `log(150000)`，仍有極端價格外推，原價 MAE 約 1,720 萬元。兩者全部排除，沒有以任意裁切價格掩蓋問題，也沒有訓練其他年度或進入集成。修正版另有 `centered_config.json`，原始模型及設定未覆寫。','',
        '| 先導分支 | 開發 MAE（完整趨勢） |','|---|---:|']
    for name in ['cold_total','cold_net','context_net','cold_huber']:
        r=json.loads((SPATIAL/'development'/f'{name}.json').read_text())
        lines.append(f"| {name} | {r['all']['mae']:,.0f} |")
    lines+=['','## 檔案與重現','',
        f"- 提交檔 {verify['rows']:,} 筆，欄名 `編號,總價元`，ID 與官方範本完全對齊，價格為正整數。",
        '- 已從原始 test、保存的前處理與新增模型重新推論，整元集成結果逐列一致；沿用的 v2 基線也與已提交 CSV 逐列一致。',
        '- 官方 valid 沒有加入訓練。未使用外部成交價格、經緯度或 test 標籤。',
        f"- 提交 SHA-256：`{verify['sha256']}`。舊提交檔均保留。",'',
        '在已有前版完整 artifacts 的專案中重現：','',
        '```bash\n.venv/bin/python train_spatial.py prepare\n.venv/bin/python train_spatial.py stress_baseline --models cat_unit cat_log lgb_unit cat_total cat_market lgb_market lgb_price\n.venv/bin/python train_hedonic.py --fold development --models net_log deep_log\n.venv/bin/python train_hedonic.py --fold stress2021 --models net_log deep_log\n.venv/bin/python select_hedonic.py freeze\n'
        +(f".venv/bin/python train_hedonic.py --fold confirmation --models {' '.join(n for n in names if n!='centered_huber')}\n" if any(n!='centered_huber' for n in names) else '')
        +('.venv/bin/python train_centered.py --fold confirmation\n' if 'centered_huber' in names else '')+
        '.venv/bin/python select_hedonic.py evaluate\n'
        +(f".venv/bin/python train_hedonic.py --fold final --models {' '.join(n for n in names if n!='centered_huber')}\n" if any(n!='centered_huber' for n in names) else '')
        +('.venv/bin/python train_centered.py --fold final\n' if 'centered_huber' in names else '')+
        '.venv/bin/python select_hedonic.py export\n.venv/bin/python -m pytest -q\n.venv/bin/python report_hedonic.py\n```','',
        '保存模型可重現此提交；從零 GPU 訓練仍可能有浮點非決定性。此報告依賴前版 `run_v2`、`robust_v1` 與本輪 `spatial_v1` 前處理，並非只有一個模型檔就能運行。','',
        '損失函數定義可查 [CatBoost 官方迴歸文件](https://catboost.ai/docs/en/concepts/loss-functions-regression)。所有成績來自本專案實測，並非文件宣稱的效果。']
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    source=OUT/'source';source.mkdir(exist_ok=True)
    files=['train_hedonic.py','train_centered.py','select_hedonic.py','report_hedonic.py','test_hedonic.py',
        'spatial_prior.py','train_spatial.py','test_spatial.py','robust_features.py',
        'train_robust.py','select_robust.py','features.py','market.py','train.py','train_mae.py',
        'mae_models.py','comparable_sales.py','requirements.lock.txt']
    for name in files:shutil.copy2(ROOT/name,source/name)
    dump(source/'manifest.json',{f:checksum(source/f) for f in files})
    artifacts=list((OUT/'final').glob('*.joblib'))+[OUT/'config.json',OUT/'selection.json',V2/'selection.json']
    artifacts += [SPATIAL/'final'/f'{mode}_preprocess.joblib' for mode in ['total','net']]
    dump(OUT/'artifact_sha256.json',{str(p.relative_to(ROOT)):checksum(p) for p in artifacts})
    fig,ax=plt.subplots(figsize=(9,4.5),layout='constrained')
    pos=np.arange(len(rows));width=.34
    old=ax.bar(pos-width/2,[r['v2_mae']/1e6 for r in rows],width,label='Submitted v2',color='#99a6b2')
    new=ax.bar(pos+width/2,[r['v3_mae']/1e6 for r in rows],width,label='Candidate v3',color='#0e7490')
    for bars in [old,new]:ax.bar_label(bars,fmt='%.3f',padding=3)
    ax.set_xticks(pos,['2021 stress selection','2022 development','2023 confirmation'])
    ax.set_ylabel('MAE (million NTD)');ax.set_ylim(0,max(r['v2_mae'] for r in rows)/1e6*1.22)
    ax.set_title('Reused historical backtests — not Kaggle scores')
    ax.spines[['top','right']].set_visible(False);ax.legend()
    fig.savefig(OUT/'comparison.png',dpi=180);fig.savefig(OUT/'comparison.pdf');plt.close(fig)
    print(OUT/'RESULTS.md')


if __name__=='__main__':main()
