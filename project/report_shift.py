"""Summarize the completed experiments, preserving failed hypotheses."""
import json,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from train_shift import ROOT,OUT
from train_mae import checksum,dump


def main():
    sel=json.loads((OUT/'selection.json').read_text())
    evaluation=json.loads((OUT/'confirmation/evaluation.json').read_text())
    promoted=(OUT/'verification.json').exists()
    records=[]
    for fold,label in [('stress2021','2021-08 → 2022-08'),('development','2022-08 → 2023-08')]:
        r=sel[f'{fold}_mae'];records.append((label,r['v3'],r['candidate']))
    records.append(('2023-08 → 2024-08',evaluation['v3']['mae'],evaluation['candidate']['mae']))
    table='\n'.join(f'| {label} | {old:,.0f} | {new:,.0f} | {(1-new/old):.2%} |' for label,old,new in records)
    choices='\n'.join(f"- **{branch}**：候選 `{choice['candidate']}`，新模型權重 {choice['alpha']:.0%}、v3 權重 {1-choice['alpha']:.0%}。" for branch,choice in sel['branches'].items())
    interval=evaluation['uncertainty']['improvement_over_legacy_95pct']
    title='v4 候選：近期資料權重與區位轉移模型' if promoted else '本輪候選未通過輸出條件'
    status='已產生 `submission1017_v4.csv`；尚未提交，沒有新的 public MAE。' if promoted else '保留 v3，未產生新的正式提交檔。'
    text=f'''# {title}

{status} 使用者回報 v3 的 public MAE 為 **2,092,770.8572**。下表為相同時間切分下的歷史比較，不能直接與 public leaderboard 比較。

| 歷史窗口 | v3 MAE（元） | 本輪候選 MAE（元） | 改善 |
|---|---:|---:|---:|
{table}

確認年度以建案／建物群組 bootstrap 估計的 MAE 改善量 95% 區間：**{interval[0]:,.0f}–{interval[1]:,.0f} 元**。這個區間未校正累積模型搜尋與歷史資料重用，不能當作獨立測試的統計保證。

改善集中在預測前四個月：MAE 從 {evaluation['groups']['early']['v3']['mae']:,.0f} 降至 {evaluation['groups']['early']['candidate']['mae']:,.0f} 元。後八個月從 {evaluation['groups']['later']['v3']['mae']:,.0f} 到 {evaluation['groups']['later']['candidate']['mae']:,.0f} 元，幾乎持平；長期外推的主要問題仍未解決。

## 本輪實作

保留 v3 的面積與行情分解，新增 11 個備註語義旗標。只辨識地上權、安置住宅、減免、毛胚、設備、露台、雨遮、農地、員工、換約、專用車位等詞；沒有將金額文字餵入模型。

四個 CatBoost 分支都預測扣除部分車位面積後的單價對數殘差。比較完整歷史、最近 24 個月、18 個月指數衰減、衰減加建案樣本平衡。18 個月是衰減時間常數，不是半衰期；所有權重只由各次 fitting 資料決定。

另一個 LightGBM 分支移除 `project`、`canonical_project`、`project_building_type`、`address_base`、`tower`，保留道路與地段、建商前綴、面積、樓層和房型等。目標為 `總價 / (有效面積 × 行情指數 × 100000)`，以 `有效面積 × 行情指數` 作 L1 權重，因此在額外近期權重前，損失與原價 MAE 成比例。這個分支不使用建案專屬識別欄位。

選定分支：

{choices}

候選名中的 `@d` 表示新增分支的趨勢延續比例：以 `exp((d−1) × time_index)` 修正原預測。此修正只套用新增分支，v3 部分保留原本計算。

## 選模與過擬合控制

1. 迭代數只由 2022 年開發窗口的前四個月 early stopping 決定，其他窗口與最終重訓沿用固定迭代數。
2. 五個模型各比較趨勢延續 0、0.5、1，以及集成權重 0.25、0.5、0.75；按已見／未見建案分支選擇。所有比較以原價 MAE 計算。
3. 每個分支必須在 2021 與 2022 兩個窗口各改善至少 0.5%。在合格候選中最小化較差年度的相對 MAE；接近時優先採用較低新模型權重。
4. 權重先寫入 `selection.json`，再訓練確認年度模型。確認年度整體 MAE 若惡化，輸出程式拒絕產生 v4。
5. 這些歷史窗口已在過去版本反覆檢查；本輪亦有多個模型比較，不能宣稱是全新、完全獨立的驗證。沒有用 public 分數反推出提交縮放係數。

目前測試套件 24 項通過，包括行情時間方向、未見類別處理、MAE 加權等價性，以及新增的推出時點指數與語義旗標測試。保存模型從原始 test 重建特徵的驗證狀態記錄於 `verification.json`（若候選通過輸出條件）。GPU 重訓不保證逐位元相同，保存模型推論則要求可重現。

最後的逐筆差異稽核見 `largest_changes.csv`。最大金額變動來自 ID 30528：原資料建物面積 94,344.74 平方公尺、830 個車位，v3 約 135.83 億元，v4 約 140.42 億元，相對變動 3.38%。這是大宗交易的尺度敏感性，並非已發現的單位轉換錯誤；沒有測試真值可判定何者較準，也沒有事後依 test 手動截斷。確認年度總價逾 1 億元的 319 筆資料，MAE 從約 1,944 萬降至約 1,896 萬元，仍不足以保證這筆極端交易的準確度。

## 未採用的趨勢假設

另外實作了以各建案最初三個月銷售估計的「推出時點指數」。六個候選沒有任何一個在兩個窗口都改善至少 0.5%；較強外推在上漲期有效，另一年度反而增加誤差。沒有為採用它而放寬門檻。詳見 [launch_v1/RESULTS.md](../launch_v1/RESULTS.md)。

## 尚未解決、可能較重要的分布差異

test 有 **1,408 筆安置住宅（2.99%）**，有效 train 中為 **0 筆**。v3 預測均值約 **2,062 萬元**。沒有 test 真值，不能把這個均值解讀成已知誤差；也不能把備註中的房屋組成項當作含土地、車位的總價。詳見 [regime_audit/RESULTS.md](../regime_audit/RESULTS.md)。

官方政策背景顯示這批住宅存在成本配售、減收等不同定價機制。目前尚未取得完整競賽外部資料規則，因此沒有使用外部配售價格、地理座標或成交標籤。本輪的語義旗標無法從零筆對應案例學到安置住宅定價，這個缺口仍然存在。

若要處理此缺口，需要先取得競賽網址或外部資料規則，再核對官方政策適用日期及土地／房屋／車位／減收組成，建立可稽核的政策定價分支；目前沒有測得此分支的泛化效果或 public 改善。

## 重現

```bash
.venv/bin/python train_shift.py --fold development
.venv/bin/python train_shift.py --fold stress2021
.venv/bin/python train_transfer.py --fold development
.venv/bin/python train_transfer.py --fold stress2021
.venv/bin/python select_shift.py freeze
```

依 `selection.json` 中的候選名，只對選中的模型執行 `--fold confirmation`；`transfer` 使用 `train_transfer.py`，其餘使用 `train_shift.py --models <name>`。之後：

```bash
.venv/bin/python select_shift.py evaluate
# 確認不惡化後，對同一批模型執行 --fold final
.venv/bin/python select_shift.py export
.venv/bin/python report_shift.py
```

來源雜湊、資料雜湊、模型、預測與被淘汰候選均保留。原版、v2、v3 的提交檔沒有改寫。
'''
    (OUT/'RESULTS.md').write_text(text,encoding='utf-8')
    x=np.arange(3);fig,ax=plt.subplots(figsize=(8.8,4.8))
    old=np.array([r[1] for r in records])/1e6;new=np.array([r[2] for r in records])/1e6
    for offsets,vals,color,label in [(-.18,old,'#7a8795','v3'),(.18,new,'#087e8b','Candidate')]:
        bars=ax.bar(x+offsets,vals,.34,color=color,label=label)
        ax.bar_label(bars,labels=[f'{v:.3f}' for v in vals],padding=3,fontsize=9)
    ax.set_xticks(x,['2021–2022\nSelection','2022–2023\nSelection','2023–2024\nConfirmation'])
    ax.set_ylabel('MAE (million NTD)');ax.set_ylim(0,max(old.max(),new.max())*1.18)
    ax.set_title('Same-window historical comparison (reused data)');ax.legend(frameon=False)
    ax.spines[['top','right']].set_visible(False);fig.tight_layout()
    fig.savefig(OUT/'comparison.png',dpi=180);fig.savefig(OUT/'comparison.pdf');plt.close(fig)
    source=OUT/'source';source.mkdir(exist_ok=True)
    files=['train_shift.py','select_shift.py','train_transfer.py','launch_index.py','audit_launch.py',
           'audit_regimes.py','report_shift.py','test_shift.py']
    for f in files:shutil.copy2(ROOT/f,source/f)
    dump(OUT/'source_manifest.json',{f:checksum(ROOT/f) for f in files})
    print(OUT/'RESULTS.md')


if __name__=='__main__':main()
