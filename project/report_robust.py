"""Record the submitted baseline and the new, historically tested candidate."""
import json
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from train_robust import ROOT,OUT
from train_mae import checksum,dump


def main():
    selected=json.loads((OUT/'selection.json').read_text())
    report=json.loads((OUT/'confirmation/evaluation.json').read_text())
    verification=json.loads((OUT/'verification.json').read_text())
    development=selected['development_metrics']
    manifests={fold:json.loads((OUT/fold/'manifest.json').read_text()) for fold in ['development','confirmation','final']}
    old,new=report['models']['previous_mae']['mae'],report['selected']['mae']
    ci=report['uncertainty']['improvement_over_legacy_95pct']
    lines=['# 榜單回饋後的一年期回測版本','',
      '提交候選：`submission1017_v2.csv`。先前 `submission.csv` 改名為 `submission1017.csv` 後，使用者回報 Kaggle MAE **2,393,304.6257 元**；當時第一名為 1,585,763.6199 元。新檔尚無榜單分數。', '',
      '## 為何需要改版','',
      '官方 valid 幾乎都是已見建案，測試集卻有 63.35% 的建案／建物群組未在訓練中出現，且預測期間接近一年。上一版較短的時間回測低估了實際測試難度。新版本以較長的前向切分開發，並用相同切分重訓舊模型作比較；沒有以榜單分數推算或反推測試標籤。','',
      '## 已實作的架構','',
      '1. 地址拆解：行政區名稱正規化、道路、第二道路、交叉路口、地段、巷與門牌數字；補充建案名稱前綴、房型與面積關係。地址字元 TF-IDF/SVD 僅由 fitting 資料建立，不使用價格標籤或外部地理資料。',
      '2. 時間指數：最近 24 個月同建案／型態／房數／面積區間內的 log 單價變化，消除各群組固定價格水準後估計城市斜率，向全國斜率收縮並設上限。這只是預測假設，不是政策因果推論。',
      '3. 面積尺度：令 B=A×時間指數。CatBoost、LightGBM 的單價分支以 y/(10⁵B) 為目標、B/100 為權重，使加權 L1 精確等價於總價 MAE/10⁷；另保留加權對數單價 CatBoost 與開發年度重新校準的舊模型分支，以增加模型差異。',
      '4. 可比成交：僅檢索 fitting 資料內同建案交易，依面積、車位比例、樓層、房數、建物型態與交易時間找相近單位，取加權單價中位數；沒有匹配時權重為零。',
      '5. 集成：已見／未見建案各使用非負、總和為一的 MAE 權重，附等權先驗收縮。趨勢延續比例只比較 0、0.5、1，可比成交比例只比較 0、0.25、0.5、0.75；在最低開發 MAE 的 0.5% 範圍內優先選較少可比成交校正。全部選擇只使用開發年度，簡化規則在查看確認成績前加入。','',
      '這是既有方法的整合與實測，沒有宣稱提出已經論文審查的新演算法。比較支持整體候選的結果，不能將改善全部歸因於某一項特徵。','',
      '## 切分與結果','',
      '| 用途 | fitting 截止 | 預測截止 | fitting 筆數 | 預測筆數 | 未見群組比例 |',
      '|---|---|---|---:|---:|---:|']
    for name,m in manifests.items():
        lines.append(f"| {name} | {m['cutoff']} | {m['forecast_end'] or '2025-07-31'} | {m['fit_rows']:,} | {m['forecast_rows']:,} | {1-m['seen_group_fraction']:.2%} |")
    lines+=['','三個新增分支的迭代數由開發年度前四個月的 early stopping 決定，其後固定迭代數重訓確認年度及全量模型；舊架構四個分支沿用先前固定的迭代數。開發年度同時參與迭代數與權重選擇，不能視為獨立驗證。','',
      '| 同切分比較 | 舊版 MAE（元） | 新版 MAE（元） | 下降比例 |','|---|---:|---:|---:|',
      f"| 開發年度 2022-08 → 2023-08 | {development['previous_mae']['mae']:,.0f} | {development['selected']['mae']:,.0f} | {1-development['selected']['mae']/development['previous_mae']['mae']:.2%} |",
      f'| 確認年度 2023-08 → 2024-08 | {old:,.0f} | {new:,.0f} | {1-new/old:.2%} |','',
      '| 確認年度分組 | 筆數 | 舊版 MAE | 新版 MAE |','|---|---:|---:|---:|']
    for name,row in report['groups'].items():
        lines.append(f"| {name} | {row['rows']:,} | {row['previous']['mae']:,.0f} | {row['selected']['mae']:,.0f} |")
    lines+=['',f'建案群組 bootstrap 的 MAE 改善量 95% 區間：[{ci[0]:,.0f}, {ci[1]:,.0f}] 元；正值代表改善。此區間不涵蓋模型選擇偏差、未來市場漂移或 public/private 子集差異。','',
      '## 凍結設定','',f"趨勢延續比例：{selected['trend_damping']}；可比成交比例上限：{selected['comparable_fraction']}，實際仍乘上樣本數與距離信心權重。",'',
      '| 分支 | 已見建案權重 | 未見建案權重 |','|---|---:|---:|']
    for j,name in enumerate(selected['models']):
        lines.append(f"| {name} | {selected['weights']['seen'][j]:.6f} | {selected['weights']['unseen'][j]:.6f} |")
    if selected['comparable_fraction']==0:
        lines+=['','可比成交校正已實作並測試，但開發改善不足以支持增加複雜度，因此最終預測不使用此分支。']
    lines+=['','## 限制與交付','',
      '- 確認年度結果在本次開發選擇完成後才用來確認候選；但其歷史標籤、舊模型迭代數曾在先前工作中使用，因此這是新切分下的歷史確認，不能稱全新未接觸的測試集。舊版已提交集成的權重只用作比較基準；新候選內的舊模型分支權重已在開發年度重新求解，不繼承原本 2024 年校準的數值權重。',
      '- 沒有使用 test 價格，也沒有把官方 valid 加入 fitting。只用競賽提供欄位；未下載外部成交資料或地理編碼。',
      '- 新版 public MAE 尚未知；回測改善不能保證榜單改善或第一名。請以上傳後成績為準。',
      f"- 已驗證 {verification['rows']:,} 筆、`編號,總價元`、範本 ID 順序、正整數價格，並從全部保存模型與前處理重新推論，整元預測逐列一致。SHA-256：`{verification['sha256']}`。",'',
      '## 重現','',
      '```bash\n.venv/bin/python -m pytest -q\n.venv/bin/python train_robust.py --prepare\n.venv/bin/python train_robust.py --fold development --models cat_unit cat_log lgb_unit cat_total cat_market lgb_market lgb_price\n.venv/bin/python select_robust.py freeze\n.venv/bin/python train_robust.py --fold confirmation --models cat_unit cat_log lgb_unit cat_total cat_market lgb_market lgb_price\n.venv/bin/python select_robust.py evaluate\n.venv/bin/python train_robust.py --fold final --models cat_unit cat_log lgb_unit cat_total cat_market lgb_market lgb_price\n.venv/bin/python select_robust.py export\n.venv/bin/python report_robust.py\n```', '',
      '保留上一版所有檔案。舊架構需要 `artifacts/run_v2` 的原始模型與固定迭代數；本次各年度重新訓練的基線存於對應目錄。保存模型推論可重現；GPU 從零重訓可能有浮點非決定性。']
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    source=OUT/'source';source.mkdir(exist_ok=True)
    files=['robust_features.py','train_robust.py','comparable_sales.py','select_robust.py','report_robust.py','test_robust.py',
           'features.py','market.py','train.py','train_mae.py','mae_models.py','requirements.lock.txt']
    for name in files:shutil.copy2(ROOT/name,source/name)
    dump(source/'manifest.json',{name:checksum(source/name) for name in files})
    model_files=list((OUT/'final').glob('*.joblib'))
    model_files += [ROOT/'artifacts/run_v2'/f'final_{name}.joblib' for name in ['cat_total','cat_market','lgb_market','lgb_price']]
    model_files += [ROOT/'artifacts/run_v2/market_final.joblib',OUT/'selection.json',OUT/'legacy_development_calibration.json']
    dump(OUT/'artifact_sha256.json',{str(path.relative_to(ROOT)):checksum(path) for path in model_files})
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    rows=[('Development year',development['previous_mae']['mae'],development['selected']['mae']),('Confirmation year',old,new)]
    for ax,(title,a,b) in zip(axes,rows):
        bars=ax.bar(['Previous','New'],np.array([a,b])/1e6,color=['#9aa5b1','#166c88'],width=.55)
        ax.bar_label(bars,fmt='%.3f',padding=4)
        ax.set_title(title);ax.set_ylabel('MAE (million NTD)');ax.set_ylim(0,max(a,b)/1e6*1.2)
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Historical backtests, not Kaggle scores')
    fig.savefig(OUT/'comparison.png',dpi=180);fig.savefig(OUT/'comparison.pdf');plt.close(fig)
    print((OUT/'RESULTS.md').read_text())


if __name__=='__main__':main()
