"""Diagnose an unseen housing regime without inventing test labels or prices."""
import json
import numpy as np
import pandas as pd
from train_shift import ROOT, semantic_features
from train_robust import data
from train_mae import dump, checksum

OUT=ROOT/'artifacts/regime_audit'


def main():
    OUT.mkdir(exist_ok=True,parents=True)
    train,_,_=data();test=pd.read_parquet(ROOT/'data/test.parquet')
    tr=semantic_features(train);te=semantic_features(test)
    submitted=pd.read_csv(ROOT/'submission1017_v3.csv')
    np.testing.assert_array_equal(pd.to_numeric(test['編號']).to_numpy(),submitted['編號'].to_numpy())
    counts=pd.DataFrame({'train_count':tr.sum().astype(int),'test_count':te.sum().astype(int)})
    counts.to_csv(OUT/'semantic_counts.csv',encoding='utf-8-sig')
    mask=te.sem_relocation.astype(bool)
    rows=test.loc[mask,['編號','建案名稱','鄉鎮市區','土地位置建物門牌',
        '建物移轉總面積平方公尺','車位移轉總面積平方公尺','土地移轉總面積平方公尺',
        '移轉層次','交易年月日','備註']].copy()
    rows['v3_prediction']=submitted.loc[mask,'總價元'].to_numpy()
    # These are house components as explicitly named in the provided notes.
    # They are diagnostics ONLY; neither is treated as the total target.
    note=rows['備註'].fillna('').str.replace(',','',regex=False).str.replace('，','',regex=False)
    rows['note_original_house_component']=pd.to_numeric(note.str.extract(r'原房屋價款\s*(\d+)元',expand=False),errors='coerce')
    rows['note_reduced_house_component']=pd.to_numeric(note.str.extract(r'減收後之房屋價款(?:為)?\s*(\d+)元',expand=False),errors='coerce')
    rows.to_csv(OUT/'relocation_rows.csv',index=False,encoding='utf-8-sig')
    groups=rows.groupby('建案名稱',sort=False).agg(rows=('編號','size'),
        v3_mean=('v3_prediction','mean'),v3_median=('v3_prediction','median'),
        notes_with_house_component=('note_original_house_component','count'))
    groups.to_csv(OUT/'relocation_projects.csv',encoding='utf-8-sig')
    result=dict(train_rows=len(train),test_rows=len(test),relocation_train=int(tr.sem_relocation.sum()),
        relocation_test=int(mask.sum()),test_fraction=float(mask.mean()),
        v3_prediction_mean=float(rows.v3_prediction.mean()),v3_prediction_median=float(rows.v3_prediction.median()),
        notes_with_house_component=int(rows.note_original_house_component.notna().sum()),
        observed_test_labels_available=False,subgroup_mae=None,external_prices_used=False,
        warning='House components in provided notes are NOT total transaction prices; no replacement rule was applied.',
        source_sha256=checksum(ROOT/'audit_regimes.py'),parent_sha256=checksum(ROOT/'submission1017_v3.csv'))
    dump(OUT/'audit.json',result)
    (OUT/'RESULTS.md').write_text(f'''# 未見定價制度稽核

競賽提供的 test 中有 **{int(mask.sum()):,} 筆安置住宅（{mask.mean():.2%}）**，有效 train 中為 **0 筆**。v3 對這批資料的預測均值是 **{rows.v3_prediction.mean():,.0f} 元**，中位數 **{rows.v3_prediction.median():,.0f} 元**。這是預測分布，不是已觀測的誤差；沒有 test 真值，因此無法計算此族群 MAE。

備註有 {rows.note_original_house_component.notna().sum()} 筆明列原房屋價款與減收後房屋價款。**房屋價款只是組成項，不可直接代替含土地／車位的總價元。** 原文與組成項解析結果只保存在診斷 CSV，未加入模型特徵，也未用來改寫提交價格。

## 官方政策背景與待確認事項

政策查閱顯示航空城安置住宅有成本配售及差額減收安排，與一般市場建案存在不同定價機制。資料中的「璟都航空城」是不同建案，不能因名稱近似而合併。

- [桃園市政府：安置住宅成本價配售說明](https://www.tycg.gov.tw/NewsPage_Content.aspx?n=10&s=315111)
- [航空城工程處：2024-04-23 安置住宅配售與差額減收會議結論](https://oapc.tycg.gov.tw/News_Content.aspx?n=5311&s=1275285)
- [桃園市政府：2024-08-23 安置住宅配售說明](https://www.tycg.gov.tw/NewsPage_Content.aspx?n=10&s=1346615)

目前只查閱政策背景，尚未將外部價格、配售單價或成交標籤用於任何預測。完整競賽外部資料規則尚未取得。下一步需要先核對規則及政策適用日期，再核對土地、房屋、車位與減收金額的組成；不能憑預期榜單改善幅度指定折扣。

若規則允許，可研究固定公式的「政策配售專家」，以競賽欄位辨認適用資格並分開計算組成項。沒有同制度的歷史真值時，仍須明示此分支無法用目前回測直接驗證。

## 可重現產物

- `semantic_counts.csv`：語義旗標在 train/test 的筆數。
- `relocation_rows.csv`：各筆輸入欄位、原有 v3 預測、備註明列的房屋組成項。
- `relocation_projects.csv`：建案統計。
- `audit.json`：來源與提交檔 SHA-256、診斷摘要。

重現：`.venv/bin/python audit_regimes.py`。此程式不產生或改寫提交檔。
''',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
