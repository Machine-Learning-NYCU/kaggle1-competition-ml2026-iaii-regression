# 314580075｜ML2026@IAII - Regression｜Kaggle 1 Competition

以時間回測、建案分組與外部資料隔離，建立台灣住宅成交總價預測系統。評分指標為原始新臺幣總價的 **MAE**。

作者：**黃駿甯（學號 314580075）** · GitHub：`jimhuang0731` · 提交 email：`adogg5528@gmail.com`。

目標作業 repo：[Machine-Learning-NYCU/kaggle1-competition-ml2026-iaii-regression](https://github.com/Machine-Learning-NYCU/kaggle1-competition-ml2026-iaii-regression)。

> 指定參考 repo 目前回傳 404，因此報告採方法、實驗、泛化診斷與重現說明的研究報告結構，尚未核對參考版型。發布狀態記錄於 `publication_status.json`。

## 報告與主要結果

- [完整報告 PDF](report/report.pdf)／[LaTeX 原始檔](report/report.tex)
- [v7 過擬合診斷](evidence/overfitting_v7/REPORT.txt)
- [安置住宅泛化測試](evidence/policy_generalization_v1/REPORT.txt)
- [逐版本公開分數紀錄](evidence/leaderboard_history.json)

| 提交候選 | Public MAE（元，越低越好） | 方法 |
|---|---:|---|
| [v7，主選](submissions/submission1017_v7.csv) | **1,716,414.5323** | 新建案加入兩個 seed 的地理 CatBoost；沿用學習式政策分支 |
| [v6，備選](submissions/submission1017_v6.csv) | 1,745,458.0575 | 一般住宅集成＋排除 test 同戶報價的政策回歸 |

分數來自使用者截至 **2026-10-18** 提供的 Kaggle 回報；不是本次重新取得的官方成績。**Private score 未知，沒有取得第一名的可驗證紀錄。** v5 的逐戶報價查表方法已淘汰，不列為候選。

## 方法

1. 清理訓練資料，保留 455,776 筆有效交易；官方 valid 不加入 fitting。
2. 從面積、格局、樓層、建材、地址、建案與時間建立特徵。行情及地址表示只由當折 fitting 資料估計。
3. 以 CatBoost／LightGBM 與面積尺度建模，使用原價 MAE 決定集成。
4. 新增非價格地理特徵：僅使用 2020 年官方道路地址／座標參考，不以外部成交價回填 test。
5. 針對備註標示的安置住宅，從排除 test 同戶後的 441 戶、10 個基地學習物理特徵與報價關係；推論不需要建案／戶號或逐戶價表。

v7 對 28,374 筆嚴格未見建案採用 `0.5 × v6 + 0.5 × mean(兩個地理 CatBoost)`；其餘 18,699 筆維持 v6，包含 1,408 筆安置住宅。

## 驗證與限制

| 歷史年度 | 一般住宅基準 MAE | v7 對應方法 MAE | 改善 |
|---|---:|---:|---:|
| 2021-08 → 2022-08 | 2,267,217 | 2,222,048 | 1.99% |
| 2022-08 → 2023-08 | 1,308,238 | 1,285,717 | 1.72% |
| 2023-08 → 2024-08 | 1,440,869 | 1,399,774 | 2.85% |

歷史窗口沒有安置住宅成交標籤，因此此表驗證地理分支相對一般住宅基準的改善，不能驗證政策分支的實際成交誤差。窗口曾在前期研究使用，不是全新獨立測試。

12 個保存 CatBoost 模型的 500–3,000 棵樹檢查點中，地理集成的三年度未見建案 MAE 持續下降。但約 45% 的未見建案仍退步，2022 年城市群組重抽樣的改善區間跨過零。

安置住宅追加測試包括整基地留出、相關基地合併留出、整份手冊留出，以及排除同樓層／同露臺且面積接近的樣本。報價 MAE 相對各自基準改善 17.08%、17.01%、6.54%、16.40%；全部來源仍屬同一政策體系，不能外推為所有安置制度或真實成交價的保證。

## 檔案配置

```text
report/             報告 PDF、LaTeX 與圖表
project/            原研究程式及 46 項測試，保持來源內容不變
evidence/           凍結設定、聚合驗證結果、來源網址與雜湊
submissions/        v6 / v7 最終候選 CSV
scripts/            本提交包檢查與報告建置工具
copied_sources.json 來源檔案與雜湊對照
publication_status.json 提交／版型對齊狀態
```

`evidence/` 是研究紀錄，不是供訓練程式直接載入的 `artifacts/` 快取；歷史報告中的 `public_mae: null` 等欄位保留提交前狀態，最新公開回饋另存於 leaderboard feedback。

## 執行檢查

```bash
python3 scripts/verify_package.py
python3 -m venv .venv
.venv/bin/pip install -r project/requirements.lock.txt
cd project
../.venv/bin/python -m pytest -q
```

46 項測試使用合成資料，不需要競賽原始檔。完整模型推論／重訓另需競賽資料與保存模型、外部來源快照；本 Git 包不包含約 4.5 GB 的完整研究快取，也不聲稱從此輕量包即可逐位元組重訓所有歷史候選。所需輸入與執行層級見 [重現說明](REPRODUCIBILITY.txt)。

重建 PDF（需要 XeLaTeX、latexmk、Noto CJK 字型）：

```bash
bash scripts/build_report.sh
```

## 外部資料與使用邊界

- [官方 2020 交通事故資料](https://data.gov.tw/dataset/12197)：只取地址、座標與年份作道路參考；目前下載位元組未獨立認證為 2021 年原始快照。
- 政策配售手冊：價格只用作已排除 test 同戶後的訓練標籤。來源記錄見 [policy_v1/sources](evidence/policy_v1/sources)。部分文件晚於競賽 train 截止日、早於受影響交易。
- 備註明列減收差額是確定性帳務運算；不是模型學習，也不是外部同戶價格查表。
- 方法依使用者指定的研究限制進行；尚未取得完整課程競賽規則，不能宣稱主辦方已認定外部資料使用合規。
- 報告整理、程式實作與診斷使用 AI 輔助；採用或淘汰決策以保存的程式、輸入與評估結果追溯。
