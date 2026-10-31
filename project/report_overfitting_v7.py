"""Render post-selection diagnostics; no training, model selection or export."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'artifacts/overfitting_v7'


def main():
    a = json.loads((OUT/'audit.json').read_text())
    curves = pd.read_csv(OUT/'learning_curves.csv')
    ensembles = pd.read_csv(OUT/'ensemble_checkpoints.csv')
    subgroups = pd.read_csv(OUT/'subgroup_metrics.csv')
    labels = {'stress2021': '2021-08 to 2022-08', 'development': '2022-08 to 2023-08',
              'confirmation': '2023-08 to 2024-08'}
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.6))
    for i, r in enumerate(a['folds']):
        fold = r['fold']
        for name, label, color in [('cat_geo', 'Frozen 50% geography blend', '#007c78'),
                                   ('cat_control', 'Matched 50% no-geo blend', '#b36b19')]:
            s = ensembles[(ensembles.fold==fold)&(ensembles.model==name)&(ensembles.scope=='cold')]
            axes[0, i].plot(s.trees, s.candidate_mae/1e6, 'o-', label=label, color=color)
        axes[0, i].axhline(r['cold']['baseline_mae']/1e6, color='#68737d', ls='--', label='Previous baseline')
        axes[0, i].set_title(labels[fold])
        axes[0, i].set_ylabel('Unseen-project MAE (million NTD)')
        s = curves[(curves.fold==fold)&(curves.model=='cat_geo')]
        for split, label, color in [('fit_sample', 'Fit sample: re-applied model', '#386cb0'),
                                    ('future', 'All future rows', '#ad494a')]:
            t = s[s.split==split].groupby('trees').weighted_log_rmse.mean()
            axes[1, i].plot(t.index, t, 'o-', label=label, color=color)
        axes[1, i].set_ylabel('Geography branch weighted log RMSE')
        for row in range(2):
            axes[row, i].set_xlabel('Trees (saved-model checkpoints)')
            axes[row, i].grid(alpha=.2)
            axes[row, i].spines[['top', 'right']].set_visible(False)
        axes[0, i].legend(fontsize=7)
        axes[1, i].legend(fontsize=7)
    fig.suptitle('v7 overfitting audit: held-out errors keep decreasing; a generalization gap remains', fontsize=13)
    fig.text(.5, .02, 'Two seeds; 50,000 fitting rows per year, all future rows. Historical windows were reused.\n'
        'Post-selection diagnostics only; no retraining or submission change. Training re-application is optimistic (categorical statistics).',
        ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .065, 1, .95))
    fig.savefig(OUT/'learning_curves.png', dpi=180)
    fig.savefig(OUT/'learning_curves.pdf')
    plt.close(fig)

    stage_summary = []
    for r in a['folds']:
        s = ensembles[(ensembles.fold==r['fold'])&(ensembles.model=='cat_geo')&(ensembles.scope=='cold')].sort_values('trees')
        stage_summary.append(dict(fold=r['fold'], cold_mae_monotonically_decreases=bool((np.diff(s.candidate_mae)<0).all()),
            cold_mae_500=float(s.iloc[0].candidate_mae), cold_mae_3000=float(s.iloc[-1].candidate_mae)))
    assessment = dict(
        conclusion='No observed aggregate late-training overfit within 500-3000 trees; generalization gap, subgroup instability, and selection-bias risk remain.',
        scope='Frozen v7 geography increment plus inherited learned policy branch; not a certificate for all historical submodels.',
        checkpoint_results=stage_summary,
        all_three_years_improve=True,
        historical_windows_reused=True,
        private_test_performance_unknown=True,
        current_action='Keep frozen v7; diagnostics do not justify retrospective per-city routing or blend tuning.',
        next_validation='Freeze a future candidate before a genuinely unexamined time/project-block holdout; tune only inside earlier folds. Reused-window tests cannot become pristine again.')
    (OUT/'assessment.json').write_text(json.dumps(assessment, ensure_ascii=False, indent=2)+'\n')

    lines = [
        'v7 過擬合診斷（2026-10-18）',
        '',
        '結論：在已檢查的 500–3,000 棵樹範圍，沒有看到 v7 整體隨訓練增加而驗證誤差反彈的現象。',
        '但訓練與未來資料有明顯誤差落差，部分建案／城市退步，歷史窗口也已重複使用。不能稱為「沒有過擬合」或保證私人榜表現。',
        '建議保留已凍結的 v7。這次是診斷，不以診斷結果挑輪數、改區域路由、調權重或產生新提交檔。',
        '',
        '一、實際執行範圍',
        '重新載入 3 個年度 × 有／無地理特徵 × 2 seeds，共 12 個保存 CatBoost 模型；',
        '每個模型在 500、1,000、1,500、2,000、2,500、3,000 棵樹重算推論。',
        '訓練診斷每年度固定隨機抽 50,000 筆；未來驗證採全部資料，另外報告嚴格未見建案。',
        '全部 3,000 棵樹預測與原始保存預測一致。沒有重訓、沒有使用 test 標籤。',
        f'本輪檢查 {a["checked_files_unchanged"]} 個既有來源、模型、證據及提交檔未變更。',
        '現有 43 項測試重跑通過；有一個既有 Pandas4Warning（類別未知值相容性警告）。',
        '本輪不重新訓練 v1–v6 的每一個舊分支；舊分支的選模歷史風險仍由 v7 繼承。',
        '',
        '二、跨年度與訓練階段（下列 MAE 是未見建案、實際 50% 混合、兩 seed 平均）',
    ]
    for r, s in zip(a['folds'], stage_summary):
        lines.append(f'{labels[r["fold"]]}：500 棵 {s["cold_mae_500"]:,.0f} 元 → 3,000 棵 {s["cold_mae_3000"]:,.0f} 元；'
            f'六個檢查點均逐步下降。相對基準全體改善 {r["all"]["gain_pct"]:.2f}%，未見建案改善 {r["cold"]["gain_pct"]:.2f}%。')
    lines += [
        '單 seed 的未來整體 MAE 亦隨檢查點下降；不是僅靠 seed 平均抵銷明顯的後期惡化。',
        '不能由此推論超過 3,000 棵仍有益；本輪沒有搜尋更多輪數。',
        '',
        '三、訓練／未來誤差落差（只針對新增地理分支，不是整個 v7 的訓練 MAE）',
        '以下使用相同 log(y/base) 目標及 sqrt(price) 截斷權重，列出兩 seed 的 RMSE 平均：',
    ]
    for r in a['folds']:
        s = curves[(curves.fold==r['fold'])&(curves.model=='cat_geo')&(curves.trees==3000)]
        m = s.groupby('split').weighted_log_rmse.mean()
        lines.append(f'{labels[r["fold"]]}：訓練樣本 {m["fit_sample"]:.5f}；全部未來資料 {m["future"]:.5f}；約 {m["future"]/m["fit_sample"]:.2f} 倍。')
    lines += [
        '落差值得警覺，但時間／市場漂移、未見建案占比與類別統計都會造成落差；單憑倍數無法辨認過擬合的貢獻。',
        'CatBoost 對訓練資料重推時的類別統計可使用整個 fitting 集合，通常比訓練時的逐序統計更樂觀。',
        '因此不可把此 resubstitution 分數當成泛化成績，也不可把訓練 log-RMSE 直接與總價 MAE 比較。',
        '依據：CatBoost 官方 FAQ https://catboost.ai/docs/en/concepts/faq',
        '',
        '四、建案／區位不穩定與改善集中程度',
    ]
    for r in a['folds']:
        macro = 100*(1-r['macro_project_candidate_mae']/r['macro_project_baseline_mae'])
        remaining = r['leave_top_gain_projects_out'][-1]
        ci = r['city_cluster_gain_ci95']
        lines.append(f'{labels[r["fold"]]}：{r["projects"]:,} 個未見建案，{r["projects_improved"]:,} 改善、{r["projects_worsened"]:,} 退步'
            f'（退步占 {100*r["projects_worsened"]/r["projects"]:.1f}%）；每建案等權 MAE 改善 {macro:+.2f}%。')
        lines.append(f'  排除改善總額最大的 10 個建案後，剩餘交易仍改善 {remaining["gain_pct"]:.2f}%；'
            f'按城市群組 bootstrap 的 MAE 改善 95% 區間 [{ci[0]:,.0f}, {ci[1]:,.0f}] 元。')
    lines += [
        '2021 年每建案等權平均略差，雖然競賽採逐交易加權的 MAE 仍改善，表示改善不是均勻分布。',
        '2022 年城市群組區間跨過零，對城市組成改變的穩健性證據較弱。這不是獨立的地理留出重訓。',
        '2022 年路口定位群組（5,761 筆）MAE 退步 5.79%；2023 年同類別（不同交易）改善 13.09%。',
        '2023 年嘉義縣（1,852 筆、41 建案）退步 10.71%；苗栗縣（3,195 筆、116 建案）退步 4.45%。',
        '這些是局部失穩的實測，不足以單獨證明原因是過擬合，也不能據此針對某城市事後切回舊版。',
        f'共 {int((subgroups.dimension=="month").sum())} 個年度內月份群組，其中 '
        f'{int(((subgroups.dimension=="month")&(subgroups.gain_pct<0)).sum())} 個退步；完整正負結果見 subgroup_metrics.csv。',
        '城市與建案 bootstrap 各重抽 2,000 次；區間只描述固定預測在現有資料上的組成不確定性。',
        '區間未校正歷史選模與多重群組檢視，也不涵蓋未來市場漂移。排除最大受益建案是事後敏感度檢查。',
        '',
        '五、繼承自 v6 的安置住宅分支',
    ]
    p = a['policy']; ci = p['project_cluster_gain_ci95']
    lines += [
        f'確認訓練 {p["training_rows"]} 戶、{p["projects"]} 個基地，與排除的 {p["excluded_same_test_unit_offers"]} 個 test 同戶身分無交集。',
        f'最終模型訓練報價 MAE {p["final_model_resubstitution_offer_mae"]:,.0f} 元；巢狀選模程序外層 MAE {p["nested_selection_outer_offer_mae"]:,.0f} 元。',
        f'外層基準 {p["nested_baseline_offer_mae"]:,.0f} 元；基地群組 bootstrap 改善區間 [{ci[0]:,.0f}, {ci[1]:,.0f}] 元，跨過零。',
        '10 個基地只有 5 個改善、4 個退步、1 個持平；其中兩基地各僅 3 戶。',
        '最終固定模型與巢狀程序不同，不能把其誤差差距全部歸因於過擬合。',
        '驗證目標是官方房地報價，不是競賽實際成交總價；缺乏安置住宅的實際成交驗證，仍是整體系統較大的證據缺口。',
        '備註明列差額仍依帳務關係扣除，屬確定性運算，沒有當作學到的參數。',
        '',
        '六、資料洩漏與選模歷史',
        '三個年度的 fitting 日期均早於該年度預測；未見建案的正規化鍵不出現在 fitting；跨切分完整原始特徵重複數為 0。',
        '完整特徵重複檢查包含交易日期，不等同於證明所有別名、相鄰戶或跨期同戶都獨立。',
        '保存模型沒有直接讀取 ID 或 target 欄；新增外部地理資料沒有價格欄。現有測試驗證目標／ID 擾動不改變地理特徵。',
        '目前檢查未發現新增分支直接讀入驗證價格或同戶外部價格的路徑；這不是對所有可能洩漏的形式化保證。',
        'v7 的選擇與權重在其公開分數回報前凍結；但三個歷史年度曾被多輪模型研究使用，不能視為全新獨立測試。',
        '反覆用同一驗證資料做模型選擇可能使評估偏樂觀；這種程序層面的風險不會因雙 seed 或 bootstrap 自動消失。',
        '依據：Cawley & Talbot (JMLR, 2010) https://www.jmlr.org/papers/v11/cawley10a.html',
        'v5 曾讀取逐戶官方報價；v6/v7 的政策模型已排除 test 同戶標籤，但整個研究歷程不能宣稱從未接觸那些報價。',
        '',
        '七、決策與下一步',
        '保留 v7：目前沒有驗證誤差後期反彈的證據支持縮短訓練；也沒有依局部負結果事後調路由的充分證據。',
        '使用者回報 public MAE：v6 1,745,458.0575 → v7 1,716,414.5323（改善 1.66%）；這是額外的公開集觀察，不保證 private。',
        '下一個候選應先凍結方法，再用真正未查看的時間／建案區塊確認；選模留在更早的內層切分。',
        '現有歷史資料已看過，改切分不能把它重新變成從未使用的資料；私人榜標籤目前也不可取得。',
        '',
        '重現：OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 .venv/bin/python audit_overfitting_v7.py',
        '繪圖：.venv/bin/python report_overfitting_v7.py',
        '資料：audit.json、learning_curves.csv、ensemble_checkpoints.csv、subgroup_metrics.csv、各年度 project_metrics.csv。',
        '圖表：learning_curves.png / learning_curves.pdf。',
    ]
    (OUT/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps(assessment, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
