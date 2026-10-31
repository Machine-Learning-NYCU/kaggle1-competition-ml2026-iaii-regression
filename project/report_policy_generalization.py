"""Report the policy-branch generalization audit without changing deployments."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/policy_generalization_v1'


def main():
    a=json.loads((OUT/'audit.json').read_text())
    labels={
        'leave_one_project_out':('整基地留出（10 折）','Leave one base out\n10 folds'),
        'leave_related_family_out':('A18 三期合併留出（8 折）','Related phases grouped\n8 folds'),
        'leave_source_out':('整份來源手冊留出（2 折）','Leave one handbook out\n2 folds'),
        'purged_leave_project_out':('基地留出＋排除相似樣本','Base out + similar rows purged\n10 folds'),
    }
    fig,axes=plt.subplots(2,1,figsize=(12,8.7))
    names=list(labels); at=np.arange(len(names)); width=.36
    b=[a['validation'][n]['fixed']['baseline_mae']/1000 for n in names]
    m=[a['validation'][n]['fixed']['row_mae']/1000 for n in names]
    axes[0].bar(at-width/2,b,width,color='#87959f',label='Other-base median-unit-price baseline')
    axes[0].bar(at+width/2,m,width,color='#007c78',label='Learned physical-feature model')
    for i,(bv,mv) in enumerate(zip(b,m)):
        axes[0].text(i-width/2,bv+3,f'{bv:.1f}',ha='center',fontsize=9)
        axes[0].text(i+width/2,mv+3,f'{mv:.1f}',ha='center',fontsize=9)
    axes[0].set_xticks(at,[labels[n][1] for n in names],fontsize=9)
    axes[0].set_ylim(0,max(b)*1.22)
    axes[0].set_ylabel('House/land OFFER MAE (thousand NTD)')
    axes[0].legend(loc='upper right',fontsize=8)
    o=pd.read_csv(OUT/'leave_one_project_out_oof.csv')
    g=o.assign(ae=abs(o.target_offer-o.fixed_prediction),be=abs(o.target_offer-o.baseline_prediction)).groupby('project').agg(
        rows=('unit','size'),model=('ae','mean'),baseline=('be','mean'))
    improved=(1-g.model/g.baseline)*100
    short=g.index.str.replace('桃園航空城機場園區街廓','Airport block ',regex=False).str.replace('桃園航空城','',regex=False).str.replace('基地','',regex=False)
    positions=np.arange(len(g))
    axes[1].bar(positions,improved,color=np.where(improved>0,'#007c78','#b05042'))
    axes[1].axhline(0,color='#58616b',lw=.8)
    axes[1].set_xticks(positions,[f'{n}\nn={int(k)}' for n,k in zip(short,g.rows)],fontsize=8)
    axes[1].set_ylabel('Per-base offer MAE improvement (%)')
    axes[1].set_title('Leave-base-out results: 8 of 10 bases improve; small bases remain uncertain',fontsize=11)
    for i,v in enumerate(improved):
        axes[1].text(i,v+(1 if v>0 else -2),f'{v:+.1f}%',ha='center',va='bottom' if v>0 else 'top',fontsize=8)
    axes[1].set_ylim(min(improved)-10,max(improved)+10)
    for ax in axes:
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    fig.suptitle('Policy housing: evidence of transfer within one regional policy scheme',fontsize=14)
    fig.text(.5,.017,'Only 441 screened other-unit offers; excluded query-unit prices never used for fitting/evaluation.\n'
        'Post-selection diagnostics; not actual-sale validation or evidence for other cities/policy regimes. v6/v7 unchanged.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.065,1,.95))
    fig.savefig(OUT/'generalization.png',dpi=180);fig.savefig(OUT/'generalization.pdf');plt.close(fig)

    purged=a['sensitivity']['exclude_query_similar_features']
    own=a['sensitivity']['exclude_query_own_project']
    evidence=dict(method_general_without_identity_lookup=True,
        same_policy_cross_base_transfer_supported=True,
        source_domain_transfer_less_certain=True,
        all_relocation_policies_generalization_established=False,
        actual_sale_price_accuracy_established=False,
        retired_lookup_method_reintroduced=False,
        action='Retain frozen v7 and v6; no test-price lookup, no new per-project adjustments, no submission changes.')
    (OUT/'assessment.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')
    lines=[
        '安置住宅泛化測試（2026-10-18）',
        '',
        '結論：通過推論不查同戶價格的檢查；新增實驗支持同一桃園航空城政策制度內的跨基地泛化。',
        '跨來源手冊的證據較弱，不能推論到所有安置制度、其他城市或實際成交總價。',
        'v6/v7 的模型及提交檔保持不變。本輪診斷重訓均使用副本，只產生評估結果，沒有替換部署模型。',
        '',
        '一、資料範圍與防止答案回查',
        '只讀取先前核准的 training_units.csv：441 戶、10 個基地。',
        'excluded_test_units.csv 僅含被排除的 1,404 個 project/unit 身分；不讀取其實際報價作訓練或評估。',
        '原始完整報價檔僅做 SHA-256 完整性核對；沒有把該檔載入模型或評估標籤。',
        '441 戶與 test 同戶身分無交集。但訓練的 10 個基地都出現在 test，僅有同戶排除不足以證明跨基地能力。',
        '所有來源同屬桃園航空城安置政策；沒有另一城市或另一政策制度的獨立標籤。',
        '',
        '二、1,408 筆實際查詢的反查表檢查',
        '移除地址、建案、棟號、ID：預測逐筆完全相同。',
        '換成從未出現的地址／建案／棟號，並加入任意目標價格欄：預測逐筆完全相同。',
        '打亂列順序、改用每批 37 筆、重複同一筆：結果一致。',
        '模型載入後封鎖 open / io.open / os.open / Path.open / pandas 讀檔 / joblib.load，仍能完成全部推論。',
        '把被排除身分拼上人工捏造的極端價格與無效面積，prepare 仍只留下相同 441 戶；原模型可由這些安全標籤重現。',
        '這項排除測試使用合成假價格，沒有讀回被排除戶的真實價格。',
        '備註原房價與減收後房價同時加相同金額，結果不變；明列減收差額增加 10 萬，總價恰好減少 10 萬。',
        '因此備註只提供明列帳務差額；該差額是規則運算，沒有宣稱為模型學習。',
        '',
        '三、重新訓練的泛化測試：全部只評估外部房地報價',
        '固定現有六特徵與正則化中位數回歸 alpha=0.001，沒有新增超參數搜尋。',
        '每個外層留出另重跑原三個候選的內層 3 折選模；內層將 A18 三期視為同一相關基地群。',
        '本輪 30 個外層折的內層選擇均為 linear_l1，因此固定模型與巢狀選模的 OOF 相同。',
        '外層標籤只在選模及訓練完成後計算 MAE；外層整組不出現在 fitting。',
        '整組留出的原理參考 scikit-learn 官方文件：',
        'https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeaveOneGroupOut.html',
        '',
        '測試 | 當折單價中位數基準 MAE | 模型 MAE | 改善',
    ]
    for name in names:
        s=a['validation'][name]['fixed']
        lines.append(f'{labels[name][0]} | {s["baseline_mae"]:,.0f} 元 | {s["row_mae"]:,.0f} 元 | {s["improvement_pct"]:.2f}%')
    lines += [
        '相似樣本定義在評估前固定：同樓層、同露臺旗標、淨面積差 <= 1 平方公尺。',
        '在整基地留出後，再從 fitting 排除與 heldout 特徵相似的其他基地住宅；每折最多再排除 88 戶。',
        '因此結果支持模型並非只能依靠同基地鄰戶或相同物理特徵組合。',
        '此測試未窮盡所有建築設計、社區鄰近與政策共通性；area/floor/terrace 以外的相似性仍可能存在。',
        '',
        '整基地留出中，8 個基地改善、2 個退步。退步的是 A11（15 戶）與 A18-3（3 戶）。',
        'A18-2 亦僅 3 戶，小基地誤差非常不穩定。未隱藏負結果，也沒有據此調整這些基地的提交價格。',
        '',
        '四、跨來源與原有驗證的限制',
    ]
    source=json.loads((OUT/'leave_source_out.json').read_text())
    for f in source['folds']:
        lines.append(f'留出 {f["heldout_group"]}：只用另一來源的 {f["train_rows"]} 戶訓練，'
            f'驗證 {f["heldout_rows"]} 戶；模型 MAE {f["fixed_mae"]:,.0f}，基準 {f["baseline_mae"]:,.0f} 元。')
    ci=a['validation']['leave_source_out']['fixed']['descriptive_group_gain_ci95']
    lines += [
        f'跨來源的基地群組重抽樣改善 95% 區間為 [{ci[0]:,.0f}, {ci[1]:,.0f}] 元，包含零。',
        '只有兩份手冊、同一政策體系，來源與時間、區域彼此混雜；不能稱為獨立跨城市或嚴格向前時間驗證。',
        '舊手冊標籤仍沿用已凍結的政策版本對齊規則，該轉換不是模型學得的政策效果。',
        '原先 5 折巢狀驗證 274,038 元仍保留；本輪 10 折每次可用更多訓練基地，內層相關群組也不同。',
        '因此本輪 243,592 元是另一種切分的診斷結果，不是模型升級，也不能宣稱對未知成交價又改善。',
        '所有 441 戶曾參與前期研究。本輪雖嚴格隔離每折資料，仍是事後穩健性研究，不能充當全新未見資料。',
        '',
        '五、與 test 相似樣本的依賴程度：只評估預測穩定性，不評估 test 誤差',
        f'test 有 {a["support"]["query_rows_near_training"]} 筆與安全訓練樣本接近，'
        f'其中 {a["support"]["query_rows_physically_identical"]} 筆在這六個物理特徵完全相同；這不等於同戶。',
        f'從 441 戶中一次移除與任何 test 戶接近的 {purged["removed_rows"]} 戶，只用剩下 {purged["training_rows"]} 戶診斷重訓：',
        f'房地預測的絕對變動中位數 {purged["median_absolute_change_pct"]:.2f}%，P95 {purged["p95_absolute_change_pct"]:.2f}%，最大 {purged["max_absolute_change_pct"]:.2f}%。',
        f'每筆 test 改用「排除自身整基地」的模型：房地預測變動中位數 {own["median_absolute_change_pct"]:.2f}%，P95 {own["p95_absolute_change_pct"]:.2f}%。',
        '這些小幅變動支持預測對相似戶／同基地標籤的依賴有限，但絕不代表知道 test 真正誤差。',
        f'{a["support"]["query_rows_outside_any_feature_range"]} 筆 test 超出訓練特徵單欄範圍；訓練面積上限 202.37 m²，test 上限 204.89 m²。',
        '',
        '六、訊號破壞與物理合理性',
        f'對每折 fitting 的每平方米價格作 20 次隨機打亂，保留面積尺度；20 次 OOF MAE 均比原模型差。',
        f'打亂後 MAE 中位數 {a["negative_control"]["median_mae"]:,.0f} 元，原模型 {a["negative_control"]["observed_fixed_oof_mae"]:,.0f} 元。',
        '這是破壞特徵／價格關係的負對照；政策價格不具可交換性，不能把 20 次結果轉成正式顯著性 p 值。',
        '在訓練面積／樓層範圍的 600 個物理特徵組合上，房地預測為正，增加 0.1 m² 均提高總價。',
        '模型可預測新組合，不需要建案、戶號或逐戶價表。物理合理性不是準確率證明。',
        '',
        '七、車位分支',
        f'對 2,090 筆車位報價作整基地留出，每折從其他基地估計的中位數仍為 950,000 元；MAE {a["parking"]["leave_project_out_offer_mae"]:,.0f} 元／位。',
        '車位使用全體庫存估計的常數模型；沒有依 test 車位身分配對報價，也沒有宣稱逐車位同戶排除。',
        '此車位 MAE 不與房地 MAE直接相加作成交誤差，因為組合後絕對誤差不具可加性。',
        '',
        '八、學到的函數可直接列出',
        '令 A = 扣除車位後面積（m²）、F = 樓層、T = 露臺旗標，I1/I2 = 一樓／二樓旗標。',
        '房地預測 = A × 學到的單價。單價是以下六個特徵的線性函數（係數四捨五入展示）：',
    ]
    formula=a['learned_formula']
    lines.append(f'單價截距 = {formula["unit_price_intercept_ntd_per_sqm"]:.6f} 元/m²')
    for key,value in formula['coefficients_for_original_six_features'].items():
        lines.append(f'  {key}: {value:+.6f}')
    lines += [
        'net_area_100sqm=A/100、floor_10=F/10、floor_10_squared=(F/10)^2，其餘為 0/1 旗標。',
        '這是從安全訓練標籤估計的整體函數，沒有每戶或每建案獨立係數。實際推論仍用原保存模型的完整精度。',
        '',
        '九、結論邊界',
        '可以說：目前安置住宅分支是依物理特徵學習的泛化方法，通過反查表測試，並有同政策跨基地的驗證證據。',
        '不能說：已證明所有安置住宅都適用、沒有任何過擬合、private 必然較好，或已驗證實際成交總價。',
        f'原部署模型、來源與提交共 {a["protected_files_unchanged"]} 個保護檔案雜湊一致；沒有產生新 submission。',
        '',
        '重現：OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 .venv/bin/python audit_policy_generalization.py',
        '報告與圖：.venv/bin/python report_policy_generalization.py',
        '測試：.venv/bin/python -m pytest -q',
        '資料：audit.json、inference_checks.json、各 scheme 的分割與 OOF CSV、protected_inventory.json。',
    ]
    (OUT/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps(evidence,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
