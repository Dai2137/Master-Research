import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_absolute_error, mean_squared_error, cohen_kappa_score, confusion_matrix, classification_report
from scipy.stats import spearmanr, t, pearsonr
import time
import os
import matplotlib
matplotlib.use("Agg")  # GUIなしで描画できるバックエンドに変更
import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns
import csv
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel



# 保存パスの指定と準備（変更するの忘れないように！！）
result_dir = r"D:\fujiwara\M\result\ordinal_multilabel\single\ordinal\olr\5_undersample"
os.makedirs(result_dir, exist_ok=True)



# ---------- Utility Functions ----------
def compute_ordered_metrics(y_true, y_pred):
    # acc = accuracy_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    corr, _ = pearsonr(y_true, y_pred)
    # spearman, _ = spearmanr(y_true, y_pred)
    # qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    
    return {
        # "Accuracy": acc,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "Corr": corr,
        # "Spearman": spearman,
        # "QWK": qwk
    }

def mean_ci(data, confidence=0.95):
    n = len(data)
    m = np.mean(data)
    se = np.std(data, ddof=1) / np.sqrt(n)
    h = se * t.ppf((1 + confidence) / 2., n-1)
    return m, m - h, m + h




# ---------- データ読み込み ----------
print("データ読み込みを開始...")
df = pd.read_csv(r"D:\fujiwara\M\data\after_preprocess\land_data_for_prediction.csv")
print("データ読み込み完了")


X = df.drop(columns=['will_not_be_re_registered', 'days_until_next_category', 'days_until_next'] +
            [col for col in df.columns if col.startswith("on_day_reason_group_") and col.endswith("_next")]).astype(np.float32)

y_ordinal = df['days_until_next_category'].values

# 順序クラスの数を決定
num_ord_classes = len(np.unique(y_ordinal))  # ← これで「元のラベルの個数 = 6」が得られる
olr_output_dim = num_ord_classes - 1 # 0以下〜4以下の5本の回帰式




# 評価指標格納用
all_metrics = []
# metric_names = ["Accuracy", "MAE", "MSE", "RMSE", "Spearman", "QWK"]
metric_names = ["MAE", "MSE", "RMSE", "Corr"]



# ループ外で定義（foldループの前）
all_y_true = []
all_y_pred = []

# 追加（統合フェーズの“日数”可視化/集計用）
all_true_days = []   # 実測日数（days_until_next）
all_pred_days = []   # 予測日数（カテゴリ→中央値[月]→日）


# 最終予測カテゴリに対する「y ≤ k」二値化の評価（AUCなし）
final_bincls_scores = {
    k: {"accuracy": [], "precision": [], "recall": [], "f1": []}
    for k in range(olr_output_dim)  # 最後は全て1になるので除外
}


# olrの二値分類の評価指標格納用
olr_per_task_scores = {  # foldごとに保持
    k: {"accuracy": [], "precision": [], "recall": [], "f1": []}
    for k in range(olr_output_dim)
}


# olrの二値分類のAUC格納用
olr_auc_per_task = {k: [] for k in range(olr_output_dim)}


label_names = [
    '-1 months', 
    '1-4 months', 
    '4-9 months', 
    '9-24 months', 
    '24-120 months',
    '120- months'
]


# ラベルと対応する範囲の中央値を定義
label_to_midpoint = {
    0: 31 * 0.5,     # <1 month
    1: 31 * 2.5,     # 1–4 months
    2: 31 * 6.5,     # 4–9 months
    3: 31 * 16.5,    # 9–24 months
    4: 31 * 72.0,      # 24–120 months
    5: 31 * 120.0      # >120 months（再登記なし）
}


seeds = list(range(3))

for seed in seeds:
    print(f"========== Seed {seed + 1} ==========")
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    for fold, (trainval_idx, test_idx) in enumerate(kf.split(X, y_ordinal)):
        print(f"\n========== Seed {seed + 1} Fold {fold+1} ==========")

        # ========== Step 1: 学習フェーズ ==========
        print("--------------------------------")
        print("学習フェーズ開始")

        # --- Train/Val/Test 分割 ---
        train_idx, val_idx = train_test_split(trainval_idx, test_size=0.1, random_state=seed * 100 + fold, stratify=y_ordinal[trainval_idx])
        print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")


        # ===== アンダーサンプリング処理 (train_idxのみ適用) =====
        train_df = df.iloc[train_idx].copy()

        print("Before undersampling (train only, categories 0-5):")
        print(train_df['days_until_next_category'].value_counts().sort_index())

        # 0〜5 の全カテゴリの件数を確認
        counts_all = train_df['days_until_next_category'].value_counts().sort_index()

        # 最小件数を基準にする
        target_counts = counts_all.min()

        sampled_dfs = []
        for cat in counts_all.index:   # 0〜5 全部対象
            cat_df = train_df[train_df['days_until_next_category'] == cat]
            sampled_df = cat_df.sample(n=target_counts, random_state=42)
            sampled_dfs.append(sampled_df)

        # 結合してシャッフル
        balanced_train_df = pd.concat(sampled_dfs, axis=0).sample(frac=1, random_state=42)

        # df 全体に対するインデックスに変換
        train_idx = balanced_train_df.index.values

        print("After undersampling (train only, categories 0-5):")
        print(balanced_train_df['days_until_next_category'].value_counts().sort_index())



        # ========== OLRの学習 ==========
        print("順序回帰モデル（Ordered Logistic Regression）の学習を開始...")
        print("Seed:", seed+1, "Fold:", fold+1)
        start_time = time.time()
        print("学習開始時刻：", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)))

        X_train = X.iloc[train_idx]
        y_ord_train = y_ordinal[train_idx]

        X_val = X.iloc[val_idx]
        y_ord_val = y_ordinal[val_idx]

        # 順序回帰モデルの学習
        # statsmodels の OrderedModel
        ord_model = OrderedModel(
            y_ord_train,
            X_train,
            distr='logit'
        )
        ord_res = ord_model.fit(method='bfgs', disp=False)
        # P値・係数を保存
        if fold == 0 and seed == 0:
            with open(os.path.join(result_dir, f"ordered_logit_summary_seed{seed+1}_fold{fold+1}.txt"), "w", encoding="utf-8") as f:
                f.write(str(ord_res.summary()))


        
        end_time = time.time()
        print("学習終了時刻：", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)))
        print(f"順序回帰モデル（Ordered Logistic Regression）の学習完了（経過時間: {end_time - start_time:.2f} 秒）")
        print("学習フェーズ完了")
        print("--------------------------------")


        # ========== Step 2: 推論 ==========
        print("--------------------------------")
        print("推論フェーズ開始")

        # Step 1: テストデータを取得
        X_test = X.iloc[test_idx]
        y_ord_test = y_ordinal[test_idx]

        # Step 2: 推論
        probs_ord = ord_res.predict(X_test)
        preds_ord_final = np.argmax(probs_ord.values, axis=1)



        print("推論フェーズ完了")
        print("--------------------------------")


        # ---------- Step 3: 評価 ----------
        print("--------------------------------")
        print("評価フェーズ開始")

        # --------- Step 3-1: 順序分類の評価 ----------
        y_true_days = df.loc[test_idx, "days_until_next"].values
        y_pred_days = np.array([label_to_midpoint[y] for y in preds_ord_final])
        mae = mean_absolute_error(y_true_days, y_pred_days)
        mse = mean_squared_error(y_true_days, y_pred_days)
        rmse = np.sqrt(mse)
        corr, _ = pearsonr(y_true_days, y_pred_days)

        print(f"[最終的な順序分類評価指標] MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, Corr: {corr:.4f}")


        # ===== Final preds (argmax後) に対する「各しきい値 y ≤ k」の二値分類指標（AUCなし） =====
        print("\n[Final (argmax) y ≤ k : Binary Classification Metrics]")
        for k in range(olr_output_dim):
            y_bin_true_k = (y_ord_test <= k).astype(int)
            y_bin_pred_k = (preds_ord_final <= k).astype(int)

            acc_k = accuracy_score(y_bin_true_k, y_bin_pred_k)
            precision_k = precision_score(y_bin_true_k, y_bin_pred_k, zero_division=0)
            recall_k = recall_score(y_bin_true_k, y_bin_pred_k, zero_division=0)
            f1_k = f1_score(y_bin_true_k, y_bin_pred_k, zero_division=0)

            final_bincls_scores[k]["accuracy"].append(acc_k)
            final_bincls_scores[k]["precision"].append(precision_k)
            final_bincls_scores[k]["recall"].append(recall_k)
            final_bincls_scores[k]["f1"].append(f1_k)

            print(f"y ≤ '{label_names[k]}' : "
                  f"Acc={acc_k:.4f}, Precision={precision_k:.4f}, "
                  f"Recall={recall_k:.4f}, F1={f1_k:.4f}")


        # olr の出力（二値タスクごとの評価）
        # OrderedModel 出力（各クラス確率）
        class_probs = probs_ord.to_numpy()  # shape: (n_samples, n_classes)

        # 累積ラベル生成（y <= k）
        y_ord_bin_targets = (y_ord_test[:, None] <= np.arange(num_ord_classes)[None, :]).astype(int)

        # 各クラス確率 → 累積確率（y <= k の確率）
        cum_probs = np.cumsum(class_probs, axis=1)[:, :-1]  # 最後のクラス以外が対象

        # 0.5 閾値で二値予測
        y_ord_bin_preds = (cum_probs > 0.5).astype(int)

        print("\n[olr内部の二値タスクごとの評価指標]")
        for k in range(olr_output_dim):
            acc_k = accuracy_score(y_ord_bin_targets[:, k], y_ord_bin_preds[:, k])
            precision_k = precision_score(y_ord_bin_targets[:, k], y_ord_bin_preds[:, k], zero_division=0)
            recall_k = recall_score(y_ord_bin_targets[:, k], y_ord_bin_preds[:, k], zero_division=0)
            f1_k = f1_score(y_ord_bin_targets[:, k], y_ord_bin_preds[:, k], zero_division=0)

            olr_per_task_scores[k]["accuracy"].append(acc_k)
            olr_per_task_scores[k]["precision"].append(precision_k)
            olr_per_task_scores[k]["recall"].append(recall_k)
            olr_per_task_scores[k]["f1"].append(f1_k)

            # --- AUC の追加 ---
            try:
                auc_k = roc_auc_score(y_ord_bin_targets[:, k], cum_probs[:, k])
            except ValueError:
                auc_k = np.nan  # 正例 or 負例が1クラスしかないと計算できない
            olr_auc_per_task[k].append(auc_k)

            print(f"Task y <= '{label_names[k]}': Acc={acc_k:.4f}, Precision={precision_k:.4f}, Recall={recall_k:.4f}, F1={f1_k:.4f}, AUC={auc_k:.4f}")



        # 順序回帰の指標をまとめて記録（“日数”ベース）
        metrics = compute_ordered_metrics(y_true_days, y_pred_days)
        metrics["Fold"] = fold + 1
        all_metrics.append(metrics)

        # 混同行列のためのカテゴリ保持は既存どおり
        all_y_true.extend(y_ord_test.tolist())
        all_y_pred.extend(preds_ord_final.tolist())

        # 統合フェーズ（散布図/相関/ヒスト）用に“日数”を蓄積
        all_true_days.extend(y_true_days.tolist())
        all_pred_days.extend(y_pred_days.tolist())


        print("評価フェーズ完了")

        print("--------------------------------")



print("--------------------------------")
print("統合評価の保存フェーズ開始")
# ========= 統合評価CSVの出力（完全版） =========
summary_all_path = os.path.join(result_dir, "metrics_all_summary_olr_5_undersample.csv")

with open(summary_all_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Category", "Averaging", "Metric", "Mean", "95% CI Lower", "95% CI Upper"])


    # 1. 順序回帰の評価指標
    for name in metric_names:
        values = [m[name] for m in all_metrics]
        mean, ci_lower, ci_upper = mean_ci(values)
        writer.writerow(["Ordinal", "-", name, mean, ci_lower, ci_upper])

    # 2. Final (argmax) に対する各しきい値 y ≤ k の二値指標（AUCなし）
    for k in range(olr_output_dim):
        label = label_names[k]
        for metric in ["accuracy", "precision", "recall", "f1"]:
            values = final_bincls_scores[k][metric]
            mean, ci_lower, ci_upper = mean_ci(values)
            writer.writerow(["OrdinalBinary(Final-Argmax)", f"y <= '{label}'", metric, mean, ci_lower, ci_upper])


    # 3. olr 順序回帰の各二値タスクごとのスコア
    for k in range(olr_output_dim):
        label = label_names[k]  # 例: '〜1 month'
        for metric in ["accuracy", "precision", "recall", "f1"]:
            values = olr_per_task_scores[k][metric]
            mean, ci_lower, ci_upper = mean_ci(values)
            writer.writerow(["OrdinalBinary", f"y <= '{label}'", metric, mean, ci_lower, ci_upper])

        # --- AUC の統合評価 ---
        auc_values = [v for v in olr_auc_per_task[k] if not np.isnan(v)]
        if len(auc_values) > 0:
            mean, ci_lower, ci_upper = mean_ci(auc_values)
            writer.writerow(["OrdinalBinary", f"y <= '{label}'", "AUC", mean, ci_lower, ci_upper])
    
    # 4. 順序回帰（実測日数 vs 予測日数）の Pearson 相関係数
    true_days_np = np.asarray(all_true_days, dtype=float)
    pred_days_np = np.asarray(all_pred_days, dtype=float)
    corr, pval = pearsonr(true_days_np, pred_days_np)
    writer.writerow(["Ordinal(days_until_next)", "-", "Pearson_corr_days", corr, "-", f"p={pval}"])




# ==== 順序分類 全fold統合の混同行列を作成・保存 ====
label_indices = list(range(len(label_names)))  # → [0, 1, 2, 3, 4, 5]

cm_all = confusion_matrix(all_y_true, all_y_pred, labels=label_indices)
cm_all_df = pd.DataFrame(cm_all, index=label_names, columns=label_names)

plt.figure(figsize=(10, 8))
sns.heatmap(cm_all_df, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix (All Folds)")
plt.tight_layout()
plt.savefig(os.path.join(result_dir, "confusion_matrix_all_folds_olr.png"))
plt.close()



# ==== 散布図（実測日数 vs 予測日数） ====
true_days_np = np.asarray(all_true_days, dtype=float)
pred_days_np = np.asarray(all_pred_days, dtype=float)

plt.figure(figsize=(8, 6))
plt.scatter(true_days_np, pred_days_np, alpha=0.3, s=20, edgecolor='none')
max_val = max(true_days_np.max(), pred_days_np.max())
plt.plot([0, max_val], [0, max_val], color='red', linestyle='--', linewidth=1.5, label="y=x")
plt.xlabel("True Days Until Next Registration")
plt.ylabel("Predicted Days Until Next Registration")
plt.title("True vs Predicted Days Until Next Registration (All Folds)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(result_dir, "scatter_true_vs_pred_days.png"))
plt.close()




# ==== 予測カテゴリごとの正解日数ヒストグラム ====
print("予測カテゴリごとの正解日数ヒストグラムを作成中...")

y_pred_cat_np = np.asarray(all_y_pred, dtype=int)         # 予測カテゴリ（0..5）
true_days_np   = np.asarray(all_true_days, dtype=float)    # 実測日数

hist_dir = os.path.join(result_dir, "hist_true_days_by_predcat")
os.makedirs(hist_dir, exist_ok=True)

# 個別（カテゴリごと1枚）
for cat in range(len(label_names)):  # 0..5
    mask = (y_pred_cat_np == cat)
    true_days_cat = true_days_np[mask]
    if true_days_cat.size == 0:
        continue

    plt.figure(figsize=(8, 4))
    plt.hist(true_days_cat, bins=50, alpha=0.8, edgecolor="black")
    plt.xlabel("True Days Until Next Registration")
    plt.ylabel("Count")
    plt.title(f"True Days Histogram (Predicted Category = {cat}: {label_names[cat]})")
    plt.tight_layout()
    plt.savefig(os.path.join(hist_dir, f"hist_true_days_by_predcat_{cat}.png"))
    plt.close()

# まとめ図（2x3）
fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
axes = axes.ravel()
for cat in range(len(label_names)):
    ax = axes[cat]
    mask = (y_pred_cat_np == cat)
    true_days_cat = true_days_np[mask]
    if true_days_cat.size > 0:
        ax.hist(true_days_cat, bins=50, alpha=0.85, edgecolor="black")
    ax.set_title(f"{cat}: {label_names[cat]}")

for i, ax in enumerate(axes):
    if i % 3 == 0:
        ax.set_ylabel("Count")
    if i >= 3:
        ax.set_xlabel("True Days Until Next Registration")

plt.suptitle("True Days per Predicted Category", y=1.02, fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(hist_dir, "hist_true_days_by_predcat_grid.png"))
plt.close()

print("予測カテゴリごとの正解日数ヒストグラムを保存しました。")


print("統合評価の保存フェーズ完了")
print("--------------------------------")
