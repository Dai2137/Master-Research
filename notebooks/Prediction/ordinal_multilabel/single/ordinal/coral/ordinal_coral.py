import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_absolute_error, mean_squared_error, cohen_kappa_score, confusion_matrix, classification_report
from scipy.stats import spearmanr, t, pearsonr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import time
from tqdm import tqdm, trange
import os
import matplotlib
matplotlib.use("Agg")  # GUIなしで描画できるバックエンドに変更
import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns
import csv
import copy
import shap

# 保存パスの指定と準備（変更するの忘れないように！！）
result_dir = r"D:\fujiwara\M\result\ordinal_multilabel\single\ordinal\coral\no_5_undersample"
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


# ---------- CORAL ----------
class CoralOrdinalNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.shared_weight = nn.Parameter(torch.randn(16))
        self.raw_bias = nn.Parameter(torch.zeros(num_classes - 1))
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x_shared = self.relu(self.fc3(x))
        logits = x_shared @ self.shared_weight
        logits = logits.unsqueeze(1)
        ordered_bias = torch.cumsum(F.softplus(self.raw_bias), dim=0)
        logits = logits + ordered_bias
        probs = torch.sigmoid(logits)
        return probs, x_shared

def coral_loss(probs, labels, num_classes):
    labels = labels.view(-1, 1)
    target = (torch.arange(num_classes - 1).to(labels.device) >= labels).float()
    return F.binary_cross_entropy(probs, target, reduction='mean')

def predict_classes(probs):
    batch_size = probs.shape[0]
    device = probs.device
    probs_ext = torch.cat([
        torch.zeros(batch_size, 1, device=device),
        probs,
        torch.ones(batch_size, 1, device=device)
    ], dim=1)
    probs_exact = probs_ext[:, 1:] - probs_ext[:, :-1]
    return torch.argmax(probs_exact, dim=1)

def predict_midpoint(model, X_df):
    """
    CORALモデルの出力から「各クラスの範囲中央値ベース」の予測値を返す関数。
    SHAP可視化に用いる。

    Parameters
    ----------
    model : torch.nn.Module
        学習済みのCORALモデル（foldごとに異なる）
    X_df : pd.DataFrame
        特徴量名付きDataFrame（.iloc[]で分割後のX_testなど）

    Returns
    -------
    np.ndarray : shape (N, 1)
        各サンプルの予測中央値（月単位）
    """
    model.eval()
    x_tensor = torch.tensor(X_df.values, dtype=torch.float32)
    label_to_midpoint = torch.tensor([0.5, 2.5, 6.5, 16.5, 72.0, 200.0], dtype=torch.float32)  # 各クラスの中央値

    with torch.no_grad():
        # CORAL出力: shape (N, K−1)
        probs, _ = model(x_tensor)

        # 累積確率に 0, 1 を追加して差分を取ることでクラス確率を得る (N, K)
        probs_ext = torch.cat([
            torch.zeros((probs.shape[0], 1)),
            probs,
            torch.ones((probs.shape[0], 1))
        ], dim=1)
        probs_exact = probs_ext[:, 1:] - probs_ext[:, :-1]

        # 各クラス確率 × 範囲中央値の加重平均 → 予測中央値（月単位）
        y_pred_mid = (probs_exact * label_to_midpoint).sum(dim=1)

    return y_pred_mid.cpu().numpy().reshape(-1, 1)


# ---------- データ読み込み ----------
print("データ読み込みを開始...")
df = pd.read_csv(r"D:\fujiwara\M\data\after_preprocess\land_data_for_prediction.csv")
print("データ読み込み完了")


X = df.drop(columns=['will_not_be_re_registered', 'days_until_next_category', 'days_until_next'] +
            [col for col in df.columns if col.startswith("on_day_reason_group_") and col.endswith("_next")]).astype(np.float32)

# y_binary = df['will_not_be_re_registered'].values 
y_ordinal = df['days_until_next_category'].values


# 順序クラスの数を決定
num_ord_classes_coral = len(np.unique(y_ordinal))  # ← これで「元のラベルの個数 = 6」が得られる
coral_output_dim = num_ord_classes_coral - 1  # = 5





# 評価指標格納用
all_metrics = []
# metric_names = ["Accuracy", "MAE", "MSE", "RMSE", "Spearman", "QWK"]
metric_names = ["MAE", "MSE", "RMSE", "Corr"]

# ループ外で定義（foldループの前）
all_y_true = []
all_y_pred = []

# 追加（統合フェーズで実測/予測“日数”を使うため）
all_true_days = []   # 実測日数（days_until_next）
all_pred_days = []   # 予測カテゴリを日数換算した値



# CORALの二値分類の評価指標格納用
coral_per_task_scores = {  # foldごとに保持
    k: {"accuracy": [], "precision": [], "recall": [], "f1": []}
    for k in range(coral_output_dim)
}


# CORALの二値分類のAUC格納用
coral_auc_per_task = {k: [] for k in range(coral_output_dim)}

# 最終予測カテゴリに対する「y ≤ k」二値化の評価（AUCなし）
final_bincls_scores = {
    k: {"accuracy": [], "precision": [], "recall": [], "f1": []}
    for k in range(coral_output_dim)  # 最後は全て1になるので除外
}


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

        print("Before undersampling (train only, all categories):")
        print(train_df['days_until_next_category'].value_counts().sort_index())

        counts_0to4 = train_df[train_df['days_until_next_category'].between(0, 4)]['days_until_next_category'].value_counts()
        min_cat = counts_0to4.idxmin()
        target_counts = counts_0to4.min()

        target_categories = [c for c in counts_0to4.index if c != min_cat]

        sampled_dfs = []
        for cat in target_categories:
            cat_df = train_df[train_df['days_until_next_category'] == cat]
            sampled_df = cat_df.sample(n=target_counts, random_state=42)
            sampled_dfs.append(sampled_df)

        other_df = train_df[~train_df['days_until_next_category'].isin(target_categories)]
        balanced_train_df = pd.concat(sampled_dfs + [other_df], axis=0).sample(frac=1, random_state=42)

        train_idx = balanced_train_df.index.values

        print("After undersampling (train only, all categories):")
        print(balanced_train_df['days_until_next_category'].value_counts().sort_index())




        # ========== CORALの学習 ==========
        print("CORALの学習を開始...")
        start_time = time.time()


        X_train = X.iloc[train_idx]
        y_ord_train = y_ordinal[train_idx]

        X_val = X.iloc[val_idx]
        y_ord_val = y_ordinal[val_idx]
        
        
        # torch tensor に変換
        X_train_mt = torch.tensor(X_train.values, dtype=torch.float32)
        y_ord_train = torch.tensor(y_ord_train, dtype=torch.long)

        X_val_mt = torch.tensor(X_val.values, dtype=torch.float32)
        y_ord_val = torch.tensor(y_ord_val, dtype=torch.long)

        # DataLoader
        train_dataset = TensorDataset(X_train_mt, y_ord_train)
        train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, generator=torch.Generator().manual_seed(seed * 100 + fold))

        # 損失記録用リスト
        train_losses = []
        val_losses = []


        model = CoralOrdinalNN(input_dim=X.shape[1], num_classes=num_ord_classes_coral)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


        num_epochs = 100

        # Early Stopping
        early_stop_patience = 10
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        

        for epoch in trange(num_epochs, desc="Training Epochs"):
            model.train()
            total_train_loss = 0
            for xb, y_ord in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
                optimizer.zero_grad()
                probs_ord, _ = model(xb)
                loss = coral_loss(probs_ord, y_ord, num_ord_classes_coral)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
            
            avg_train_loss = total_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # --- Validation損失の計算 ---
            model.eval()
            with torch.no_grad():
                probs_ord_val, _ = model(X_val_mt)
                val_loss = coral_loss(probs_ord_val, y_ord_val, num_ord_classes_coral)
                avg_val_loss = val_loss.item()
                val_losses.append(avg_val_loss)

            print(f"Epoch {epoch+1:3d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

            # Early Stopping チェック
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = copy.deepcopy(model.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break


        # ベストモデルを復元
        model.load_state_dict(best_model_state)
        
        # 保存ディレクトリを作成（なければ作る）
        learning_curve_dir = os.path.join(result_dir, "learning_curve")
        os.makedirs(learning_curve_dir, exist_ok=True)

        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Learning Curve (Seed {seed+1}, Fold {fold+1})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(learning_curve_dir, f"learning_curve_seed{seed+1}_fold{fold+1}.png"))
        plt.close()


        
        end_time = time.time()
        print(f"CORALの学習完了（経過時間: {end_time - start_time:.2f} 秒）")
        print("学習フェーズ完了")
        print("--------------------------------")


        # ========== Step 2: 推論 ==========
        print("--------------------------------")
        print("推論フェーズ開始")

        # Step 1: テストデータを取得
        X_test = X.iloc[test_idx]
        y_ord_test = y_ordinal[test_idx]

        # Step 2: 推論
        X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)

        with torch.no_grad():
            probs_ord_all, _ = model(X_test_tensor)
            preds_ord_all = predict_classes(probs_ord_all)

        preds_ord_final = preds_ord_all.numpy()

        if fold == 0 and seed == 0:
            print("SHAPの計算を開始...")
            # 保存ディレクトリを作成（なければ作る）
            shap_dir = os.path.join(result_dir, "shap_summary_plot")
            os.makedirs(shap_dir, exist_ok=True)

            print("CORALのSHAPの計算を開始...")
            
            predict_fn = lambda x: predict_midpoint(model, pd.DataFrame(x, columns=X_test.columns))

            explainer_coral = shap.Explainer(predict_fn, X_test, model_output="raw")
            shap_values_coral = explainer_coral(X_test)

            # 可視化
            shap.summary_plot(shap_values_coral, X_test, show=False)
            plt.title(f"SHAP Summary for Predicted Midpoint (months) Seed {seed+1} Fold {fold+1}")
            plt.tight_layout()
            plt.savefig(os.path.join(shap_dir, f"shap_summary_y_pred_mid_seed{seed+1}_fold{fold+1}.png"))
            plt.close()
            print("CORALのSHAPの計算完了（y_pred_midベース）")
            print("SHAPの計算完了")


        print("推論フェーズ完了")
        print("--------------------------------")


        # ---------- Step 3: 評価 ----------
        print("--------------------------------")
        print("評価フェーズ開始")

        # --------- Step 3-1: 順序分類の評価 ----------
        # acc = accuracy_score(y_ord_test, preds_ord_final)
        y_true_days = df.loc[test_idx, "days_until_next"].values
        y_pred_days = np.array([label_to_midpoint[y] for y in preds_ord_final])
        mae = mean_absolute_error(y_true_days, y_pred_days)
        mse = mean_squared_error(y_true_days, y_pred_days)
        rmse = np.sqrt(mse)
        corr, _ = pearsonr(y_true_days, y_pred_days)
        # spearman, _ = spearmanr(y_ord_test, preds_ord_final)
        # qwk = cohen_kappa_score(y_ord_test, preds_ord_final, weights='quadratic')

        print(f"[最終的な順序分類評価指標] MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, Corr: {corr:.4f}")



        # ===== Final preds (argmax 後) に対する「各しきい値 y ≤ k」の二値分類指標（AUCなし） =====
        print("\n[Final (argmax) y ≤ k : Binary Classification Metrics]")
        for k in range(coral_output_dim):
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



        # CORAL の出力（二値タスクごとの評価）
        probs_ord_np = probs_ord_all.cpu().numpy()
        y_ord_bin_targets = (np.arange(num_ord_classes_coral)[None, :] >= y_ord_test[:, None]).astype(int)
        y_ord_bin_preds = (probs_ord_all.cpu().numpy() > 0.5).astype(int)

        print("\n[CORAL内部の二値タスクごとの評価指標]")
        for k in range(coral_output_dim):
            acc_k = accuracy_score(y_ord_bin_targets[:, k], y_ord_bin_preds[:, k])
            precision_k = precision_score(y_ord_bin_targets[:, k], y_ord_bin_preds[:, k], zero_division=0)
            recall_k = recall_score(y_ord_bin_targets[:, k], y_ord_bin_preds[:, k], zero_division=0)
            f1_k = f1_score(y_ord_bin_targets[:, k], y_ord_bin_preds[:, k], zero_division=0)

            coral_per_task_scores[k]["accuracy"].append(acc_k)
            coral_per_task_scores[k]["precision"].append(precision_k)
            coral_per_task_scores[k]["recall"].append(recall_k)
            coral_per_task_scores[k]["f1"].append(f1_k)

            # --- AUC の追加 ---
            try:
                auc_k = roc_auc_score(y_ord_bin_targets[:, k], probs_ord_np[:, k])
            except ValueError:
                auc_k = np.nan  # 正例 or 負例が1クラスしかないと計算できない
            coral_auc_per_task[k].append(auc_k)

            print(f"Task y <= '{label_names[k]}': Acc={acc_k:.4f}, Precision={precision_k:.4f}, Recall={recall_k:.4f}, F1={f1_k:.4f}, AUC={auc_k:.4f}")



        # 順序分類の指標をまとめて記録（1〜6分類）
        metrics = compute_ordered_metrics(y_true_days, y_pred_days)
        metrics["Fold"] = fold + 1
        all_metrics.append(metrics)


        # ======= 順序分類の混同行列のための格納 =======
        all_y_true.extend(y_ord_test.tolist())
        all_y_pred.extend(preds_ord_final.tolist())

        # ======= 順序分類の散布図のための格納 =======
        all_true_days.extend(y_true_days.tolist())
        all_pred_days.extend(y_pred_days.tolist())




        print("評価フェーズ完了")

        print("--------------------------------")



print("--------------------------------")
print("統合評価の保存フェーズ開始")
# ========= 統合評価CSVの出力（完全版） =========
summary_all_path = os.path.join(result_dir, "metrics_all_summary_coral_no_5_undersample.csv")

with open(summary_all_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Category", "Averaging", "Metric", "Mean", "95% CI Lower", "95% CI Upper"])

    # 1. 順序回帰の評価指標
    for name in metric_names:
        values = [m[name] for m in all_metrics]
        mean, ci_lower, ci_upper = mean_ci(values)
        writer.writerow(["Ordinal", "-", name, mean, ci_lower, ci_upper])


    # 2. Final (argmax) に対する各しきい値 y ≤ k の二値指標（AUCなし）
    for k in range(coral_output_dim):
        label = label_names[k]
        for metric in ["accuracy", "precision", "recall", "f1"]:
            values = final_bincls_scores[k][metric]
            mean, ci_lower, ci_upper = mean_ci(values)
            writer.writerow(["OrdinalBinary(Final-Argmax)", f"y <= '{label}'", metric, mean, ci_lower, ci_upper])

    # 3. CORAL 順序回帰の各二値タスクごとのスコア
    for k in range(coral_output_dim):
        label = label_names[k]  # 例: '〜1 month'
        for metric in ["accuracy", "precision", "recall", "f1"]:
            values = coral_per_task_scores[k][metric]
            mean, ci_lower, ci_upper = mean_ci(values)
            writer.writerow(["OrdinalBinary", f"y <= '{label}'", metric, mean, ci_lower, ci_upper])

        # --- AUC の統合評価 ---
        auc_values = [v for v in coral_auc_per_task[k] if not np.isnan(v)]
        if len(auc_values) > 0:
            mean, ci_lower, ci_upper = mean_ci(auc_values)
            writer.writerow(["OrdinalBinary", f"y <= '{label}'", "AUC", mean, ci_lower, ci_upper])
    
    # 4. 順序回帰 (日数マッピング後) の相関係数
    true_days_np = np.asarray(all_true_days, dtype=float)
    pred_days_np = np.asarray(all_pred_days, dtype=float)
    corr, pval = pearsonr(true_days_np, pred_days_np)
    writer.writerow(["Ordinal", "-", "Pearson_corr_days", corr, "-", f"p={pval}"])




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
plt.savefig(os.path.join(result_dir, "confusion_matrix_all_folds_ordinal_coral.png"))
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
