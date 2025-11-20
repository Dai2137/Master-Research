import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_absolute_error, mean_squared_error, cohen_kappa_score, confusion_matrix, classification_report
from scipy.stats import spearmanr, t
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


# 保存パスの指定と準備（変更するの忘れないように！！）
result_dir = r"D:\fujiwara\M\result\binary+ordinal_multilabel\single\binary+ordinal\lr+coral"
os.makedirs(result_dir, exist_ok=True)



# ---------- Utility Functions ----------
def compute_ordered_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    spearman, _ = spearmanr(y_true, y_pred)
    qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    
    return {
        "Accuracy": acc,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "Spearman": spearman,
        "QWK": qwk
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
        self.fc3 = nn.Linear(32, 16)  # ← 新たな16次元の隠れ層を追加
        self.shared_weight = nn.Parameter(torch.randn(16))  # ← 対応する重みも16次元に
        self.raw_bias = nn.Parameter(torch.zeros(num_classes - 1))
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x_shared = self.relu(self.fc3(x))  # ← 16次元に変換された共有表現
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


# ---------- データ読み込み ----------
print("データ読み込みを開始...")
df = pd.read_csv(r"D:\fujiwara\M\data\after_preprocess\land_data_for_prediction.csv")
print("データ読み込み完了")


X = df.drop(columns=['will_not_be_re_registered', 'days_until_next_category'] +
            [col for col in df.columns if col.startswith("on_day_reason_group_") and col.endswith("_next")]).values

y_binary = df['will_not_be_re_registered'].values
y_ordinal = df['days_until_next_category'].values

# 順序クラスの数を決定
num_ord_classes = len(np.unique(y_ordinal))  # ← これで「元のラベルの個数 = 6」が得られる
num_ord_classes_coral = num_ord_classes - 1 # 0〜4の5クラス
coral_output_dim = num_ord_classes_coral - 1 # 0以下〜3以下の4ニューロン


binary_metrics = {
    "accuracy": [],
    "precision": [],
    "recall": [],
    "f1": [],
    "auc": []
}


# 評価指標格納用
all_metrics = []
metric_names = ["Accuracy", "MAE", "MSE", "RMSE", "Spearman", "QWK"]

all_y_bin_true = []
all_y_bin_pred = []

# ループ外で定義（foldループの前）
all_y_true = []
all_y_pred = []


# 最終予測カテゴリに対する「y ≤ k」二値化の評価（AUCなし）
final_bincls_scores = {
    k: {"accuracy": [], "precision": [], "recall": [], "f1": []}
    for k in range(coral_output_dim)  # 最後は全て1になるので除外
}



# CORALの二値分類の評価指標格納用
coral_per_task_scores = {  # foldごとに保持
    k: {"accuracy": [], "precision": [], "recall": [], "f1": []}
    for k in range(coral_output_dim)
}


# CORALの二値分類のAUC格納用
coral_auc_per_task = {k: [] for k in range(coral_output_dim)}


label_names = [
    '-1 months', 
    '1-4 months', 
    '4-9 months', 
    '9-24 months', 
    '24-120 months', 
    '120- months'
]



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

        # ===== アンダーサンプリング処理 (train_idxのみ適用, 0-4対象) =====
        train_df = df.iloc[train_idx].copy()

        print("Before undersampling (train only, categories 0-4):")
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

        print("After undersampling (train only, categories 0-4):")
        print(balanced_train_df['days_until_next_category'].value_counts().sort_index())


        # ========== Step 1-1: 二値分類器の学習 ==========
        print("二値分類モデル（ロジスティック回帰）の学習を開始...")

        # チューニング対象パラメータ
        param_grid_lr = {
            "C": [0.01, 0.1, 1.0, 10.0],
            "penalty": ["l2"],
            "solver": ["liblinear"],  # small dataset
            "class_weight": ["balanced"]
        }

        best_score = -np.inf
        best_params_lr = None

        # グリッドサーチ（valで評価）
        for C in param_grid_lr["C"]:
            for penalty in param_grid_lr["penalty"]:
                for solver in param_grid_lr["solver"]:
                    for cw in param_grid_lr["class_weight"]:
                        model = LogisticRegression(
                            C=C, penalty=penalty, solver=solver, class_weight=cw, max_iter=1000, random_state=seed * 100 + fold
                        )
                        model.fit(X[train_idx], y_binary[train_idx])
                        val_preds = model.predict(X[val_idx])
                        score = f1_score(y_binary[val_idx], val_preds)
                        if score > best_score:
                            best_score = score
                            best_params_lr = {
                                "C": C, "penalty": penalty, "solver": solver, "class_weight": cw
                            }
        print("Logistic Regression 最適なハイパーパラメータ:", best_params_lr)

        # ベストモデルを train_idx で再学習
        clf_bin_lr = LogisticRegression(**best_params_lr, max_iter=1000, random_state=seed * 100 + fold)
        clf_bin_lr.fit(X[train_idx], y_binary[train_idx])

        print("二値分類モデルの学習完了")

        # ========== Step 1-2: CORALの学習 ==========
        print("CORALの学習を開始...")
        start_time = time.time()

        # 再登記されるものだけ使う（binary=0）
        mask_train = y_binary[train_idx] == 0
        X_train = X[train_idx][mask_train]
        y_ord_train = y_ordinal[train_idx][mask_train]

        # 再登記されるものだけ使う（binary=0）
        mask_val = y_binary[val_idx] == 0
        X_val = X[val_idx][mask_val]
        y_ord_val = y_ordinal[val_idx][mask_val]


        # torch tensor に変換
        X_train_mt = torch.tensor(X_train, dtype=torch.float32)
        y_ord_train = torch.tensor(y_ord_train, dtype=torch.long)

        X_val_mt = torch.tensor(X_val, dtype=torch.float32)
        y_ord_val = torch.tensor(y_ord_val, dtype=torch.long)

        # DataLoader
        train_dataset = TensorDataset(X_train_mt, y_ord_train)
        train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, generator=torch.Generator().manual_seed(seed * 100 + fold))

        # 損失記録用リスト
        train_losses = []
        val_losses = []


        model = CoralOrdinalNN(input_dim=X.shape[1], num_classes=num_ord_classes_coral)

        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)


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
        X_test = X[test_idx]
        y_bin_test = y_binary[test_idx]
        y_ord_test = y_ordinal[test_idx]

        # Step 2: 二値分類予測
        y_bin_pred = clf_bin_lr.predict(X_test)
        mask_bin0 = y_bin_pred == 0

        # Step 3: NNの対象だけ取り出して推論
        X_test_masked = torch.tensor(X_test[mask_bin0], dtype=torch.float32)

        with torch.no_grad():
            probs_ord_masked, _ = model(X_test_masked)
            preds_ord_masked = predict_classes(probs_ord_masked)

        # Step 4: 統合予測ラベルの作成（順序分類）
        no_re_registration_class = label_names.index('120- months')  # → 5
        preds_ord_final = np.full_like(y_ord_test, fill_value=no_re_registration_class)
        preds_ord_final[mask_bin0] = preds_ord_masked.numpy()



        print("推論フェーズ完了")
        print("--------------------------------")


        # ---------- Step 3: 評価 ----------
        print("--------------------------------")
        print("評価フェーズ開始")

        # 二値分類の評価
        acc_bin = accuracy_score(y_bin_test, y_bin_pred)
        precision_bin = precision_score(y_bin_test, y_bin_pred, zero_division=0)
        recall_bin = recall_score(y_bin_test, y_bin_pred, zero_division=0)
        f1_bin = f1_score(y_bin_test, y_bin_pred, zero_division=0)
        auc_bin = roc_auc_score(y_bin_test, clf_bin_lr.predict_proba(X_test)[:, 1])

        print(f"[Binary Classification] Acc={acc_bin:.4f}, Precision={precision_bin:.4f}, Recall={recall_bin:.4f}, F1={f1_bin:.4f}, AUC={auc_bin:.4f}")
        
        binary_metrics["accuracy"].append(acc_bin)
        binary_metrics["precision"].append(precision_bin)
        binary_metrics["recall"].append(recall_bin)
        binary_metrics["f1"].append(f1_bin)
        binary_metrics["auc"].append(auc_bin)

        # 二値分類の混同行列のための格納
        all_y_bin_true.extend(y_bin_test.tolist())
        all_y_bin_pred.extend(y_bin_pred.tolist())



        # --------- Step 3-1: 順序分類の評価 ----------
        acc = accuracy_score(y_ord_test, preds_ord_final)
        mae = mean_absolute_error(y_ord_test, preds_ord_final)
        mse = mean_squared_error(y_ord_test, preds_ord_final)
        rmse = np.sqrt(mse)
        spearman, _ = spearmanr(y_ord_test, preds_ord_final)
        qwk = cohen_kappa_score(y_ord_test, preds_ord_final, weights='quadratic')

        print(f"[最終的な順序分類評価指標] Accuracy: {acc:.4f}, MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, Spearman: {spearman:.4f}, QWK: {qwk:.4f}")

        # ===== Final preds (argmax後) に対する「各しきい値 y ≤ k」の二値分類指標（AUCなし） =====
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
        if mask_bin0.sum() > 0:
            probs_ord_np = probs_ord_masked.cpu().numpy()
            y_ord_bin_targets = (np.arange(num_ord_classes_coral)[None, :] >= (y_ord_test[mask_bin0][:, None])).astype(int)
            y_ord_bin_preds = (probs_ord_np > 0.5).astype(int)

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

                # --- AUC の追加 ---er
                try:
                    auc_k = roc_auc_score(y_ord_bin_targets[:, k], probs_ord_np[:, k])
                except ValueError:
                    auc_k = np.nan  # 正例 or 負例が1クラスしかないと計算できない
                coral_auc_per_task[k].append(auc_k)

                print(f"Task y <= '{label_names[k]}': Acc={acc_k:.4f}, Precision={precision_k:.4f}, Recall={recall_k:.4f}, F1={f1_k:.4f}, AUC={auc_k:.4f}")



        # 順序分類の指標をまとめて記録（1〜6分類）
        metrics = compute_ordered_metrics(y_ord_test, preds_ord_final)
        metrics["Fold"] = fold + 1
        all_metrics.append(metrics)


        # ======= 順序分類の混同行列のための格納 =======
        all_y_true.extend(y_ord_test.tolist())
        all_y_pred.extend(preds_ord_final.tolist())


        print("評価フェーズ完了")

        print("--------------------------------")



print("--------------------------------")
print("統合評価の保存フェーズ開始")
# ========= 統合評価CSVの出力（完全版） =========
summary_all_path = os.path.join(result_dir, "metrics_all_summary_lr_coral.csv")

with open(summary_all_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Category", "Averaging", "Metric", "Mean", "95% CI Lower", "95% CI Upper"])

    # 1. 二値分類の評価指標
    for metric in ["accuracy", "precision", "recall", "f1", "auc"]:
        values = binary_metrics[metric]
        mean, ci_lower, ci_upper = mean_ci(values)
        writer.writerow(["Binary", "-", metric, mean, ci_lower, ci_upper])

    # 2. 順序回帰の評価指標
    for name in metric_names:
        values = [m[name] for m in all_metrics]
        mean, ci_lower, ci_upper = mean_ci(values)
        writer.writerow(["Ordinal", "-", name, mean, ci_lower, ci_upper])
    
    # 3. Final (argmax) に対する各しきい値 y ≤ k の二値指標（AUCなし）
    for k in range(coral_output_dim):
        label = label_names[k]
        for metric in ["accuracy", "precision", "recall", "f1"]:
            values = final_bincls_scores[k][metric]
            mean, ci_lower, ci_upper = mean_ci(values)
            writer.writerow(["OrdinalBinary(Final-Argmax)", f"y <= '{label}'", metric, mean, ci_lower, ci_upper])


    # 4. CORAL 順序回帰の各二値タスクごとのスコア
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


# ==== 二値分類 全fold統合の混同行列を作成・保存 ====
cm_bin_all = confusion_matrix(all_y_bin_true, all_y_bin_pred, labels=[0, 1])
cm_bin_all_df = pd.DataFrame(cm_bin_all, index=["True: 0", "True: 1"], columns=["Pred: 0", "Pred: 1"])

plt.figure(figsize=(8, 6))
sns.heatmap(cm_bin_all_df, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix (Binary Classification - All Folds)")
plt.tight_layout()
plt.savefig(os.path.join(result_dir, "confusion_matrix_all_folds_lr.png"))
plt.close()


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
plt.savefig(os.path.join(result_dir, "confusion_matrix_all_folds_lr_coral.png"))
plt.close()

print("統合評価の保存フェーズ完了")
print("--------------------------------")
