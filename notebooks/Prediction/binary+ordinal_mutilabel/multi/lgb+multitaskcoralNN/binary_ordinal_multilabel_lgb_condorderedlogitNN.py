import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, confusion_matrix, classification_report
)
from scipy.stats import pearsonr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import time
from tqdm import tqdm, trange
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns
import csv
import copy

# =========================
# Path
# =========================
result_dir = r"D:\fujiwara\M\result\binary+ordinal_multilabel\multi\condorderdlogitNN"
os.makedirs(result_dir, exist_ok=True)

# =========================
# Utility
# =========================
def mean_std(values):
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))
    return mean, std

def coral_loss(probs, labels, num_classes):
    """
    probs: (N, K-1) cumulative probs (sigmoid)
    labels: (N,) integer ordinal label in [0, K-1]
    """
    labels = labels.view(-1, 1)
    target = (torch.arange(num_classes - 1, device=labels.device) >= labels).float()
    return F.binary_cross_entropy(probs, target, reduction="mean")

def predict_classes(probs):
    """
    probs: (N, K-1) cumulative probs P(y <= k)
    return: (N,) argmax of exact probs
    """
    batch_size = probs.shape[0]
    device = probs.device
    probs_ext = torch.cat(
        [torch.zeros(batch_size, 1, device=device), probs, torch.ones(batch_size, 1, device=device)],
        dim=1
    )
    probs_exact = probs_ext[:, 1:] - probs_ext[:, :-1]  # (N, K)
    return torch.argmax(probs_exact, dim=1)

def predict_midpoint_from_coral_probs(probs, label_to_midpoint_tensor):
    """
    probs: (N, K-1) cumulative
    label_to_midpoint_tensor: (K,) midpoint per class (days)
    """
    device = probs.device
    n = probs.shape[0]
    probs_ext = torch.cat(
        [torch.zeros(n, 1, device=device), probs, torch.ones(n, 1, device=device)],
        dim=1
    )
    probs_exact = probs_ext[:, 1:] - probs_ext[:, :-1]  # (N, K)
    y_pred_mid = (probs_exact * label_to_midpoint_tensor.to(device)).sum(dim=1)
    return y_pred_mid


class MultiLabelCondOrderedLogit(nn.Module):
    """
    shared -> (multilabel hidden, ordinal hidden)
           -> concat(ordinal_hidden, multilabel_prob)
           -> fuse MLP
           -> Ordered Logistic (cumulative link) for ordinal output
    """
    def __init__(
        self,
        input_dim: int,
        num_ord_classes: int,      # K
        num_multilabels: int,      # L
        d_shared: int = 32,
        d_multi: int = 16,
        d_ord: int = 16,
        d_fuse: int = 16,
    ):
        super().__init__()
        self.K = num_ord_classes
        self.L = num_multilabels

        # ---- shared trunk ----
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, d_shared)
        self.relu = nn.ReLU()

        # ---- branch: multilabel ----
        self.multi_fc = nn.Linear(d_shared, d_multi)
        self.multi_out = nn.Linear(d_multi, num_multilabels)

        # ---- branch: ordinal hidden ----
        self.ord_fc = nn.Linear(d_shared, d_ord)

        # ---- fuse ----
        self.fuse1 = nn.Linear(d_ord + num_multilabels, 64)
        self.fuse2 = nn.Linear(64, d_fuse)

        # ---- ordered logistic head ----
        # latent score η = w^T h_fuse + b
        self.score = nn.Linear(d_fuse, 1)

        # thresholds θ (monotonic)
        self.raw_theta = nn.Parameter(torch.zeros(num_ord_classes - 1))

    def forward(self, x, detach_multilabel_for_ordinal: bool = False):
        # shared
        h = self.relu(self.fc1(x))
        h_shared = self.relu(self.fc2(h))  # (N, d_shared)

        # multilabel branch
        h_multi = self.relu(self.multi_fc(h_shared))          # (N, d_multi)
        p_multi = torch.sigmoid(self.multi_out(h_multi))      # (N, L)

        # ordinal branch hidden
        h_ord = self.relu(self.ord_fc(h_shared))              # (N, d_ord)

        # concat & fuse
        p_feat = p_multi.detach() if detach_multilabel_for_ordinal else p_multi
        z = torch.cat([h_ord, p_feat], dim=1)                 # (N, d_ord+L)
        z = self.relu(self.fuse1(z))                          # (N, 64)
        h_fuse = self.relu(self.fuse2(z))                     # (N, d_fuse)

        # ordered logistic (cumulative probs)
        eta = self.score(h_fuse)                              # (N, 1)
        theta = torch.cumsum(F.softplus(self.raw_theta), dim=0)  # (K-1,)

        # P(y <= k) = sigmoid(theta_k - eta)
        logits = theta.unsqueeze(0) - eta                     # (N, K-1)
        p_ord = torch.sigmoid(logits)                         # (N, K-1)

        return p_ord, p_multi


# =========================
# Data
# =========================
print("データ読み込みを開始...")
df = pd.read_csv(r"D:\fujiwara\M\data\after_preprocess\land_data_for_prediction.csv")
print("データ読み込み完了")

multilabel_colnames = [c for c in df.columns if c.startswith("on_day_reason_group_") and c.endswith("_next")]

X = df.drop(
    columns=["will_not_be_re_registered", "days_until_next_category", "days_until_next"] + multilabel_colnames
).astype(np.float32)

y_binary = df["will_not_be_re_registered"].values
y_ordinal = df["days_until_next_category"].values
y_multilabel = df[multilabel_colnames].values

# ordinal classes
num_ord_classes = len(np.unique(y_ordinal))          # usually 6
num_ord_classes_coral = num_ord_classes - 1          # exclude "no re-registration" bucket from NN training (0..4)
coral_output_dim = num_ord_classes_coral - 1         # K-1 internal tasks (usually 4)

label_names = [
    "-1 months",
    "1-4 months",
    "4-9 months",
    "9-24 months",
    "24-120 months",
    "120- months",
]

# Midpoints (days) for final "days" evaluation
label_to_midpoint = {
    0: 31 * 0.5,     # <1 month
    1: 31 * 2.5,     # 1–4
    2: 31 * 6.5,     # 4–9
    3: 31 * 16.5,    # 9–24
    4: 31 * 72.0,    # 24–120
    5: 31 * 120.0,   # >120 (no re-registration bucket)
}
label_to_midpoint_tensor = torch.tensor(
    [label_to_midpoint[k] for k in range(len(label_names))],
    dtype=torch.float32
)

# =========================
# Metrics containers
# =========================
binary_metrics = {k: [] for k in ["accuracy", "precision", "recall", "f1", "auc"]}

metric_names = ["MAE", "MSE", "RMSE", "Corr"]
all_metrics = []

all_y_bin_true, all_y_bin_pred = [], []
all_y_true, all_y_pred = [], []

all_true_days, all_pred_days = [], []

# multilabel metrics
multilabel_macro_metrics = {"precision": [], "recall": [], "f1-score": []}
multilabel_weighted_metrics = {"precision": [], "recall": [], "f1-score": []}
per_label_scores = {lab: {"precision": [], "recall": [], "f1-score": []} for lab in multilabel_colnames}

# coral per-task (internal binary tasks)
coral_per_task_scores = {k: {"accuracy": [], "precision": [], "recall": [], "f1": []} for k in range(coral_output_dim)}
coral_auc_per_task = {k: [] for k in range(coral_output_dim)}

# final (argmax) thresholded binary eval (AUCなし)
final_bincls_scores = {k: {"accuracy": [], "precision": [], "recall": [], "f1": []} for k in range(coral_output_dim)}

# time logs
time_logs = []

# =========================
# Training config
# =========================
seeds = list(range(3))  # file0 と合わせて 3
num_epochs = 100
early_stop_patience = 10
batch_size = 1024
lr_nn = 5e-4

# マルチタスク損失の重み
lambda_multilabel = 1.00

# =========================
# CV
# =========================
for seed in seeds:
    print(f"========== Seed {seed + 1} ==========")
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    for fold, (trainval_idx, test_idx) in enumerate(kf.split(X, y_ordinal)):
        print(f"\n========== Seed {seed + 1} Fold {fold+1} ==========")

        # -------------------------
        # split train/val/test
        # -------------------------
        train_idx, val_idx = train_test_split(
            trainval_idx,
            test_size=0.1,
            random_state=seed * 100 + fold,
            stratify=y_ordinal[trainval_idx]
        )
        print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

        # -------------------------
        # undersampling (train only, 0..4)
        # -------------------------
        train_df = df.iloc[train_idx].copy()

        print("Before undersampling (train only, all categories):")
        print(train_df["days_until_next_category"].value_counts().sort_index())

        counts_0to4 = train_df[train_df["days_until_next_category"].between(0, 4)]["days_until_next_category"].value_counts()
        min_cat = counts_0to4.idxmin()
        target_counts = counts_0to4.min()
        target_categories = [c for c in counts_0to4.index if c != min_cat]

        sampled_dfs = []
        for cat in target_categories:
            cat_df = train_df[train_df["days_until_next_category"] == cat]
            sampled_dfs.append(cat_df.sample(n=target_counts, random_state=42))

        other_df = train_df[~train_df["days_until_next_category"].isin(target_categories)]
        balanced_train_df = pd.concat(sampled_dfs + [other_df], axis=0).sample(frac=1, random_state=42)
        train_idx = balanced_train_df.index.values

        print("After undersampling (train only, all categories):")
        print(balanced_train_df["days_until_next_category"].value_counts().sort_index())

        # ============================================================
        # Step 1-1: Binary (LightGBM)
        # ============================================================
        print("二値分類モデル（LightGBM）の学習を開始...")

        param_grid_lgb = {
            "num_leaves": [15, 31, 63],
            "max_depth": [-1, 5, 10],
            "learning_rate": [0.05, 0.1, 0.2],
            "n_estimators": [100, 200, 300],
            "class_weight": ["balanced"],
        }

        best_score = -np.inf
        best_params_lgb = None

        for num_leaves in param_grid_lgb["num_leaves"]:
            for max_depth in param_grid_lgb["max_depth"]:
                for lr in param_grid_lgb["learning_rate"]:
                    for n_est in param_grid_lgb["n_estimators"]:
                        for cw in param_grid_lgb["class_weight"]:
                            model = LGBMClassifier(
                                num_leaves=num_leaves,
                                max_depth=max_depth,
                                learning_rate=lr,
                                n_estimators=n_est,
                                class_weight=cw,
                                random_state=seed * 100 + fold,
                                n_jobs=-1
                            )
                            model.fit(X.iloc[train_idx], y_binary[train_idx])
                            val_preds = model.predict(X.iloc[val_idx])
                            score = f1_score(y_binary[val_idx], val_preds, pos_label=0)
                            if score > best_score:
                                best_score = score
                                best_params_lgb = {
                                    "num_leaves": num_leaves,
                                    "max_depth": max_depth,
                                    "learning_rate": lr,
                                    "n_estimators": n_est,
                                    "class_weight": cw
                                }

        print("LightGBM 最適なハイパーパラメータ:", best_params_lgb)

        t0 = time.perf_counter()
        clf_bin_lgb = LGBMClassifier(**best_params_lgb, random_state=seed * 100 + fold, n_jobs=-1)
        clf_bin_lgb.fit(X.iloc[train_idx], y_binary[train_idx])
        t1 = time.perf_counter()
        lgb_train_time = t1 - t0
        print("二値分類モデルの学習完了")

        # ============================================================
        # Step 1-2: MultiTask NN (CORAL ordinal + multilabel)
        #   train only on y_binary==0 (re-registered)
        # ============================================================
        print("マルチタスクNN（CORAL + MultiLabel）の学習を開始...")
        t0 = time.perf_counter()

        mask_train = y_binary[train_idx] == 0
        X_train = X.iloc[train_idx].loc[mask_train]
        y_ord_train = y_ordinal[train_idx][mask_train]
        y_multi_train = y_multilabel[train_idx][mask_train]

        mask_val = y_binary[val_idx] == 0
        X_val = X.iloc[val_idx].loc[mask_val]
        y_ord_val = y_ordinal[val_idx][mask_val]
        y_multi_val = y_multilabel[val_idx][mask_val]

        X_train_t = torch.tensor(X_train.values, dtype=torch.float32)
        y_ord_train_t = torch.tensor(y_ord_train, dtype=torch.long)
        y_multi_train_t = torch.tensor(y_multi_train, dtype=torch.float32)

        X_val_t = torch.tensor(X_val.values, dtype=torch.float32)
        y_ord_val_t = torch.tensor(y_ord_val, dtype=torch.long)
        y_multi_val_t = torch.tensor(y_multi_val, dtype=torch.float32)

        train_dataset = TensorDataset(X_train_t, y_ord_train_t, y_multi_train_t)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed * 100 + fold)
        )

        
        model_mt = MultiLabelCondOrderedLogit(
            input_dim=X.shape[1],
            num_ord_classes=num_ord_classes_coral,      # K (re-registered側のクラス数)
            num_multilabels=y_multilabel.shape[1],
        )



        optimizer = torch.optim.Adam(model_mt.parameters(), lr=lr_nn)
        loss_fn_multi = nn.BCELoss()

        best_val_loss = float("inf")
        patience_counter = 0
        best_model_state = None

        for epoch in trange(num_epochs, desc="Training Epochs"):
            model_mt.train()
            total_train_loss = 0.0

            for xb, yord_b, ymulti_b in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
                optimizer.zero_grad()

                probs_ord, probs_multi = model_mt(xb)
                loss_ord = coral_loss(probs_ord, yord_b, num_ord_classes_coral)
                loss_multi = loss_fn_multi(probs_multi, ymulti_b)

                loss = loss_ord + lambda_multilabel * loss_multi
                loss.backward()
                optimizer.step()

                total_train_loss += float(loss.item())

            avg_train_loss = total_train_loss / max(len(train_loader), 1)

            # validation
            model_mt.eval()
            with torch.no_grad():
                probs_ord_val, probs_multi_val = model_mt(X_val_t)
                val_loss_ord = coral_loss(probs_ord_val, y_ord_val_t, num_ord_classes_coral)
                val_loss_multi = loss_fn_multi(probs_multi_val, y_multi_val_t)
                avg_val_loss = float((val_loss_ord + lambda_multilabel * val_loss_multi).item())

            print(f"Epoch {epoch+1:3d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = copy.deepcopy(model_mt.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break

        if best_model_state is not None:
            model_mt.load_state_dict(best_model_state)

        t1 = time.perf_counter()
        multitask_train_time = t1 - t0
        print("マルチタスクNNの学習完了")

        # ============================================================
        # Step 2: Inference
        # ============================================================
        print("推論フェーズ開始")

        X_test = X.iloc[test_idx]
        y_bin_test = y_binary[test_idx]
        y_ord_test = y_ordinal[test_idx]
        y_multi_test = y_multilabel[test_idx]

        # binary inference
        t0 = time.perf_counter()
        y_bin_pred = clf_bin_lgb.predict(X_test)
        t1 = time.perf_counter()
        lgb_inference_time = t1 - t0

        mask_bin0 = y_bin_pred == 0  # NN対象

        # multitask inference (masked)
        X_test_masked = torch.tensor(X_test[mask_bin0].values, dtype=torch.float32)

        t0 = time.perf_counter()
        with torch.no_grad():
            probs_ord_masked, probs_multi_masked = model_mt(X_test_masked)
            preds_ord_masked = predict_classes(probs_ord_masked)
            preds_multi_masked = (probs_multi_masked > 0.5).int().cpu().numpy()
        t1 = time.perf_counter()
        multitask_inference_time = t1 - t0

        # merge ordinal
        no_re_registration_class = label_names.index("120- months")  # 5
        preds_ord_final = np.full_like(y_ord_test, fill_value=no_re_registration_class)
        preds_ord_final[mask_bin0] = preds_ord_masked.cpu().numpy()

        # merge multilabel
        preds_multi_final = np.zeros_like(y_multi_test)
        preds_multi_final[mask_bin0] = preds_multi_masked

        print("推論フェーズ完了")

        # ============================================================
        # Step 3: Evaluation
        # ============================================================
        print("評価フェーズ開始")

        # ---- binary metrics ----
        acc_bin = accuracy_score(y_bin_test, y_bin_pred)
        precision_bin = precision_score(y_bin_test, y_bin_pred, pos_label=0, zero_division=0)
        recall_bin = recall_score(y_bin_test, y_bin_pred, pos_label=0, zero_division=0)
        f1_bin = f1_score(y_bin_test, y_bin_pred, pos_label=0, zero_division=0)
        # auc_bin = roc_auc_score(y_bin_test, clf_bin_lgb.predict_proba(X_test)[:, 1])
        proba0 = clf_bin_lgb.predict_proba(X_test)[:, 0]
        y0 = (y_bin_test == 0).astype(int)
        auc_bin = roc_auc_score(y0, proba0)


        print(f"[Binary] Acc={acc_bin:.4f}, Precision={precision_bin:.4f}, Recall={recall_bin:.4f}, F1={f1_bin:.4f}, AUC={auc_bin:.4f}")

        binary_metrics["accuracy"].append(acc_bin)
        binary_metrics["precision"].append(precision_bin)
        binary_metrics["recall"].append(recall_bin)
        binary_metrics["f1"].append(f1_bin)
        binary_metrics["auc"].append(auc_bin)

        all_y_bin_true.extend(y_bin_test.tolist())
        all_y_bin_pred.extend(y_bin_pred.tolist())

        # ---- ordinal: days metrics (midpoint) ----
        y_true_days = df.loc[test_idx, "days_until_next"].values
        y_pred_days = np.array([label_to_midpoint[int(y)] for y in preds_ord_final], dtype=float)

        mae = mean_absolute_error(y_true_days, y_pred_days)
        mse = mean_squared_error(y_true_days, y_pred_days)
        rmse = float(np.sqrt(mse))
        corr, _ = pearsonr(y_true_days, y_pred_days)

        print(f"[Ordinal(days)] MAE={mae:.4f}, MSE={mse:.4f}, RMSE={rmse:.4f}, Corr={corr:.4f}")

        all_metrics.append({"MAE": mae, "MSE": mse, "RMSE": rmse, "Corr": corr, "Fold": fold + 1})

        all_y_true.extend(y_ord_test.tolist())
        all_y_pred.extend(preds_ord_final.tolist())

        all_true_days.extend(y_true_days.tolist())
        all_pred_days.extend(y_pred_days.tolist())

        # ---- final (argmax) threshold binary metrics (AUCなし) ----
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

        # ---- CORAL internal tasks metrics (only masked subset) ----
        if mask_bin0.sum() > 0:
            probs_ord_np = probs_ord_masked.cpu().numpy()
            y_ord_bin_targets = (np.arange(num_ord_classes_coral)[None, :] >= (y_ord_test[mask_bin0][:, None])).astype(int)
            y_ord_bin_preds = (probs_ord_np > 0.5).astype(int)

            for k in range(coral_output_dim):
                acc_k = accuracy_score(y_ord_bin_targets[:, k], y_ord_bin_preds[:, k])
                precision_k = precision_score(y_ord_bin_targets[:, k], y_ord_bin_preds[:, k], zero_division=0)
                recall_k = recall_score(y_ord_bin_targets[:, k], y_ord_bin_preds[:, k], zero_division=0)
                f1_k = f1_score(y_ord_bin_targets[:, k], y_ord_bin_preds[:, k], zero_division=0)

                coral_per_task_scores[k]["accuracy"].append(acc_k)
                coral_per_task_scores[k]["precision"].append(precision_k)
                coral_per_task_scores[k]["recall"].append(recall_k)
                coral_per_task_scores[k]["f1"].append(f1_k)

                # AUC
                try:
                    auc_k = roc_auc_score(y_ord_bin_targets[:, k], probs_ord_np[:, k])
                except ValueError:
                    auc_k = np.nan
                coral_auc_per_task[k].append(auc_k)

        # ---- multilabel report ----
        report_dict = classification_report(
            y_multi_test,
            preds_multi_final,
            target_names=multilabel_colnames,
            zero_division=0,
            output_dict=True
        )

        macro_avg = report_dict["macro avg"]
        multilabel_macro_metrics["precision"].append(macro_avg["precision"])
        multilabel_macro_metrics["recall"].append(macro_avg["recall"])
        multilabel_macro_metrics["f1-score"].append(macro_avg["f1-score"])

        weighted_avg = report_dict["weighted avg"]
        multilabel_weighted_metrics["precision"].append(weighted_avg["precision"])
        multilabel_weighted_metrics["recall"].append(weighted_avg["recall"])
        multilabel_weighted_metrics["f1-score"].append(weighted_avg["f1-score"])

        for lab in multilabel_colnames:
            for m in ["precision", "recall", "f1-score"]:
                per_label_scores[lab][m].append(report_dict[lab][m])

        # time logs
        time_logs.append({
            "seed": seed,
            "fold": fold + 1,
            "lgb_train_time_sec": lgb_train_time,
            "multitask_train_time_sec": multitask_train_time,
            "lgb_inference_time_sec": lgb_inference_time,
            "multitask_inference_time_sec": multitask_inference_time
        })

        print("評価フェーズ完了")

# =========================
# Save summary
# =========================
print("統合評価の保存フェーズ開始")

summary_all_path = os.path.join(result_dir, "metrics_all_summary_lgb_multitaskcoralNN_mean_std.csv")
with open(summary_all_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Category", "Averaging", "Metric", "Mean", "Std"])

    # binary
    for metric in ["accuracy", "precision", "recall", "f1", "auc"]:
        mean, std = mean_std(binary_metrics[metric])
        writer.writerow(["Binary(1:Re-registered, 0:No re-registered)", "-", metric, mean, std])

    # ordinal days metrics
    for name in metric_names:
        values = [m[name] for m in all_metrics]
        mean, std = mean_std(values)
        writer.writerow(["Ordinal(days_until_next)", "-", name, mean, std])

    # final argmax threshold metrics
    for k in range(coral_output_dim):
        label = label_names[k]
        for metric in ["accuracy", "precision", "recall", "f1"]:
            mean, std = mean_std(final_bincls_scores[k][metric])
            writer.writerow(["OrdinalBinary(Final-Argmax)", f"y <= '{label}'", metric, mean, std])

    # coral internal tasks
    for k in range(coral_output_dim):
        label = label_names[k]
        for metric in ["accuracy", "precision", "recall", "f1"]:
            mean, std = mean_std(coral_per_task_scores[k][metric])
            writer.writerow(["OrdinalBinary", f"y <= '{label}'", metric, mean, std])

        auc_values = [v for v in coral_auc_per_task[k] if not np.isnan(v)]
        if len(auc_values) > 0:
            mean, std = mean_std(auc_values)
            writer.writerow(["OrdinalBinary", f"y <= '{label}'", "AUC", mean, std])

    # multilabel macro/weighted
    for avg_type, metrics in [("Macro", multilabel_macro_metrics), ("Weighted", multilabel_weighted_metrics)]:
        for metric in ["precision", "recall", "f1-score"]:
            mean, std = mean_std(metrics[metric])
            writer.writerow(["Multilabel", avg_type, metric, mean, std])

    # multilabel per label
    for lab in multilabel_colnames:
        for metric in ["precision", "recall", "f1-score"]:
            mean, std = mean_std(per_label_scores[lab][metric])
            writer.writerow(["Multilabel", lab, metric, mean, std])

print(f"[Saved] {summary_all_path}")

# time summary
time_keys = [
    "lgb_train_time_sec",
    "multitask_train_time_sec",
    "lgb_inference_time_sec",
    "multitask_inference_time_sec",
]
output_time_path = os.path.join(result_dir, "time_summary_mean_std.txt")
with open(output_time_path, "w", encoding="utf-8") as f:
    f.write("=== Training / Inference Time Summary (mean ± std) ===\n")
    f.write("Unit: seconds\n")
    f.write("Std: sample std (ddof=1)\n\n")
    for key in time_keys:
        values = [log[key] for log in time_logs]
        mean, std = mean_std(values)
        f.write(f"{key}: {mean:.6f} ± {std:.6f}\n")
print(f"[Saved] {output_time_path}")

# =========================
# Confusion matrices + scatter
# =========================
# Binary CM (English)
cm_bin_all = confusion_matrix(all_y_bin_true, all_y_bin_pred, labels=[0, 1])
cm_bin_all_df = pd.DataFrame(
    cm_bin_all,
    index=["True: Re-registration observed", "True: No re-registration observed"],
    columns=["Pred: Re-registration observed", "Pred: No re-registration observed"],
)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_bin_all_df, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix (Binary Classification - All Folds)")
plt.tight_layout()
plt.savefig(os.path.join(result_dir, "confusion_matrix_all_folds_lgb_en.png"))
plt.close()

# Binary CM (Japanese)
cm_bin_all_df = pd.DataFrame(
    cm_bin_all,
    index=["正解: 再登記あり", "正解: 再登記なし"],
    columns=["予測: 再登記あり", "予測: 再登記なし"],
)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_bin_all_df, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix (Binary Classification - All Folds)")
plt.tight_layout()
plt.savefig(os.path.join(result_dir, "confusion_matrix_all_folds_lgb_ja.png"))
plt.close()

# Ordinal CM
label_indices = list(range(len(label_names)))
cm_all = confusion_matrix(all_y_true, all_y_pred, labels=label_indices)
cm_all_df = pd.DataFrame(cm_all, index=label_names, columns=label_names)

plt.figure(figsize=(10, 8))
sns.heatmap(cm_all_df, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix (Ordinal(days_until_next) - All Folds)")
plt.tight_layout()
plt.savefig(os.path.join(result_dir, "confusion_matrix_all_folds_lgb_multitaskcoralNN.png"))
plt.close()

# Scatter (days)
true_days_np = np.asarray(all_true_days, dtype=float)
pred_days_np = np.asarray(all_pred_days, dtype=float)

plt.figure(figsize=(8, 6))
plt.scatter(true_days_np, pred_days_np, alpha=0.3, s=20, edgecolor="none")
max_val = max(true_days_np.max(), pred_days_np.max())
plt.plot([0, max_val], [0, max_val], color="red", linestyle="--", linewidth=1.5, label="y=x")
plt.xlabel("True Days Until Next Registration")
plt.ylabel("Predicted Days Until Next Registration")
plt.title("True vs Predicted Days Until Next Registration (All Folds)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(result_dir, "scatter_true_vs_pred_days_until_next.png"))
plt.close()

print("統合評価の保存フェーズ完了")
