import os
import copy
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score
from lightgbm import LGBMClassifier

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt


# =========================================================
# Settings
# =========================================================

# ★ Change paths for your environment
DATA_PATH = r"D:\fujiwara\M\data\after_preprocess\land_data_for_prediction.csv"

RESULT_DIR = r"D:\fujiwara\M\result\dm_simulation\topN_binary_coral_multilabelNN"
os.makedirs(RESULT_DIR, exist_ok=True)

RANDOM_SEED = 0

# H (months) list to evaluate
H_LIST = [1, 4, 9, 24, 120]

# Mapping from H to max category index (for days_until_next_category)
# 0=<1, 1=1-4, 2=4-9, 3=9-24, 4=24-120, 5=>120 or no re-registration
H_TO_MAX_CATEGORY = {
    1: 0,
    4: 1,
    9: 2,
    24: 3,
    120: 4,
}

# Top-N DM candidates per month
TOP_N_LIST = [1000, 2000, 5000, 8000, 10000]

# Business parameters
ALPHA = 0.40  # deal rate
AVG_PRICE = (60_000_000 + 70_000_000) / 2  # average deal price
PI_DEAL = AVG_PRICE * 0.03 + 60_000        # revenue per deal (JPY)


# =========================================================
# Model definitions (CORAL + MultiLabel NN)
# =========================================================

class CoralOrdinalNN(nn.Module):
    def __init__(self, input_dim, num_classes: int):
        """
        num_classes: ordinal class count (e.g., 6 classes -> 0..5)
        """
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
        logits = x_shared @ self.shared_weight  # (batch,)
        logits = logits.unsqueeze(1)            # (batch, 1)
        ordered_bias = torch.cumsum(F.softplus(self.raw_bias), dim=0)
        logits = logits + ordered_bias         # (batch, num_classes-1)
        probs = torch.sigmoid(logits)          # P(y <= k)
        return probs, x_shared


def coral_loss(probs, labels, num_classes: int):
    """
    probs: (batch, num_classes-1)  with P(y <= k)
    labels: (batch,)
    """
    labels = labels.view(-1, 1)
    target = (torch.arange(num_classes - 1).to(labels.device) >= labels).float()
    return F.binary_cross_entropy(probs, target, reduction="mean")


class MultiLabelNN(nn.Module):
    def __init__(self, input_dim, num_labels):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(16, num_labels)

    def forward(self, x):
        features = self.feature_extractor(x)
        probs = torch.sigmoid(self.classifier(features))
        return probs


def plot_confusion_matrix(cm, class_names, title, save_path):
    """
    cm: 2x2 numpy array [[TN, FP],[FN, TP]]
    """
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=[f"Pred: {c}" for c in class_names],
        yticklabels=[f"True: {c}" for c in class_names],
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


# =========================================================
# Load data & sample 5/120 (≈5 months)
# =========================================================

print("Loading data...")
df = pd.read_csv(DATA_PATH)
print(f"Original data shape: {df.shape}")

frac_sample = 5 / 120
df_sampled = df.sample(frac=frac_sample, random_state=RANDOM_SEED).reset_index(drop=True)
print(f"Sampled data shape (≈5 months): {df_sampled.shape}")

# Multi-label columns and sale label index
multilabel_colnames = [
    col for col in df_sampled.columns
    if col.startswith("on_day_reason_group_") and col.endswith("_next")
]
sale_label_candidates = [i for i, col in enumerate(multilabel_colnames) if "sale" in col.lower()]
if len(sale_label_candidates) == 0:
    raise ValueError("Multi-label columns do not contain a 'sale' label.")
sale_label_idx = sale_label_candidates[0]

# Features & targets
X = df_sampled.drop(
    columns=["will_not_be_re_registered", "days_until_next_category", "days_until_next"]
    + multilabel_colnames
).astype(np.float32)

y_binary = df_sampled["will_not_be_re_registered"].values
y_ordinal = df_sampled["days_until_next_category"].values
y_multilabel = df_sampled[multilabel_colnames].values
y_days = df_sampled["days_until_next"].values  # true interval (days)

num_ord_classes = len(np.unique(y_ordinal))


# =========================================================
# Containers for per-fold outputs
# =========================================================

fold_class_probs = []      # list of np.ndarray (n_test, num_ord_classes)
fold_p_sale = []           # list of np.ndarray (n_test,)
fold_days = []             # list of np.ndarray (n_test,)
fold_is_sale_true = []     # list of np.ndarray (n_test,)


# =========================================================
# 5-fold CV (each fold ≒ 1 month)
# =========================================================

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

for fold, (trainval_idx, test_idx) in enumerate(kf.split(X, y_ordinal)):
    print(f"\n========== Fold {fold + 1} ==========")

    X_trainval = X.iloc[trainval_idx]
    y_ordinal_trainval = y_ordinal[trainval_idx]

    train_idx, val_idx = train_test_split(
        trainval_idx,
        test_size=0.1,
        random_state=RANDOM_SEED * 100 + fold,
        stratify=y_ordinal_trainval,
    )

    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    # ------------------------------------
    # Undersampling (train only)
    # ------------------------------------
    train_df = df_sampled.iloc[train_idx].copy()
    counts_0to4 = train_df[train_df["days_until_next_category"].between(0, 4)][
        "days_until_next_category"
    ].value_counts()

    min_cat = counts_0to4.idxmin()
    target_counts = counts_0to4.min()
    target_categories = [c for c in counts_0to4.index if c != min_cat]

    sampled_dfs = []
    for cat in target_categories:
        cat_df = train_df[train_df["days_until_next_category"] == cat]
        sampled_df = cat_df.sample(n=target_counts, random_state=42)
        sampled_dfs.append(sampled_df)

    other_df = train_df[~train_df["days_until_next_category"].isin(target_categories)]
    balanced_train_df = (
        pd.concat(sampled_dfs + [other_df], axis=0).sample(frac=1, random_state=42)
    )

    train_idx = balanced_train_df.index.values

    # ------------------------------------
    # Step 1: Binary classification (re-registration or not)
    # ------------------------------------
    print("Training LightGBM binary classifier...")

    param_grid_lgb = {
        "num_leaves": [31],
        "max_depth": [-1],
        "learning_rate": [0.1],
        "n_estimators": [200],
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
                            random_state=RANDOM_SEED * 100 + fold,
                            n_jobs=-1,
                        )
                        model.fit(X.iloc[train_idx], y_binary[train_idx])
                        val_preds = model.predict(X.iloc[val_idx])
                        score = f1_score(y_binary[val_idx], val_preds, zero_division=0)
                        if score > best_score:
                            best_score = score
                            best_params_lgb = {
                                "num_leaves": num_leaves,
                                "max_depth": max_depth,
                                "learning_rate": lr,
                                "n_estimators": n_est,
                                "class_weight": cw,
                            }

    clf_bin_lgb = LGBMClassifier(
        **best_params_lgb,
        random_state=RANDOM_SEED * 100 + fold,
        n_jobs=-1,
    )
    clf_bin_lgb.fit(X.iloc[train_idx], y_binary[train_idx])
    print("Binary LGBM trained.")

    # ------------------------------------
    # Step 2: CORAL (only for re-registered)
    # ------------------------------------
    print("Training CORAL ordinal NN...")
    start_time = time.time()

    mask_train = y_binary[train_idx] == 0
    X_train_ord = X.iloc[train_idx].loc[mask_train]
    y_ord_train = y_ordinal[train_idx][mask_train]

    mask_val = y_binary[val_idx] == 0
    X_val_ord = X.iloc[val_idx].loc[mask_val]
    y_ord_val = y_ordinal[val_idx][mask_val]

    X_train_t = torch.tensor(X_train_ord.values, dtype=torch.float32)
    y_ord_train_t = torch.tensor(y_ord_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val_ord.values, dtype=torch.float32)
    y_ord_val_t = torch.tensor(y_ord_val, dtype=torch.long)

    train_dataset = TensorDataset(X_train_t, y_ord_train_t)
    train_loader = DataLoader(
        train_dataset,
        batch_size=1024,
        shuffle=True,
        generator=torch.Generator().manual_seed(RANDOM_SEED * 100 + fold),
    )

    coral_model = CoralOrdinalNN(input_dim=X.shape[1], num_classes=num_ord_classes)
    optimizer = torch.optim.Adam(coral_model.parameters(), lr=5e-4)

    num_epochs = 100
    early_stop_patience = 10
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    for epoch in range(num_epochs):
        coral_model.train()
        total_train_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            probs_ord, _ = coral_model(xb)
            loss = coral_loss(probs_ord, yb, num_ord_classes)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        coral_model.eval()
        with torch.no_grad():
            probs_val, _ = coral_model(X_val_t)
            val_loss = coral_loss(probs_val, y_ord_val_t, num_ord_classes)
        avg_val_loss = val_loss.item()

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = copy.deepcopy(coral_model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    coral_model.load_state_dict(best_model_state)
    print(f"CORAL trained in {time.time() - start_time:.1f} sec.")

    # ------------------------------------
    # Step 3: MultiLabel NN (only for re-registered)
    # ------------------------------------
    print("Training MultiLabel NN for causes...")
    start_time = time.time()

    mask_train = y_binary[train_idx] == 0
    X_train_multi = X.iloc[train_idx].loc[mask_train]
    y_multi_train = y_multilabel[train_idx][mask_train]

    mask_val = y_binary[val_idx] == 0
    X_val_multi = X.iloc[val_idx].loc[mask_val]
    y_multi_val = y_multilabel[val_idx][mask_val]

    X_train_mt = torch.tensor(X_train_multi.values, dtype=torch.float32)
    y_multi_train_t = torch.tensor(y_multi_train, dtype=torch.float32)
    X_val_mt = torch.tensor(X_val_multi.values, dtype=torch.float32)
    y_multi_val_t = torch.tensor(y_multi_val, dtype=torch.float32)

    train_dataset_multi = TensorDataset(X_train_mt, y_multi_train_t)
    train_loader_multi = DataLoader(
        train_dataset_multi,
        batch_size=1024,
        shuffle=True,
        generator=torch.Generator().manual_seed(RANDOM_SEED * 100 + fold),
    )

    multilabel_model = MultiLabelNN(input_dim=X.shape[1], num_labels=y_multilabel.shape[1])
    optimizer_multi = torch.optim.Adam(multilabel_model.parameters(), lr=5e-4)
    loss_fn_multi = nn.BCELoss()

    num_epochs = 100
    early_stop_patience = 10
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    for epoch in range(num_epochs):
        multilabel_model.train()
        total_train_loss = 0.0
        for xb, yb in train_loader_multi:
            optimizer_multi.zero_grad()
            probs_multi = multilabel_model(xb)
            loss = loss_fn_multi(probs_multi, yb)
            loss.backward()
            optimizer_multi.step()
            total_train_loss += loss.item()

        multilabel_model.eval()
        with torch.no_grad():
            probs_val = multilabel_model(X_val_mt)
            val_loss = loss_fn_multi(probs_val, y_multi_val_t)
        avg_val_loss = val_loss.item()

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = copy.deepcopy(multilabel_model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"Early stopping (multilabel) at epoch {epoch+1}")
                break

    multilabel_model.load_state_dict(best_model_state)
    print(f"MultiLabel NN trained in {time.time() - start_time:.1f} sec.")

    # ------------------------------------
    # Collect predictions on test set
    # ------------------------------------
    X_test = X.iloc[test_idx]
    y_multi_test = y_multilabel[test_idx]
    y_days_test = y_days[test_idx]

    is_sale_next_true = (y_multi_test[:, sale_label_idx] == 1)

    y_bin_pred = clf_bin_lgb.predict(X_test)
    mask_bin0 = (y_bin_pred == 0)

    class_probs_all = np.zeros((len(X_test), num_ord_classes), dtype=np.float32)
    p_sale_all = np.zeros(len(X_test), dtype=np.float32)

    if mask_bin0.sum() > 0:
        X_test_masked = torch.tensor(X_test[mask_bin0].values, dtype=torch.float32)
        with torch.no_grad():
            probs_ord_masked, _ = coral_model(X_test_masked)
            probs_multi_masked = multilabel_model(X_test_masked)

        probs_ext = torch.cat(
            [
                torch.zeros((probs_ord_masked.shape[0], 1)),
                probs_ord_masked,
                torch.ones((probs_ord_masked.shape[0], 1)),
            ],
            dim=1,
        )
        probs_exact = probs_ext[:, 1:] - probs_ext[:, :-1]  # (n_mask, num_classes)

        class_probs_all[mask_bin0] = probs_exact.cpu().numpy()
        p_sale_all[mask_bin0] = probs_multi_masked[:, sale_label_idx].cpu().numpy()

    fold_class_probs.append(class_probs_all)
    fold_p_sale.append(p_sale_all)
    fold_days.append(y_days_test)
    fold_is_sale_true.append(is_sale_next_true.astype(int))

    print(
        f"Fold {fold+1}: test size={len(X_test)}, "
        f"true sale count={is_sale_next_true.sum()}"
    )


# =========================================================
# Top-N DM simulation (all H, all N)
# =========================================================

n_folds = len(fold_class_probs)

min_test_size = min(arr.shape[0] for arr in fold_class_probs)
usable_N_list = [N for N in TOP_N_LIST if N <= min_test_size]
if len(usable_N_list) == 0:
    raise ValueError(f"All N in TOP_N_LIST are larger than min test size ({min_test_size}).")

print(f"\nUsing N list (per month): {usable_N_list}")

results = []
fold_summary_rows = []  # per (H,N,fold) summary including test_size & positives_in_test

for H in H_LIST:
    print(f"\n===== H = {H} months =====")
    max_cat = H_TO_MAX_CATEGORY[H]
    threshold_days = 31 * H

    for N in usable_N_list:
        fold_revenue_list = []
        fold_TP_list = []
        fold_FP_list = []
        fold_FN_list = []
        fold_TN_list = []
        fold_DM_list = []

        total_TP = 0
        total_FP = 0
        total_DM = 0

        for fold_idx, (class_probs, p_sale, days_fold, is_sale_true_fold) in enumerate(
            zip(fold_class_probs, fold_p_sale, fold_days, fold_is_sale_true)
        ):
            n_test = class_probs.shape[0]
            k = min(N, n_test)

            p_interval_leq_H = class_probs[:, : (max_cat + 1)].sum(axis=1)
            scores = p_interval_leq_H * p_sale

            true_flags = (days_fold <= threshold_days) & (is_sale_true_fold == 1)

            idx_sorted = np.argsort(scores)[::-1]
            top_idx = idx_sorted[:k]

            y_top = true_flags[top_idx]
            tp = int(y_top.sum())
            fp = int(k - tp)

            positives_fold = int(true_flags.sum())
            negatives_fold = int(len(true_flags) - positives_fold)
            fn = positives_fold - tp
            tn = negatives_fold - fp

            deals_fold = tp * ALPHA
            revenue_fold = deals_fold * PI_DEAL

            fold_revenue_list.append(revenue_fold)
            fold_TP_list.append(tp)
            fold_FP_list.append(fp)
            fold_FN_list.append(fn)
            fold_TN_list.append(tn)
            fold_DM_list.append(k)

            total_TP += tp
            total_FP += fp
            total_DM += k

            # per fold summary row
            response_rate_fold = tp / k if k > 0 else 0.0
            fold_summary_rows.append(
                {
                    "H_months_for_score": H,
                    "N_per_month": N,
                    "fold": fold_idx + 1,
                    "test_size": n_test,
                    "positives_in_test": positives_fold,
                    "negatives_in_test": negatives_fold,
                    "DM_sent": k,
                    "TP": tp,
                    "FP": fp,
                    "FN": fn,
                    "TN": tn,
                    "ResponseRate_fold": response_rate_fold,
                }
            )

            # --- fold-wise confusion matrix png ---
            cm = np.array([[tn, fp],
                           [fn, tp]], dtype=int)
            cm_title = f"Confusion (Proposed) H={H}m, N={N}, Fold={fold_idx+1}"
            cm_path = os.path.join(
                RESULT_DIR,
                f"confusion_proposed_H{H}_N{N}_fold{fold_idx+1}.png",
            )
            plot_confusion_matrix(
                cm,
                class_names=["No sale within H", "Sale within H"],
                title=cm_title,
                save_path=cm_path,
            )

        # --- aggregated (5-fold total) confusion matrix png ---
        total_FN = int(np.sum(fold_FN_list))
        total_TN = int(np.sum(fold_TN_list))
        cm_total = np.array([[total_TN, total_FP],
                             [total_FN, total_TP]], dtype=int)
        cm_total_title = f"Confusion (Proposed) H={H}m, N={N}, All folds"
        cm_total_path = os.path.join(
            RESULT_DIR,
            f"confusion_proposed_H{H}_N{N}_allfolds.png",
        )
        plot_confusion_matrix(
            cm_total,
            class_names=["No sale within H", "Sale within H"],
            title=cm_total_title,
            save_path=cm_total_path,
        )

        # --- fold-wise stats ---
        revenue_mean = float(np.mean(fold_revenue_list))
        revenue_std = float(np.std(fold_revenue_list))
        TP_mean = float(np.mean(fold_TP_list))
        TP_std = float(np.std(fold_TP_list))
        FP_mean = float(np.mean(fold_FP_list))
        FP_std = float(np.std(fold_FP_list))

        # --- 5-month total to monthly average ---
        response_rate = total_TP / total_DM if total_DM > 0 else 0.0
        deals_total = ALPHA * total_TP
        revenue_total = deals_total * PI_DEAL
        deals_per_month = deals_total / n_folds
        revenue_per_month = revenue_total / n_folds

        results.append(
            {
                "H_months_for_score": H,
                "N_per_month": N,
                "total_DM_sent": total_DM,
                "TP_total": total_TP,
                "FP_total": total_FP,
                "ResponseRate": response_rate,
                "Deals_per_month": deals_per_month,
                "Revenue_per_month": revenue_per_month,
                "Revenue_mean_fold": revenue_mean,
                "Revenue_std_fold": revenue_std,
                "TP_mean_fold": TP_mean,
                "TP_std_fold": TP_std,
                "FP_mean_fold": FP_mean,
                "FP_std_fold": FP_std,
                "Pi_deal": PI_DEAL,
                "Alpha": ALPHA,
            }
        )

# Save summary CSV
out_path = os.path.join(RESULT_DIR, "dm_topN_simulation_binary_coral_multilabelNN.csv")
results_df = pd.DataFrame(results)
results_df.to_csv(out_path, index=False, encoding="utf-8-sig")

# Save per-fold summary CSV (test_size & positives_in_test etc.)
fold_summary_path = os.path.join(
    RESULT_DIR, "dm_topN_simulation_binary_coral_multilabelNN_folds.csv"
)
fold_summary_df = pd.DataFrame(fold_summary_rows)
fold_summary_df.to_csv(fold_summary_path, index=False, encoding="utf-8-sig")

print(f"\nTop-N DM simulation results saved to: {out_path}")
print(f"Per-fold summary saved to: {fold_summary_path}")
print("Confusion matrix PNGs saved to RESULT_DIR.")
