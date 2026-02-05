import os
import copy
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt
from pathlib import Path

# =========================================================
# Settings
# =========================================================

# Windows paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[5]  
# ↑ src/Prediction/binary+ordinal_multilabel/single/binary+ordinal(+multilabel)/lgb+coral+multilabelNN
#   から M_refactored/ まで戻る
DATA_PATH = PROJECT_ROOT / "data" / "after_preprocess" / "land_data_for_prediction.csv"

# 保存パスの指定と準備
RESULT_DIR = (
    PROJECT_ROOT
    / "results"
    / "dm_simulation"
    / "topN_binary_coral_multilabelNN"
)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


BASE_SEED = 0

# ★ ここが重要：テスト（評価）対象を 5/120 にする（添付コード仕様）
SAMPLE_FRAC = 5 / 120  # keep this

# Repeat: sampling differs by run_id
N_RUNS = 3
N_SPLITS = 5

H_LIST = [1, 4, 9, 24, 120]
H_TO_MAX_CATEGORY = {1: 0, 4: 1, 9: 2, 24: 3, 120: 4}

TOP_N_LIST = [1000, 2000, 5000, 8000, 10000]

ALPHA = 0.40
AVG_PRICE = (60_000_000 + 70_000_000) / 2
PI_DEAL = AVG_PRICE * 0.03 + 60_000


# =========================================================
# Model definitions (CORAL + MultiLabel NN)
# =========================================================

class CoralOrdinalNN(nn.Module):
    def __init__(self, input_dim, num_classes: int):
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
        probs = torch.sigmoid(logits)  # P(y <= k)
        return probs, x_shared


def coral_loss(probs, labels, num_classes: int):
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
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


# =========================================================
# Load full data
# =========================================================

print("Loading data...")
df_full = pd.read_csv(DATA_PATH)
print(f"Full data shape: {df_full.shape}")

multilabel_colnames = [
    c for c in df_full.columns
    if c.startswith("on_day_reason_group_") and c.endswith("_next")
]
sale_label_candidates = [i for i, c in enumerate(multilabel_colnames) if "sale" in c.lower()]
if len(sale_label_candidates) == 0:
    raise ValueError("Multi-label columns do not contain a 'sale' label.")
sale_label_idx = sale_label_candidates[0]


# =========================================================
# 5-fold CV × 3 runs (sampling differs by run)
# =========================================================

all_fold_class_probs = []
all_fold_p_sale = []
all_fold_days = []
all_fold_is_sale_true = []
all_fold_run = []
all_fold_fold = []
all_fold_test_size = []

quantitative_cols = [
    'month_sin', 'same_day_count', 'size', 'official_price',
    'population_density', 'building_coverage_ratio',
    'floor_area_ratio', 'on_foot'
]


for run_id in range(N_RUNS):
    # ★ run ごとにサンプリングが変わる（ここが差）
    run_seed = BASE_SEED + run_id
    print(f"\n==============================")
    print(f" Run {run_id+1}/{N_RUNS} : sampling frac={SAMPLE_FRAC}, seed={run_seed}")
    print(f"==============================")

    df = df_full.sample(frac=SAMPLE_FRAC, random_state=run_seed).reset_index(drop=True)
    print(f"Run-sampled data shape: {df.shape}")

    # Features & targets (run-sampled)
    X = df.drop(
        columns=["will_not_be_re_registered", "days_until_next_category", "days_until_next"]
        + multilabel_colnames
    ).astype(np.float32)

    y_binary = df["will_not_be_re_registered"].values
    y_ordinal = df["days_until_next_category"].values
    y_multilabel = df[multilabel_colnames].values
    y_days = df["days_until_next"].values

    num_ord_classes = len(np.unique(y_ordinal))
    print(f"num_ord_classes (run {run_id+1}): {num_ord_classes}")

    # fold split inside this run-sampled dataset
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=run_seed)

    for fold, (trainval_idx, test_idx) in enumerate(kf.split(X, y_ordinal)):
        print(f"\n========== Run {run_id+1} | Fold {fold+1} ==========")

        y_ordinal_trainval = y_ordinal[trainval_idx]

        train_idx, val_idx = train_test_split(
            trainval_idx,
            test_size=0.1,
            random_state=run_seed * 100 + fold,
            stratify=y_ordinal_trainval,
        )
        print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

        # ===== StandardScaler (fit on train only) =====
        scaler = StandardScaler()

        # train で fit
        X.loc[train_idx, quantitative_cols] = scaler.fit_transform(
            X.loc[train_idx, quantitative_cols]
        )

        # val / test は transform のみ
        X.loc[val_idx, quantitative_cols] = scaler.transform(
            X.loc[val_idx, quantitative_cols]
        )
        X.loc[test_idx, quantitative_cols] = scaler.transform(
            X.loc[test_idx, quantitative_cols]
        )

        # ------------------------------------
        # Undersampling (train only)
        # ------------------------------------
        train_df = df.iloc[train_idx].copy()

        counts_0to4 = train_df[train_df["days_until_next_category"].between(0, 4)][
            "days_until_next_category"
        ].value_counts()

        if len(counts_0to4) == 0:
            raise ValueError("No categories in 0..4 found in training fold; check labels.")

        min_cat = counts_0to4.idxmin()
        target_counts = counts_0to4.min()
        target_categories = [c for c in counts_0to4.index if c != min_cat]

        sampled_dfs = []
        for cat in target_categories:
            cat_df = train_df[train_df["days_until_next_category"] == cat]
            if len(cat_df) >= target_counts and target_counts > 0:
                sampled_dfs.append(cat_df.sample(n=target_counts, random_state=42))

        other_df = train_df[~train_df["days_until_next_category"].isin(target_categories)]
        balanced_train_df = pd.concat(sampled_dfs + [other_df], axis=0).sample(frac=1, random_state=42)
        train_idx_balanced = balanced_train_df.index.values

        # ------------------------------------
        # Step 1: Binary (LightGBM)
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
                                random_state=run_seed * 100 + fold,
                                n_jobs=-1,
                            )
                            model.fit(X.iloc[train_idx_balanced], y_binary[train_idx_balanced])
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
            random_state=run_seed * 100 + fold,
            n_jobs=-1,
        )
        clf_bin_lgb.fit(X.iloc[train_idx_balanced], y_binary[train_idx_balanced])
        print("Binary LGBM trained.")

        # ------------------------------------
        # Step 2: CORAL (only for y_binary==0)
        # ------------------------------------
        print("Training CORAL ordinal NN...")
        start_time = time.time()

        mask_train = y_binary[train_idx_balanced] == 0
        X_train_ord = X.iloc[train_idx_balanced].loc[mask_train]
        y_ord_train = y_ordinal[train_idx_balanced][mask_train]

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
            generator=torch.Generator().manual_seed(run_seed * 100 + fold),
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
            for xb, yb in train_loader:
                optimizer.zero_grad()
                probs_ord, _ = coral_model(xb)
                loss = coral_loss(probs_ord, yb, num_ord_classes)
                loss.backward()
                optimizer.step()

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
        # Step 3: MultiLabel NN
        # ------------------------------------
        print("Training MultiLabel NN...")
        start_time = time.time()

        mask_train = y_binary[train_idx_balanced] == 0
        X_train_multi = X.iloc[train_idx_balanced].loc[mask_train]
        y_multi_train = y_multilabel[train_idx_balanced][mask_train]

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
            generator=torch.Generator().manual_seed(run_seed * 100 + fold),
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
            for xb, yb in train_loader_multi:
                optimizer_multi.zero_grad()
                probs_multi = multilabel_model(xb)
                loss = loss_fn_multi(probs_multi, yb)
                loss.backward()
                optimizer_multi.step()

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
        # Collect predictions on test set (fold-independent)
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
            probs_exact = probs_ext[:, 1:] - probs_ext[:, :-1]
            class_probs_all[mask_bin0] = probs_exact.cpu().numpy()
            p_sale_all[mask_bin0] = probs_multi_masked[:, sale_label_idx].cpu().numpy()

        all_fold_class_probs.append(class_probs_all)
        all_fold_p_sale.append(p_sale_all)
        all_fold_days.append(y_days_test)
        all_fold_is_sale_true.append(is_sale_next_true.astype(int))
        all_fold_run.append(run_id)
        all_fold_fold.append(fold)
        all_fold_test_size.append(len(X_test))

        print(
            f"Run {run_id+1} Fold {fold+1}: test size={len(X_test)}, true sale count={is_sale_next_true.sum()}"
        )


# =========================================================
# Top-N DM simulation (H,N) over all folds (15 folds)
# =========================================================

n_total_folds = len(all_fold_class_probs)
print(f"\nTotal folds: {n_total_folds} (= {N_RUNS} runs × {N_SPLITS} folds)")

min_test_size = min(all_fold_test_size)
usable_N_list = [N for N in TOP_N_LIST if N <= min_test_size]
if len(usable_N_list) == 0:
    raise ValueError(f"All N in TOP_N_LIST are larger than min test size ({min_test_size}).")
print(f"Min test size across all folds: {min_test_size}")
print(f"Using N list (per fold): {usable_N_list}")

fold_rows = []
summary_rows = []

for H in H_LIST:
    max_cat = H_TO_MAX_CATEGORY[H]
    threshold_days = 31 * H

    for N in usable_N_list:
        rr_list = []
        revenue_list = []
        tp_list = []
        fp_list = []
        fn_list = []
        tn_list = []

        for idx in range(n_total_folds):
            class_probs = all_fold_class_probs[idx]
            p_sale = all_fold_p_sale[idx]
            days_fold = all_fold_days[idx]
            is_sale_true_fold = all_fold_is_sale_true[idx]
            run_id = all_fold_run[idx]
            fold_id = all_fold_fold[idx]

            n_test = class_probs.shape[0]
            k = min(N, n_test)

            # score
            p_interval_leq_H = class_probs[:, : (max_cat + 1)].sum(axis=1)
            scores = p_interval_leq_H * p_sale

            # ground truth: sale within H months
            true_flags = (days_fold <= threshold_days) & (is_sale_true_fold == 1)

            idx_sorted = np.argsort(scores)[::-1]
            top_idx = idx_sorted[:k]

            y_top = true_flags[top_idx]
            tp = int(y_top.sum())
            fp = int(k - tp)

            positives = int(true_flags.sum())
            negatives = int(len(true_flags) - positives)
            fn = positives - tp
            tn = negatives - fp

            response_rate_fold = tp / k if k > 0 else 0.0

            deals_fold = tp * ALPHA
            revenue_fold = deals_fold * PI_DEAL

            rr_list.append(response_rate_fold)
            revenue_list.append(revenue_fold)
            tp_list.append(tp)
            fp_list.append(fp)
            fn_list.append(fn)
            tn_list.append(tn)

            fold_rows.append(
                {
                    "H_months_for_score": H,
                    "N_per_fold": N,
                    "run": run_id + 1,
                    "fold": fold_id + 1,
                    "test_size": n_test,
                    "positives_in_test": positives,
                    "DM_sent": k,
                    "TP": tp,
                    "FP": fp,
                    "FN": fn,
                    "TN": tn,
                    "ResponseRate_fold": response_rate_fold,
                    "Revenue_fold": revenue_fold,
                    "Pi_deal": PI_DEAL,
                    "Alpha": ALPHA,
                    "sample_frac": SAMPLE_FRAC,
                    "sample_seed": BASE_SEED + run_id,
                }
            )

        # totals for confusion matrix (all folds)
        total_TP = int(np.sum(tp_list))
        total_FP = int(np.sum(fp_list))
        total_FN = int(np.sum(fn_list))
        total_TN = int(np.sum(tn_list))

        cm_total = np.array([[total_TN, total_FP],
                             [total_FN, total_TP]], dtype=int)
        cm_total_title = f"Confusion (Proposed) H={H}m, N={N}, All folds (runs×folds)"
        cm_total_path = os.path.join(
            RESULT_DIR, f"confusion_proposed_H{H}_N{N}_allfolds.png"
        )
        plot_confusion_matrix(
            cm_total,
            class_names=["No sale within H", "Sale within H"],
            title=cm_total_title,
            save_path=cm_total_path,
        )

        # fold stats (mean/std across 15 folds)
        rr_mean = float(np.mean(rr_list))
        rr_std = float(np.std(rr_list, ddof=1)) if len(rr_list) >= 2 else 0.0

        revenue_mean = float(np.mean(revenue_list))
        revenue_std = float(np.std(revenue_list, ddof=1)) if len(revenue_list) >= 2 else 0.0

        summary_rows.append(
            {
                "H_months_for_score": H,
                "N_per_fold": N,
                "n_folds_total": n_total_folds,
                "ResponseRate_mean_fold": rr_mean,
                "ResponseRate_std_fold": rr_std,
                "Revenue_mean_fold": revenue_mean,
                "Revenue_std_fold": revenue_std,
                "TP_total": total_TP,
                "FP_total": total_FP,
                "FN_total": total_FN,
                "TN_total": total_TN,
                "Pi_deal": PI_DEAL,
                "Alpha": ALPHA,
                "sample_frac": SAMPLE_FRAC,
                "n_runs": N_RUNS,
                "n_splits": N_SPLITS,
            }
        )

# Save CSVs
summary_path = os.path.join(RESULT_DIR, "dm_topN_simulation_5foldx3runs_summary.csv")
fold_path = os.path.join(RESULT_DIR, "dm_topN_simulation_5foldx3runs_folds.csv")

pd.DataFrame(summary_rows).to_csv(summary_path, index=False, encoding="utf-8-sig")
pd.DataFrame(fold_rows).to_csv(fold_path, index=False, encoding="utf-8-sig")

print(f"\nSummary saved to: {summary_path}")
print(f"Per-fold details saved to: {fold_path}")
print("Confusion matrix PNGs saved to RESULT_DIR.")
