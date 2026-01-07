import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt


# =========================================================
# Settings
# =========================================================

# ★ Change paths for your environment (Windows)
DATA_PATH = r"D:\fujiwara\M\data\after_preprocess\land_data_for_prediction.csv"

RESULT_DIR = r"D:\fujiwara\M\result\dm_simulation\topN_binaryH"
os.makedirs(RESULT_DIR, exist_ok=True)

BASE_SEED = 0

# ★ Keep this: test data size should be about "1 month" (your premise)
SAMPLE_FRAC = 5 / 120  # keep

# Repeat 5-fold CV multiple times by DIFFERENT SAMPLING per run
N_RUNS = 3
N_SPLITS = 5

H_LIST = [1, 4, 9, 24, 120]
TOP_N_LIST = [1000, 2000, 5000, 8000, 10000]

# Business parameters (used for Revenue_fold summary)
ALPHA = 0.40
AVG_PRICE = (60_000_000 + 70_000_000) / 2
PI_DEAL = AVG_PRICE * 0.03 + 60_000


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
print(f"Original data shape: {df_full.shape}")

# Identify multilabel columns, and sale label index (only for defining evaluation positives)
multilabel_colnames = [
    col for col in df_full.columns
    if col.startswith("on_day_reason_group_") and col.endswith("_next")
]
sale_label_candidates = [
    i for i, col in enumerate(multilabel_colnames) if "sale" in col.lower()
]
if len(sale_label_candidates) == 0:
    raise ValueError("Multi-label columns do not contain a 'sale' label.")
sale_label_idx = sale_label_candidates[0]


# =========================================================
# Storage across ALL runs & folds (for fold-independent summary)
# fold_scores_by_H[H] -> list of np.ndarray (n_test,)
# fold_trueflags_by_H[H] -> list of np.ndarray (n_test,)
# fold_meta_by_H[H] -> list of dict(run, fold, test_size, sample_seed)
# =========================================================

fold_scores_by_H = {H: [] for H in H_LIST}
fold_trueflags_by_H = {H: [] for H in H_LIST}
fold_meta_by_H = {H: [] for H in H_LIST}


# =========================================================
# 5-fold CV × 3 runs (sampling differs by run)
# =========================================================

for run_id in range(N_RUNS):
    sample_seed = BASE_SEED + run_id
    print("\n==============================")
    print(f"Run {run_id+1}/{N_RUNS} : sample_frac={SAMPLE_FRAC}, sample_seed={sample_seed}")
    print("==============================")

    # ★ Run-specific sampling
    df = df_full.sample(frac=SAMPLE_FRAC, random_state=sample_seed).reset_index(drop=True)
    print(f"Sampled data shape: {df.shape}")

    # Build X & y on sampled data
    X = df.drop(
        columns=[
            "will_not_be_re_registered",
            "days_until_next_category",
            "days_until_next",
        ] + multilabel_colnames
    ).astype(np.float32)

    y_ordinal = df["days_until_next_category"].values
    y_days = df["days_until_next"].values
    y_multilabel = df[multilabel_colnames].values
    is_sale_next_true_all = (y_multilabel[:, sale_label_idx] == 1).astype(int)

    # Run-specific 5-fold
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=sample_seed)

    for fold, (trainval_idx, test_idx) in enumerate(kf.split(X, y_ordinal)):
        print(f"\n========== Run {run_id+1} | Fold {fold + 1} ==========")

        X_trainval = X.iloc[trainval_idx]
        y_ordinal_trainval = y_ordinal[trainval_idx]

        train_idx, val_idx = train_test_split(
            trainval_idx,
            test_size=0.1,
            random_state=sample_seed * 100 + fold,
            stratify=y_ordinal_trainval,
        )

        print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

        X_test = X.iloc[test_idx]
        y_days_test = y_days[test_idx]
        is_sale_next_true_test = is_sale_next_true_all[test_idx]

        # For each H, train binary model and store test scores/labels
        for H in H_LIST:
            print(f"  [Run {run_id+1} Fold {fold+1}] Training binary LGBM for H={H} months...")

            threshold_days = 31 * H

            # Target: any registration within H months (cause not used) - same as original script
            y_H_all = (y_days <= threshold_days).astype(int)
            y_H_train = y_H_all[train_idx]
            y_H_val = y_H_all[val_idx]

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
                                    random_state=sample_seed * 1000 + fold * 10 + H,
                                    n_jobs=-1,
                                )
                                model.fit(X.iloc[train_idx], y_H_train)
                                val_preds = model.predict(X.iloc[val_idx])
                                score = f1_score(y_H_val, val_preds, zero_division=0)
                                if score > best_score:
                                    best_score = score
                                    best_params_lgb = {
                                        "num_leaves": num_leaves,
                                        "max_depth": max_depth,
                                        "learning_rate": lr,
                                        "n_estimators": n_est,
                                        "class_weight": cw,
                                    }

            clf_bin_H = LGBMClassifier(
                **best_params_lgb,
                random_state=sample_seed * 1000 + fold * 10 + H,
                n_jobs=-1,
            )
            clf_bin_H.fit(X.iloc[train_idx], y_H_train)
            print(f"    -> best F1 on val = {best_score:.4f}")

            proba_test = clf_bin_H.predict_proba(X_test)[:, 1]

            # True label for evaluation: sale within H months (same as original)
            y_H_test_true = np.logical_and(
                (y_days_test <= threshold_days),
                (is_sale_next_true_test == 1),
            ).astype(int)

            fold_scores_by_H[H].append(proba_test)
            fold_trueflags_by_H[H].append(y_H_test_true)
            fold_meta_by_H[H].append(
                {
                    "run": run_id + 1,
                    "fold": fold + 1,
                    "test_size": len(test_idx),
                    "sample_frac": SAMPLE_FRAC,
                    "sample_seed": sample_seed,
                }
            )

            print(
                f"    Test size={len(test_idx)}, "
                f"Positives in test (H<= & sale)={y_H_test_true.sum()}"
            )


# =========================================================
# Top-N DM simulation (all H, all N) over ALL folds (runs×folds)
# =========================================================

# Find min test size across ALL folds & H (to ensure N is feasible everywhere)
min_test_size = min(
    meta["test_size"] for H in H_LIST for meta in fold_meta_by_H[H]
)
usable_N_list = [N for N in TOP_N_LIST if N <= min_test_size]
if len(usable_N_list) == 0:
    raise ValueError(
        f"All N in TOP_N_LIST are larger than min test size ({min_test_size})."
    )

print(f"\nMin test size across all (run×fold): {min_test_size}")
print(f"Using N list: {usable_N_list}")

# Outputs
fold_rows = []     # per (H, N, run, fold)
summary_rows = []  # per (H, N): mean/std across all folds (15)

for H in H_LIST:
    print(f"\n===== H = {H} months =====")

    scores_folds = fold_scores_by_H[H]
    true_folds = fold_trueflags_by_H[H]
    meta_folds = fold_meta_by_H[H]

    if not (len(scores_folds) == len(true_folds) == len(meta_folds)):
        raise RuntimeError("Internal length mismatch among fold containers.")

    n_total_folds = len(scores_folds)  # should be N_RUNS*N_SPLITS (=15)
    print(f"Total folds for H={H}: {n_total_folds}")

    for N in usable_N_list:
        rr_list = []
        revenue_list = []
        tp_list = []
        fp_list = []
        fn_list = []
        tn_list = []

        for scores, true_flags, meta in zip(scores_folds, true_folds, meta_folds):
            n_test = len(scores)
            k = min(N, n_test)

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
                    "run": meta["run"],
                    "fold": meta["fold"],
                    "test_size": n_test,
                    "positives_in_test": positives,
                    "negatives_in_test": negatives,
                    "DM_sent": k,
                    "TP": tp,
                    "FP": fp,
                    "FN": fn,
                    "TN": tn,
                    "ResponseRate_fold": response_rate_fold,
                    "Revenue_fold": revenue_fold,
                    "Pi_deal": PI_DEAL,
                    "Alpha": ALPHA,
                    "sample_frac": meta["sample_frac"],
                    "sample_seed": meta["sample_seed"],
                }
            )

        # aggregated confusion matrix (all folds)
        total_TP = int(np.sum(tp_list))
        total_FP = int(np.sum(fp_list))
        total_FN = int(np.sum(fn_list))
        total_TN = int(np.sum(tn_list))

        cm_total = np.array([[total_TN, total_FP],
                             [total_FN, total_TP]], dtype=int)
        cm_total_title = f"Confusion (Binary H<=) H={H}m, N={N}, All folds (runs×folds)"
        cm_total_path = os.path.join(
            RESULT_DIR,
            f"confusion_binaryH_H{H}_N{N}_allfolds.png",
        )
        plot_confusion_matrix(
            cm_total,
            class_names=["No sale within H", "Sale within H"],
            title=cm_total_title,
            save_path=cm_total_path,
        )

        # fold stats (mean/std across all folds)
        rr_mean = float(np.mean(rr_list))
        rr_std = float(np.std(rr_list, ddof=1)) if len(rr_list) >= 2 else 0.0

        revenue_mean = float(np.mean(revenue_list))
        revenue_std = float(np.std(revenue_list, ddof=1)) if len(revenue_list) >= 2 else 0.0

        summary_rows.append(
            {
                "H_months_for_score": H,
                "N_per_fold": N,
                "n_folds_total": len(rr_list),
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
summary_path = os.path.join(RESULT_DIR, "dm_topN_simulation_binaryH_5foldx3runs_summary.csv")
fold_path = os.path.join(RESULT_DIR, "dm_topN_simulation_binaryH_5foldx3runs_folds.csv")

pd.DataFrame(summary_rows).to_csv(summary_path, index=False, encoding="utf-8-sig")
pd.DataFrame(fold_rows).to_csv(fold_path, index=False, encoding="utf-8-sig")

print(f"\nSummary saved to: {summary_path}")
print(f"Per-fold details saved to: {fold_path}")
print("Confusion matrix PNGs saved to RESULT_DIR.")
