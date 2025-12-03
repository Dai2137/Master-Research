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

# ★ Change paths for your environment
DATA_PATH = r"D:\fujiwara\M\data\after_preprocess\land_data_for_prediction.csv"

RESULT_DIR = r"D:\fujiwara\M\result\dm_simulation\topN_binaryH"
os.makedirs(RESULT_DIR, exist_ok=True)

RANDOM_SEED = 0

H_LIST = [1, 4, 9, 24, 120]
TOP_N_LIST = [1000, 2000, 5000, 8000, 10000]

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
# Load data & sample
# =========================================================

print("Loading data...")
df = pd.read_csv(DATA_PATH)
print(f"Original data shape: {df.shape}")

frac_sample = 5 / 120
df_sampled = df.sample(frac=frac_sample, random_state=RANDOM_SEED).reset_index(drop=True)
print(f"Sampled data shape (≈5 months): {df_sampled.shape}")

multilabel_colnames = [
    col for col in df_sampled.columns
    if col.startswith("on_day_reason_group_") and col.endswith("_next")
]
sale_label_candidates = [
    i for i, col in enumerate(multilabel_colnames) if "sale" in col.lower()
]
if len(sale_label_candidates) == 0:
    raise ValueError("Multi-label columns do not contain a 'sale' label.")
sale_label_idx = sale_label_candidates[0]

X = df_sampled.drop(
    columns=[
        "will_not_be_re_registered",
        "days_until_next_category",
        "days_until_next",
    ] + multilabel_colnames
).astype(np.float32)

y_ordinal = df_sampled["days_until_next_category"].values
y_multilabel = df_sampled[multilabel_colnames].values
y_days = df_sampled["days_until_next"].values

is_sale_next_true_all = (y_multilabel[:, sale_label_idx] == 1).astype(int)


# =========================================================
# Per-H containers
# =========================================================

fold_scores_by_H = {H: [] for H in H_LIST}
fold_trueflags_by_H = {H: [] for H in H_LIST}


# =========================================================
# 5-fold CV
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

    X_test = X.iloc[test_idx]
    y_days_test = y_days[test_idx]
    is_sale_next_true_test = is_sale_next_true_all[test_idx]

    for H in H_LIST:
        print(f"  [Fold {fold+1}] Training binary LGBM for H={H} months...")

        threshold_days = 31 * H

        # Target: any registration within H months (cause not used)
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
                                random_state=RANDOM_SEED * 1000 + fold * 10 + H,
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
            random_state=RANDOM_SEED * 1000 + fold * 10 + H,
            n_jobs=-1,
        )
        clf_bin_H.fit(X.iloc[train_idx], y_H_train)
        print(f"    -> best F1 on val = {best_score:.4f}")

        proba_test = clf_bin_H.predict_proba(X_test)[:, 1]

        # True label for evaluation: H-month sale
        y_H_test_true = np.logical_and(
            (y_days_test <= threshold_days),
            (is_sale_next_true_test == 1),
        ).astype(int)

        fold_scores_by_H[H].append(proba_test)
        fold_trueflags_by_H[H].append(y_H_test_true)

        print(
            f"    Test size={len(test_idx)}, "
            f"Positives in test (H<= & sale)={y_H_test_true.sum()}"
        )


# =========================================================
# Top-N DM simulation (all H, all N)
# =========================================================

n_folds = 5

min_test_size = min(arr.shape[0] for arr in fold_scores_by_H[H_LIST[0]])
usable_N_list = [N for N in TOP_N_LIST if N <= min_test_size]
if len(usable_N_list) == 0:
    raise ValueError(
        f"All N in TOP_N_LIST are larger than min test size ({min_test_size})."
    )

print(f"\nUsing N list (per month): {usable_N_list}")

results = []
fold_summary_rows = []  # per (H,N,fold) summary including test_size & positives_in_test

for H in H_LIST:
    print(f"\n===== H = {H} months =====")

    scores_folds = fold_scores_by_H[H]
    true_folds = fold_trueflags_by_H[H]

    for N in usable_N_list:
        total_TP = 0
        total_FP = 0
        total_DM = 0

        fold_revenue_list = []
        fold_TP_list = []
        fold_FP_list = []
        fold_FN_list = []
        fold_TN_list = []
        fold_DM_list = []

        for fold_idx, (scores, true_flags) in enumerate(zip(scores_folds, true_folds)):
            n_test = len(scores)
            k = min(N, n_test)

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
            cm_title = f"Confusion (Binary H<=) H={H}m, N={N}, Fold={fold_idx+1}"
            cm_path = os.path.join(
                RESULT_DIR,
                f"confusion_binaryH_H{H}_N{N}_fold{fold_idx+1}.png",
            )
            plot_confusion_matrix(
                cm,
                class_names=["No sale within H", "Sale within H"],
                title=cm_title,
                save_path=cm_path,
            )

        # --- aggregated confusion over folds ---
        total_FN = int(np.sum(fold_FN_list))
        total_TN = int(np.sum(fold_TN_list))
        cm_total = np.array([[total_TN, total_FP],
                             [total_FN, total_TP]], dtype=int)
        cm_total_title = f"Confusion (Binary H<=) H={H}m, N={N}, All folds"
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

        response_rate = total_TP / total_DM if total_DM > 0 else 0.0

        deals_total = ALPHA * total_TP
        revenue_total = deals_total * PI_DEAL

        deals_per_month = deals_total / n_folds
        revenue_per_month = revenue_total / n_folds

        revenue_mean = float(np.mean(fold_revenue_list))
        revenue_std = float(np.std(fold_revenue_list))
        TP_mean = float(np.mean(fold_TP_list))
        TP_std = float(np.std(fold_TP_list))
        FP_mean = float(np.mean(fold_FP_list))
        FP_std = float(np.std(fold_FP_list))

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

out_path = os.path.join(RESULT_DIR, "dm_topN_simulation_binaryH_allH.csv")
results_df = pd.DataFrame(results)
results_df.to_csv(out_path, index=False, encoding="utf-8-sig")

fold_summary_path = os.path.join(
    RESULT_DIR, "dm_topN_simulation_binaryH_allH_folds.csv"
)
fold_summary_df = pd.DataFrame(fold_summary_rows)
fold_summary_df.to_csv(fold_summary_path, index=False, encoding="utf-8-sig")

print(f"\nTop-N DM simulation (binary H<=) results saved to: {out_path}")
print(f"Per-fold summary saved to: {fold_summary_path}")
print("Confusion matrix PNGs saved to RESULT_DIR.")
