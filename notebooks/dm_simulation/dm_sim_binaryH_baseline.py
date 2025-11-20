import os
import csv
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold

# ===================== 設定 =====================

# データのパス（既存スクリプトと同じものを使う）
DATA_PATH = r"D:\fujiwara\M\data\after_preprocess\land_data_for_prediction.csv"

# H のリスト（ヶ月）
H_MONTHS_LIST = [1, 4, 9, 24, 120]

# DM混同行列（比較手法：Hヶ月以内/以上の二値分類）格納用
dm_confusion_binaryH = {
    H: {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    for H in H_MONTHS_LIST
}

# DMシミュレーション用パラメータ
DM_FIXED = 10_000              # 月あたり DM送付数（固定）
BASELINE_RESPONSE_RATE = 0.005 # ML導入前の反響率 0.5%
ALPHA = 0.40                   # 成約率 40%

# 平均成約価格 6,000〜7,000万円の中間（6,500万円）× 3% + 6万円
AVG_PRICE = (60_000_000 + 70_000_000) / 2
PI_DEAL = AVG_PRICE * 0.03 + 60_000    # 1成約あたりの収益（円）

# LightGBM の固定ハイパーパラメータ（必要なら調整可）
LGB_PARAMS = dict(
    num_leaves=31,
    max_depth=-1,
    learning_rate=0.1,
    n_estimators=200,
    class_weight="balanced",
    n_jobs=-1
)

# 結果保存ディレクトリ（プロジェクト直下に result/dm_simulation/ を想定）
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)          # notebooks/
RESULT_DIR = os.path.join(PROJECT_ROOT, "..", "result", "dm_simulation")
os.makedirs(RESULT_DIR, exist_ok=True)

# =================================================


def main():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    print(f"Data shape: {df.shape}")

    # --- multi-label (next reason) の列名と売買インデックス ---
    multilabel_colnames = [
        col for col in df.columns
        if col.startswith("on_day_reason_group_") and col.endswith("_next")
    ]

    sale_label_candidates = [
        i for i, col in enumerate(multilabel_colnames)
        if "sale" in col.lower()
    ]
    if len(sale_label_candidates) == 0:
        raise ValueError("Multi-label columns do not contain a 'sale' label. Please check column names.")
    sale_label_idx = sale_label_candidates[0]
    print(f"Detected sale label column: {multilabel_colnames[sale_label_idx]} (index={sale_label_idx})")

    # --- 特徴量行列 X：既存スクリプトと同じルールで構成 ---
    X = df.drop(
        columns=['will_not_be_re_registered', 'days_until_next_category', 'days_until_next']
        + multilabel_colnames
    ).astype(np.float32)

    # --- 順序ラベル（fold分割用） ---
    y_ordinal = df['days_until_next_category'].values

    # --- multi-label 真値 ---
    y_multilabel = df[multilabel_colnames].values

    # --- 売買フラグ & 日数真値 ---
    days_until_next = df["days_until_next"].values
    is_sale_next_true = (y_multilabel[:, sale_label_idx] == 1)

    # --- seeds × 5fold で StratifiedKFold（既存スクリプトに合わせる） ---
    seeds = list(range(3))

    for seed in seeds:
        print(f"========== Seed {seed + 1} ==========")
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

        for fold, (train_idx, test_idx) in enumerate(kf.split(X, y_ordinal)):
            print(f"\n========== Seed {seed + 1} Fold {fold + 1} ==========")
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]

            # H ごとに「Hヶ月以内に売買が起きるかどうか」の二値分類を実施
            for H in H_MONTHS_LIST:
                threshold_days = 31 * H
                # 全データに対して y_{H,sale} を作成
                y_H = ((days_until_next <= threshold_days) & is_sale_next_true).astype(int)

                y_train_H = y_H[train_idx]
                y_test_H = y_H[test_idx]

                # LightGBM で二値分類（比較手法）
                clf_H = LGBMClassifier(
                    **LGB_PARAMS,
                    random_state=seed * 100 + fold
                )
                clf_H.fit(X_train, y_train_H)

                # DM送付 = 予測1、非送付 = 予測0 とみなす
                y_pred_H = clf_H.predict(X_test)

                tp = np.sum((y_pred_H == 1) & (y_test_H == 1))
                fp = np.sum((y_pred_H == 1) & (y_test_H == 0))
                fn = np.sum((y_pred_H == 0) & (y_test_H == 1))
                tn = np.sum((y_pred_H == 0) & (y_test_H == 0))

                dm_confusion_binaryH[H]["TP"] += int(tp)
                dm_confusion_binaryH[H]["FP"] += int(fp)
                dm_confusion_binaryH[H]["FN"] += int(fn)
                dm_confusion_binaryH[H]["TN"] += int(tn)

    # ================= DMシミュレーション結果の保存 =================
    out_path = os.path.join(RESULT_DIR, "dm_simulation_sale_within_Hmonths_binaryH.csv")
    print(f"\nSaving DM simulation (binary H-baseline) to: {out_path}")

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "H_months",
            "TP", "FP", "FN", "TN",
            "ML_response_rate",
            "Baseline_response_rate",
            "Deals_baseline",
            "Deals_ML",
            "Delta_deals",
            "Pi_deal",
            "Delta_revenue"
        ])

        for H in H_MONTHS_LIST:
            cnt = dm_confusion_binaryH[H]
            TP = cnt["TP"]
            FP = cnt["FP"]
            FN = cnt["FN"]
            TN = cnt["TN"]

            dm_sent = TP + FP
            if dm_sent > 0:
                ml_response_rate = TP / dm_sent
            else:
                ml_response_rate = 0.0

            baseline_response_rate = BASELINE_RESPONSE_RATE

            # 成約数 = DM送付数 × 反響率 × 成約率
            deals_baseline = DM_FIXED * baseline_response_rate * ALPHA
            deals_ml = DM_FIXED * ml_response_rate * ALPHA
            delta_deals = deals_ml - deals_baseline

            # 収益への影響 = 成約数の増分 × 1成約あたりの収益
            delta_revenue = delta_deals * PI_DEAL

            writer.writerow([
                H,
                TP, FP, FN, TN,
                ml_response_rate,
                baseline_response_rate,
                deals_baseline,
                deals_ml,
                delta_deals,
                PI_DEAL,
                delta_revenue
            ])

    print("Done.")


if __name__ == "__main__":
    main()
