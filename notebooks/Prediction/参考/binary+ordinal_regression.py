# ライブラリ読み込み
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score

# from collections import Counter
from statsmodels.miscmodels.ordinal_model import OrderedModel
import numpy as np
from scipy.stats import t, spearmanr
import time
import os
import csv

# 評価指標の計算用関数
def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return acc, mae, mse, rmse

# 信頼区間の計算関数（95%）
def mean_ci(data, confidence=0.95):
    n = len(data)
    m = np.mean(data)
    se = np.std(data, ddof=1) / np.sqrt(n)
    h = se * t.ppf((1 + confidence) / 2., n-1)
    return m, m - h, m + h

def compute_cumulative_auc_and_store(y_true, y_cum_probs, auc_store_dict):
    for i, k in enumerate(range(1, 8)):
        y_binary = (y_true <= k).astype(int)
        y_prob = y_cum_probs.iloc[:, i]
        try:
            auc = roc_auc_score(y_binary, y_prob)
        except ValueError:
            auc = np.nan
        auc_store_dict[f"P(y ≤ {k})"].append(auc)

def summarize_auc_per_class(auc_dict):
    summary_auc = []
    for k in range(1, 8):
        scores = auc_dict[f"P(y ≤ {k})"]
        mean, lower, upper = mean_ci(scores)
        summary_auc.append([f"P(y ≤ {k})", f"{mean:.4f}", f"{lower:.4f}", f"{upper:.4f}"])
    return summary_auc

# 保存パスの指定と準備
result_dir = r"D:\fujiwara\M\result\binary+ordinal"
os.makedirs(result_dir, exist_ok=True)

# カテゴリ名辞書（順序カテゴリ）
category_labels = {
    1: '～1 month', 2: '1～4 month', 3: '4～9 month',
    4: '9～24 month', 5: '24 month～', 6: 'no re-registration'
}

# データ読み込み
land_data_for_prediction = pd.read_csv(r"D:\fujiwara\M\data\after_preprocess\land_data_for_prediction.csv")

kf = KFold(n_splits=5, shuffle=True, random_state=42)
X = land_data_for_prediction.drop(columns=['binary_category', 'sales1_category'])
y_bin = land_data_for_prediction['binary_category']
y_ord = land_data_for_prediction['sales1_category']

# スコア格納用
acc_list, mae_list, mse_list, rmse_list, spearman_list = [], [], [], [], []
fold_metrics = []

# クラスごとの指標の蓄積用
per_class_metrics = {
    "precision": [],
    "recall": [],
    "f1": []
}

auc_per_fold = {f"P(y ≤ {k})": [] for k in range(1, 8)}

for fold, (train_index, test_index) in enumerate(kf.split(X)):
    print(f"Fold {fold + 1}")
    # データ分割
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_bin_train, y_bin_test = y_bin.iloc[train_index], y_bin.iloc[test_index]
    y_ord_train, y_ord_test = y_ord.iloc[train_index], y_ord.iloc[test_index]
    
    # 学習フェーズ
    # 二値分類モデル構築
    # ロジスティック回帰モデルの定義・学習
    bin_model = LogisticRegression(random_state=42)
    bin_model.fit(X_train, y_bin_train)
    print("二値分類モデルの学習完了")

    # 順序回帰モデル構築
    # y_bin_trainが1の行のidxを取得
    bin_1_idx = y_bin_train[y_bin_train == 1].index
    # X_train, y_ord_trainの内，bin_1_idxに対応する行のみを抽出
    X_train_bin_1 = X_train.loc[bin_1_idx]
    y_ord_train_bin_1 = y_ord_train.loc[bin_1_idx]

    ord_model = OrderedModel(y_ord_train_bin_1, X_train_bin_1, distr='logit')
    # 学習に15分かかる
    print("順序回帰モデルの学習を開始...")
    start_time = time.time()

    ord_result = ord_model.fit(method='bfgs', disp=True)

    end_time = time.time()
    print(f"順序回帰モデルの学習完了（経過時間: {end_time - start_time:.2f} 秒）")
    

    # 推論フェーズ
    # 二値分類モデルの予測
    y_bin_pred = bin_model.predict(X_test)
    # 二値分類モデルの予測結果が1の行のidxを取得
    bin_1_idx_test = np.where(y_bin_pred == 1)[0]
    X_test_bin_1 = X_test.iloc[bin_1_idx_test]
    # y_ord_test_bin_1 = y_ord_test.iloc[bin_1_idx_test]
    # 順序回帰モデルの予測
    y_ord_probs = ord_result.predict(X_test_bin_1)
    y_ord_pred = y_ord_probs.idxmax(axis=1).astype(int) + 1
    y_ord_final_pred = np.full_like(y_bin_pred, fill_value=6, dtype=int)  # 初期値は6（再登記なし扱い）
    y_ord_final_pred[bin_1_idx_test] = y_ord_pred.values  # 販売ありの位置に予測値を代入

    # 評価フェーズ
    acc, mae, mse, rmse = compute_metrics(y_ord_test.to_numpy(), y_ord_final_pred)
    spearman, _ = spearmanr(y_ord_test.to_numpy(), y_ord_final_pred)
    spearman = float(spearman)
    acc_list.append(acc)
    mae_list.append(mae)
    mse_list.append(mse)
    rmse_list.append(rmse)
    spearman_list.append(spearman)
    fold_metrics.append([fold + 1, acc, mae, mse, rmse, spearman])

    print(f"Fold {fold + 1} Accuracy: {acc:.4f}, MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, Spearman: {spearman:.4f}")

    # クラスごとの指標の計算
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_ord_test, y_ord_final_pred,
        labels=sorted(category_labels.keys()),
        zero_division=0
    )
    per_class_metrics["precision"].append(precision)
    per_class_metrics["recall"].append(recall)
    per_class_metrics["f1"].append(f1)

    print(f"Fold {fold + 1} クラスごとの指標:")
    for idx, label in enumerate(sorted(category_labels.keys())):
        print(f"  {category_labels[label]}: Precision={precision[idx]:.3f}, Recall={recall[idx]:.3f}, F1={f1[idx]:.3f}")
    
    # 各クラス以下である二値分類に対するAUCの計算
    y_ord_test_bin_1 = y_ord_test.iloc[bin_1_idx_test].reset_index(drop=True)
    compute_cumulative_auc_and_store(
        y_true=y_ord_test_bin_1,
        y_cum_probs=y_ord_probs,
        auc_store_dict=auc_per_fold
    )
    print(f"Fold {fold + 1} AUC: {auc_per_fold}")
    
    # --- 混同行列の出力と保存 ---
    y_all = pd.Series(np.concatenate([y_ord_test.to_numpy(), y_ord_final_pred]))
    unique_labels = sorted(np.unique(y_all))

    cm = confusion_matrix(y_ord_test, y_ord_final_pred, labels=unique_labels)
    cm_df = pd.DataFrame(
        cm,
        index=[f"実際:{category_labels.get(i, i)}" for i in unique_labels],
        columns=[f"予測:{category_labels.get(i, i)}" for i in unique_labels]
    )

    print("\n==== 混同行列 ====")
    print(cm_df)

    plt.figure(figsize=(7, 5))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Oranges')
    plt.title(f"Fold {fold + 1} 混同行列")
    plt.xlabel("予測カテゴリ")
    plt.ylabel("実際のカテゴリ")
    plt.tight_layout()
    heatmap_path = os.path.join(result_dir, f"confusion_matrix_fold{fold + 1}.png")
    plt.savefig(heatmap_path)
    plt.close()


# --- 評価指標をCSVに保存 ---

# --- 各foldの評価指標をCSVに保存 ---
with open(os.path.join(result_dir, "each_fold_metrics_binary+ordinal.csv"), mode='w', newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)
    writer.writerow(["Fold", "Accuracy", "MAE", "MSE", "RMSE", "Spearman"])
    for row in fold_metrics:
        writer.writerow([row[0]] + [f"{v:.4f}" for v in row[1:]])

# --- 各foldのクラスごとの指標をCSVに保存 ---
with open(os.path.join(result_dir, "each_fold_per_class_metrics_binary+ordinal.csv"), mode='w', newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)
    writer.writerow(["Fold", "Class", "Precision", "Recall", "F1"])
    for fold in range(len(per_class_metrics["precision"])):
        num_classes = len(per_class_metrics["precision"][fold])
        for class_idx in range(num_classes):
            row = [
                fold + 1,
                class_idx + 1,
                f"{per_class_metrics['precision'][fold][class_idx]:.4f}",
                f"{per_class_metrics['recall'][fold][class_idx]:.4f}",
                f"{per_class_metrics['f1'][fold][class_idx]:.4f}"
            ]
            writer.writerow(row)


# --- 各foldのクラスごとのAUCをCSVに保存 ---
with open(os.path.join(result_dir, "each_fold_auc_binary+ordinal.csv"), mode='w', newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)
    
    # キーと fold 数の取得
    auc_keys = list(auc_per_fold.keys())  # ["P(y ≤ 1)", ..., "P(y ≤ 7)"]
    num_folds = len(auc_per_fold[auc_keys[0]])

    # ヘッダー：["AUC種別", "Fold 1", ..., "Fold N"]
    header = ["AUC種別"] + [f"Fold {i+1}" for i in range(num_folds)]
    writer.writerow(header)

    # 各AUC種別ごとに行として出力
    for key in auc_keys:
        row = [f"{v:.4f}" for v in auc_per_fold[key]]
        writer.writerow([key] + row)


metrics = ["Accuracy", "MAE", "MSE", "RMSE", "Spearman"]
values = [acc_list, mae_list, mse_list, rmse_list, spearman_list]

# --- K-foldのAccuracyなどの結果をCSVに保存 ---
with open(os.path.join(result_dir, "kfold_metrics_binary+ordinal.csv"), mode='w', newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)
    writer.writerow(["指標", "平均値", "95%CI下限", "95%CI上限"])
    print("K-foldの結果(Accuracyなど)")
    for name, scores in zip(metrics, values):
        mean, lower, upper = mean_ci(scores)
        writer.writerow([name, f"{mean:.4f}", f"{lower:.4f}", f"{upper:.4f}"])
        print(f"{name}: Mean={mean:.4f}, 95% CI=({lower:.4f}, {upper:.4f})")

# --- K-foldのクラスごとの指標をCSVに保存 (precision, recall, f1)---
# 各クラスごとにfold間平均＋95%CIを保存
with open(os.path.join(result_dir, "kfold_per_class_metrics_binary+ordinal.csv"), mode='w', newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)
    writer.writerow(["Class", "Metric", "Mean", "95% CI Lower", "95% CI Upper"])

    labels = sorted(category_labels.keys())
    for metric in ["precision", "recall", "f1"]:
        all_folds = np.array(per_class_metrics[metric])  # shape: (n_folds, n_classes)
        for i, label in enumerate(labels):
            values = all_folds[:, i]
            mean, lower, upper = mean_ci(values)
            writer.writerow([
                category_labels.get(label, label),
                metric,
                f"{mean:.4f}", f"{lower:.4f}", f"{upper:.4f}"
            ])
            print(f"{category_labels[label]} [{metric}]: Mean={mean:.4f}, 95% CI=({lower:.4f}, {upper:.4f})")


# --- K-foldのクラスごとのAUCをCSVに保存 ---
auc_summary = summarize_auc_per_class(auc_per_fold)
with open(os.path.join(result_dir, "kfold_auc_binary+ordinal.csv"), mode='w', newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)
    writer.writerow(["閾値", "平均 AUC", "95%CI 下限", "95%CI 上限"])
    writer.writerows(auc_summary)
    