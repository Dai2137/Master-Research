# ライブラリ読み込み
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns
from sklearn.model_selection import KFold
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

# AUC計算用
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
result_dir = r"D:\fujiwara\M\result\olg"
os.makedirs(result_dir, exist_ok=True)

# カテゴリ名辞書（順序カテゴリ）
category_labels = {
    1: '～1カ月', 2: '1～3カ月', 3: '3～5カ月',
    4: '5～7カ月', 5: '7～10カ月', 6: '10～18カ月',
    7: '18カ月～', 8: '再登記なし'
}

# データ読み込み
land_data_for_prediction = pd.read_csv(r"D:\fujiwara\M\data\after_preprocess\land_data_for_prediction.csv")

kf = KFold(n_splits=5, shuffle=True, random_state=42)
X = land_data_for_prediction.drop(columns=['binary_category', 'sales1_category'])
y_ord = land_data_for_prediction['sales1_category']

acc_list, mae_list, mse_list, rmse_list, spearman_list = [], [], [], [], []
fold_metrics = []
per_class_metrics = {"precision": [], "recall": [], "f1": []}
auc_per_fold = {f"P(y ≤ {k})": [] for k in range(1, 8)}

for fold, (train_index, test_index) in enumerate(kf.split(X)):
    print(f"Fold {fold + 1}")
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_ord_train, y_ord_test = y_ord.iloc[train_index], y_ord.iloc[test_index]

    # 学習フェーズ
    ord_model = OrderedModel(y_ord_train, X_train, distr='logit')
    print("順序回帰モデルの学習を開始...")
    start_time = time.time()
    ord_result = ord_model.fit(method='bfgs', disp=True)
    end_time = time.time()
    print(f"順序回帰モデルの学習完了（経過時間: {end_time - start_time:.2f} 秒）")

    # 推論フェーズ
    y_ord_probs = ord_result.predict(X_test)
    y_ord_pred = y_ord_probs.idxmax(axis=1).astype(int) + 1

    # 評価フェーズ
    acc, mae, mse, rmse = compute_metrics(y_ord_test.to_numpy(), y_ord_pred)
    spearman, _ = spearmanr(y_ord_test.to_numpy(), y_ord_pred)
    spearman = float(spearman)
    acc_list.append(acc)
    mae_list.append(mae)
    mse_list.append(mse)
    rmse_list.append(rmse)
    spearman_list.append(spearman)
    fold_metrics.append([fold + 1, acc, mae, mse, rmse, spearman])
    print(f"Fold {fold + 1} Accuracy: {acc:.4f}, MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, Spearman: {spearman:.4f}")

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_ord_test, y_ord_pred,
        labels=sorted(category_labels.keys()),
        zero_division=0
    )
    per_class_metrics["precision"].append(precision)
    per_class_metrics["recall"].append(recall)
    per_class_metrics["f1"].append(f1)
    print(f"Fold {fold + 1} クラスごとの指標: {per_class_metrics}")

    compute_cumulative_auc_and_store(y_ord_test.reset_index(drop=True), y_ord_probs, auc_per_fold)
    print(f"Fold {fold + 1} AUC: {auc_per_fold}")

    y_all = pd.Series(np.concatenate([y_ord_test.to_numpy(), y_ord_pred]))
    unique_labels = sorted(np.unique(y_all))
    cm = confusion_matrix(y_ord_test, y_ord_pred, labels=unique_labels)
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
    heatmap_path = os.path.join(result_dir, f"confusion_matrix_fold{fold + 1}_olg.png")
    plt.savefig(heatmap_path)
    plt.close()

# --- 評価指標をCSVに保存 ---

# --- 各foldの評価指標をCSVに保存 ---
with open(os.path.join(result_dir, "each_fold_metrics_olg.csv"), mode='w', newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)
    writer.writerow(["Fold", "Accuracy", "MAE", "MSE", "RMSE", "Spearman"])
    for row in fold_metrics:
        writer.writerow([row[0]] + [f"{v:.4f}" for v in row[1:]])

# --- 各foldのクラスごとの指標をCSVに保存 ---
with open(os.path.join(result_dir, "each_fold_per_class_metrics_olg.csv"), mode='w', newline='', encoding='utf-8-sig') as f:
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
with open(os.path.join(result_dir, "each_fold_auc_olg.csv"), mode='w', newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)
    
    # すべてのAUC値（リストのリスト）を取得（各行: 各P(y≤k)のfold別AUC）
    auc_keys = list(auc_per_fold.keys())  # ["P(y ≤ 1)", ..., "P(y ≤ 7)"]
    auc_rows = [auc_per_fold[k] for k in auc_keys]

    # ヘッダー：["AUC種別", "Fold 1", "Fold 2", ..., "Fold n"]
    header = ["AUC種別"] + [f"Fold {i+1}" for i in range(len(auc_rows[0]))]
    writer.writerow(header)

    # 各AUC種別について、foldごとの値を書き出す
    for key, row in zip(auc_keys, auc_rows):
        writer.writerow([key] + [f"{v:.4f}" for v in row])


metrics = ["Accuracy", "MAE", "MSE", "RMSE", "Spearman"]
values = [acc_list, mae_list, mse_list, rmse_list, spearman_list]

# --- K-foldのAccuracyなどの結果をCSVに保存 ---
with open(os.path.join(result_dir, "kfold_metrics_olg.csv"), mode='w', newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)
    writer.writerow(["指標", "平均値", "95%CI下限", "95%CI上限"])
    print("K-foldの結果(Accuracyなど)")
    for name, scores in zip(metrics, values):
        mean, lower, upper = mean_ci(scores)
        writer.writerow([name, f"{mean:.4f}", f"{lower:.4f}", f"{upper:.4f}"])
        print(f"{name}: Mean={mean:.4f}, 95% CI=({lower:.4f}, {upper:.4f})")

# --- K-foldのクラスごとの指標をCSVに保存 ---
labels = sorted(category_labels.keys())
with open(os.path.join(result_dir, "kfold_per_class_metrics_olg.csv"), mode='w', newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)
    writer.writerow(["Class", "Metric", "Mean", "95%CI下限", "95%CI上限"])
    print("K-foldの結果(クラスごとの Precision / Recall / F1)")

    for metric in ["precision", "recall", "f1"]:
        all_folds = np.array(per_class_metrics[metric])  # shape: (n_folds, n_classes)
        for i, label in enumerate(labels):
            values = all_folds[:, i]
            mean, lower, upper = mean_ci(values)
            writer.writerow([
                category_labels.get(label, label),
                metric,
                f"{mean:.4f}",
                f"{lower:.4f}",
                f"{upper:.4f}"
            ])
            print(f"{category_labels[label]} [{metric}]: Mean={mean:.4f}, 95% CI=({lower:.4f}, {upper:.4f})")

# --- K-foldのクラスごとのAUCをCSVに保存 ---
auc_summary = summarize_auc_per_class(auc_per_fold)
print("K-foldの結果(AUC)")
with open(os.path.join(result_dir, "kfold_auc_olg.csv"), mode='w', newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)
    writer.writerow(["閾値", "平均 AUC", "95%CI 下限", "95%CI 上限"])
    writer.writerows(auc_summary)
    for row in auc_summary:
        print(row)
    