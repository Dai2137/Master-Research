import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
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
# ---------- アンダーサンプリングを追加 ----------
from sklearn.utils import resample

# 保存パスの指定と準備（変更するの忘れないように！！）
result_dir = r"D:\fujiwara\M\result\ordinal_multilabel\single\multilabel\multilabelNN\5_undersample"
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




# ---------- Multi-label Classifier ----------
class MultiLabelNN(nn.Module):
    def __init__(self, input_dim, num_labels):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.classifier = nn.Linear(16, num_labels)

    def forward(self, x):
        features = self.feature_extractor(x)
        probs = torch.sigmoid(self.classifier(features))
        return probs



# ---------- データ読み込み ----------
print("データ読み込みを開始...")
df = pd.read_csv(r"D:\fujiwara\M\data\after_preprocess\land_data_for_prediction.csv")
print("データ読み込み完了")


# ラベル名の取得（マルチラベル分類用）
multilabel_colnames = [col for col in df.columns if col.startswith("on_day_reason_group_") and col.endswith("_next")]


X = df.drop(columns=['will_not_be_re_registered', 'days_until_next_category'] +
            [col for col in df.columns if col.startswith("on_day_reason_group_") and col.endswith("_next")]).values

y_ordinal = df['days_until_next_category'].values  
y_multilabel = df[[col for col in df.columns if col.startswith("on_day_reason_group_") and col.endswith("_next")]].values






# 評価指標格納用
all_metrics = []
metric_names = ["Accuracy", "MAE", "MSE", "RMSE", "Spearman", "QWK"]

# ループ外で定義（foldループの前）
all_y_true = []
all_y_pred = []




# Fold間集計用（マルチラベル macro avg）
multilabel_macro_metrics = {
    "precision": [],
    "recall": [],
    "f1-score": []
}


# weighted avg の評価指標格納用
multilabel_weighted_metrics = {
    "precision": [],
    "recall": [],
    "f1-score": []
}






# ラベルごとのスコア格納用（全fold分,マルチラベル分類）
per_label_scores = {label: {"precision": [], "recall": [], "f1-score": []} for label in multilabel_colnames}


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

        print("Before undersampling (train only, categories 0-5):")
        print(train_df['days_until_next_category'].value_counts().sort_index())

        # 0〜5 の全カテゴリの件数を確認
        counts_all = train_df['days_until_next_category'].value_counts().sort_index()

        # 最小件数を基準にする
        target_counts = counts_all.min()

        sampled_dfs = []
        for cat in counts_all.index:   # 0〜5 全部対象
            cat_df = train_df[train_df['days_until_next_category'] == cat]
            sampled_df = cat_df.sample(n=target_counts, random_state=42)
            sampled_dfs.append(sampled_df)

        # 結合してシャッフル
        balanced_train_df = pd.concat(sampled_dfs, axis=0).sample(frac=1, random_state=42)

        # df 全体に対するインデックスに変換
        train_idx = balanced_train_df.index.values

        print("After undersampling (train only, categories 0-5):")
        print(balanced_train_df['days_until_next_category'].value_counts().sort_index())

        # ========== マルチラベルNNの学習 ==========
        print("マルチラベルNNの学習を開始...")
        start_time = time.time()

        X_train = X[train_idx]
        y_multi_train = y_multilabel[train_idx]

        X_val = X[val_idx]
        y_multi_val = y_multilabel[val_idx]


        # torch tensor に変換
        X_train_mt = torch.tensor(X_train, dtype=torch.float32)
        y_multi_train = torch.tensor(y_multi_train, dtype=torch.float32)

        X_val_mt = torch.tensor(X_val, dtype=torch.float32)
        y_multi_val = torch.tensor(y_multi_val, dtype=torch.float32)

        # DataLoader
        train_dataset = TensorDataset(X_train_mt, y_multi_train)
        train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, generator=torch.Generator().manual_seed(seed * 100 + fold))

        # 損失記録用リスト
        train_losses = []
        val_losses = []


        model = MultiLabelNN(input_dim=X.shape[1], num_labels=y_multilabel.shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
        loss_fn_multi = nn.BCELoss()


        num_epochs = 100

        # Early Stopping
        early_stop_patience = 10
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        

        for epoch in trange(num_epochs, desc="Training Epochs"):
            model.train()
            total_train_loss = 0
            for xb, y_multi in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
                optimizer.zero_grad()
                probs_multi = model(xb)
                loss = loss_fn_multi(probs_multi, y_multi)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
            
            avg_train_loss = total_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # --- Validation損失の計算 ---
            model.eval()
            with torch.no_grad():
                probs_multi_val = model(X_val_mt)
                val_loss = loss_fn_multi(probs_multi_val, y_multi_val)
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
        print(f"マルチラベルNNの学習完了（経過時間: {end_time - start_time:.2f} 秒）")
        print("学習フェーズ完了")
        print("--------------------------------")


        # ========== Step 2: 推論 ==========
        print("--------------------------------")
        print("推論フェーズ開始")

        # Step 1: テストデータを取得
        X_test = X[test_idx]
        y_multi_test = y_multilabel[test_idx]

        # Step 2: 推論
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

        with torch.no_grad():
            probs_multi_all = model(X_test_tensor)
            preds_multi_all = (probs_multi_all > 0.5).int().numpy()

        preds_multi_final = preds_multi_all


        print("推論フェーズ完了")
        print("--------------------------------")


        # ---------- Step 3: 評価 ----------
        print("--------------------------------")
        print("評価フェーズ開始")

        # --------- Step 3-1: マルチラベル分類の評価 ----------
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

        # weighted avg のスコアも記録
        weighted_avg = report_dict["weighted avg"]
        multilabel_weighted_metrics["precision"].append(weighted_avg["precision"])
        multilabel_weighted_metrics["recall"].append(weighted_avg["recall"])
        multilabel_weighted_metrics["f1-score"].append(weighted_avg["f1-score"])

        for label in multilabel_colnames:
            for metric in ["precision", "recall", "f1-score"]:
                per_label_scores[label][metric].append(report_dict[label][metric])


        # 表示のみ（保存はしない）
        print("[マルチラベル分類 Classification Report]:")
        print(pd.DataFrame(report_dict).T)

        print("評価フェーズ完了")

        print("--------------------------------")



print("--------------------------------")
print("統合評価の保存フェーズ開始")
# ========= 統合評価CSVの出力（完全版） =========
summary_all_path = os.path.join(result_dir, "metrics_all_summary_multilabelNN_5_undersample.csv")

with open(summary_all_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Category", "Averaging", "Metric", "Mean", "95% CI Lower", "95% CI Upper"])


    # 1. マルチラベル分類（Macro & Weighted）
    for avg_type, metrics in [("Macro", multilabel_macro_metrics),
                               ("Weighted", multilabel_weighted_metrics)]:
        for metric in ["precision", "recall", "f1-score"]:
            values = metrics[metric]
            mean, ci_lower, ci_upper = mean_ci(values)
            writer.writerow(["Multilabel", avg_type, metric, mean, ci_lower, ci_upper])

    # 2. マルチラベル分類（ラベルごと）
    for label in multilabel_colnames:
        for metric in ["precision", "recall", "f1-score"]:
            values = per_label_scores[label][metric]
            mean, ci_lower, ci_upper = mean_ci(values)
            writer.writerow(["Multilabel", label, metric, mean, ci_lower, ci_upper])

print("統合評価の保存フェーズ完了")
print("--------------------------------")
