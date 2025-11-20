import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_absolute_error, mean_squared_error, cohen_kappa_score, confusion_matrix, classification_report
from scipy.stats import spearmanr, t, pearsonr
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
import shap


# 保存パスの指定と準備（変更するの忘れないように！！）
result_dir = r"D:\fujiwara\M\result\binary+ordinal_multilabel\single\binary+ordinal(+multilabel)\lgb+coral+multilabelNN"
os.makedirs(result_dir, exist_ok=True)



# ---------- Utility Functions ----------
def compute_ordered_metrics(y_true, y_pred):
    # acc = accuracy_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    corr, _ = pearsonr(y_true, y_pred)
    # spearman, _ = spearmanr(y_true, y_pred)
    # qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    
    return {
        # "Accuracy": acc,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "Corr": corr,
        # "Spearman": spearman,
        # "QWK": qwk
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


# def predict_expected_class(model, X_df):
#     """
#     CORALモデルの出力から予測されるクラス期待値（soft class）を返す関数
#     SHAPによる可視化のために使用

#     Parameters:
#         model : torch.nn.Module
#             学習済みのCORALモデル（foldごとに異なる）
#         X_df : pd.DataFrame
#             特徴量名付きDataFrame（.iloc[] で分割後のX_testなど）

#     Returns:
#         np.ndarray : shape (N, 1), Expected class per sample
#     """
#     model.eval()
#     x_tensor = torch.tensor(X_df.values, dtype=torch.float32)
#     with torch.no_grad():
#         probs, _ = model(x_tensor)  # Output: (N, K−1)
#         probs_ext = torch.cat([
#             torch.zeros((probs.shape[0], 1)),
#             probs,
#             torch.ones((probs.shape[0], 1))
#         ], dim=1)
#         probs_exact = probs_ext[:, 1:] - probs_ext[:, :-1]  # shape: (N, K)
#         expected_class = (probs_exact * torch.arange(0, probs_exact.shape[1])).sum(dim=1)
#     return expected_class.cpu().numpy().reshape(-1, 1)

def predict_midpoint(model, X_df):
    """
    CORALモデルの出力から「各クラスの範囲中央値ベース」の予測値を返す関数。
    SHAP可視化に用いる。

    Parameters
    ----------
    model : torch.nn.Module
        学習済みのCORALモデル（foldごとに異なる）
    X_df : pd.DataFrame
        特徴量名付きDataFrame（.iloc[]で分割後のX_testなど）

    Returns
    -------
    np.ndarray : shape (N, 1)
        各サンプルの予測中央値（月単位）
    """
    model.eval()
    x_tensor = torch.tensor(X_df.values, dtype=torch.float32)
    label_to_midpoint = torch.tensor([31 * 0.5, 31 * 2.5, 31 * 6.5, 31 * 16.5, 31 * 72.0], dtype=torch.float32)  # 各クラスの中央値

    with torch.no_grad():
        # CORAL出力: shape (N, K−1)
        probs, _ = model(x_tensor)

        # 累積確率に 0, 1 を追加して差分を取ることでクラス確率を得る (N, K)
        probs_ext = torch.cat([
            torch.zeros((probs.shape[0], 1)),
            probs,
            torch.ones((probs.shape[0], 1))
        ], dim=1)
        probs_exact = probs_ext[:, 1:] - probs_ext[:, :-1]

        # 各クラス確率 × 範囲中央値の加重平均 → 予測中央値（月単位）
        y_pred_mid = (probs_exact * label_to_midpoint).sum(dim=1)

    return y_pred_mid.cpu().numpy().reshape(-1, 1)


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

# ★追加：売買ラベルの列インデックス（"sale" を含む列名を想定）
sale_label_candidates = [i for i, col in enumerate(multilabel_colnames) if "sale" in col.lower()]
if len(sale_label_candidates) == 0:
    raise ValueError("Multi-label columns do not contain a 'sale' label. Please check column names.")
sale_label_idx = sale_label_candidates[0]



X = df.drop(columns=['will_not_be_re_registered', 'days_until_next_category', 'days_until_next'] +
            [col for col in df.columns if col.startswith("on_day_reason_group_") and col.endswith("_next")]).astype(np.float32)

y_binary = df['will_not_be_re_registered'].values
y_ordinal = df['days_until_next_category'].values
y_multilabel = df[[col for col in df.columns if col.startswith("on_day_reason_group_") and col.endswith("_next")]].values

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


# DMシミュレーション用設定
H_MONTHS_LIST = [1, 4, 9, 24, 120]  # しきい値H（ヶ月）
H_TO_MAX_CATEGORY = {  # days_until_next_category のクラスとの対応
    1: 0,    # <1 month
    4: 1,    # 1-4 months
    9: 2,    # 4-9 months
    24: 3,   # 9-24 months
    120: 4   # 24-120 months
}

# 全fold統合DM混同行列カウンタ
dm_confusion = {
    H: {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    for H in H_MONTHS_LIST
}

# ビジネスシミュレーション用パラメータ
DM_FIXED = 10_000                    # 月あたり DM送付数（固定）
BASELINE_RESPONSE_RATE = 0.005       # ML導入前の反響率 0.5%
ALPHA = 0.40                         # 成約率 40%

# 平均成約価格 6,000〜7,000万円の中間（6,500万円）× 3% + 6万円
AVG_PRICE = (60_000_000 + 70_000_000) / 2
PI_DEAL = AVG_PRICE * 0.03 + 60_000   # 1成約あたりの収益（円）


# 評価指標格納用
all_metrics = []
# metric_names = ["Accuracy", "MAE", "MSE", "RMSE", "Spearman", "QWK"]
metric_names = ["MAE", "MSE", "RMSE", "Corr"]

all_y_bin_true = []
all_y_bin_pred = []

# ループ外で定義（foldループの前）
all_y_true = []
all_y_pred = []

# 追加
all_true_days = []   # 実測「日数」を全foldで蓄積
all_pred_days = []   # 予測「日数」を全foldで蓄積



# CORALの二値分類の評価指標格納用
coral_per_task_scores = {  # foldごとに保持
    k: {"accuracy": [], "precision": [], "recall": [], "f1": []}
    for k in range(coral_output_dim)
}


# CORALの二値分類のAUC格納用
coral_auc_per_task = {k: [] for k in range(coral_output_dim)}


# 最終予測カテゴリに対する「y ≤ k」二値化の評価（AUCなし）
final_bincls_scores = {
    k: {"accuracy": [], "precision": [], "recall": [], "f1": []}
    for k in range(coral_output_dim)  # 最後は全て1になるので除外
}


label_names = [
    '-1 months',
    '1-4 months',
    '4-9 months', 
    '9-24 months', 
    '24-120 months',
    '120- months'
]


# ラベルと対応する範囲の中央値を定義
label_to_midpoint = {
    0: 31 * 0.5,     # <1 month
    1: 31 * 2.5,     # 1–4 months
    2: 31 * 6.5,     # 4–9 months
    3: 31 * 16.5,    # 9–24 months
    4: 31 * 72.0,      # 24–120 months
    5: 31 * 120.0      # >120 months（再登記なし）
}


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

        # 全カテゴリの件数確認（アンダーサンプリング前）
        print("Before undersampling (train only, all categories):")
        print(train_df['days_until_next_category'].value_counts().sort_index())

        # 0〜4のカテゴリに絞って件数確認
        counts_0to4 = train_df[train_df['days_until_next_category'].between(0, 4)]['days_until_next_category'].value_counts()

        # 最小件数のカテゴリを特定
        min_cat = counts_0to4.idxmin()
        target_counts = counts_0to4.min()

        # アンダーサンプリング対象カテゴリ（0〜4のうち最小以外）
        target_categories = [c for c in counts_0to4.index if c != min_cat]

        sampled_dfs = []
        for cat in target_categories:
            cat_df = train_df[train_df['days_until_next_category'] == cat]
            sampled_df = cat_df.sample(n=target_counts, random_state=42)
            sampled_dfs.append(sampled_df)

        # 最小カテゴリと 5 以上はそのまま保持
        other_df = train_df[~train_df['days_until_next_category'].isin(target_categories)]
        balanced_train_df = pd.concat(sampled_dfs + [other_df], axis=0).sample(frac=1, random_state=42)

        # df 全体に対するインデックスに変換
        train_idx = balanced_train_df.index.values

        # 全カテゴリの件数確認（アンダーサンプリング後）
        print("After undersampling (train only, all categories):")
        print(balanced_train_df['days_until_next_category'].value_counts().sort_index())



        

        # ========== Step 1-1: 二値分類器の学習 ========== 
        print("二値分類モデル（LightGBM）の学習を開始...")

        param_grid_lgb = {
            "num_leaves": [15, 31, 63],
            "max_depth": [-1, 5, 10],
            "learning_rate": [0.05, 0.1, 0.2],
            "n_estimators": [100, 200, 300],
            "class_weight": ["balanced"]
        }

        best_score = -np.inf
        best_params_lgb = None

        # グリッドサーチ（valで評価）
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
                            score = f1_score(y_binary[val_idx], val_preds)
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

        # ベストモデルを train_idx で再学習
        clf_bin_lgb = LGBMClassifier(**best_params_lgb, random_state=seed * 100 + fold, n_jobs=-1)
        clf_bin_lgb.fit(X.iloc[train_idx], y_binary[train_idx])
        print("二値分類モデルの学習完了")

        # ========== Step 1-2: CORALの学習 ==========
        print("CORALの学習を開始...")
        start_time = time.time()

        # 再登記されるものだけ使う（binary=0）
        mask_train = y_binary[train_idx] == 0
        X_train = X.iloc[train_idx].loc[mask_train]
        y_ord_train = y_ordinal[train_idx][mask_train]

        # 再登記されるものだけ使う（binary=0）
        mask_val = y_binary[val_idx] == 0
        X_val = X.iloc[val_idx].loc[mask_val]
        y_ord_val = y_ordinal[val_idx][mask_val]


        # torch tensor に変換
        X_train_mt = torch.tensor(X_train.values, dtype=torch.float32)
        y_ord_train = torch.tensor(y_ord_train, dtype=torch.long)

        X_val_mt = torch.tensor(X_val.values, dtype=torch.float32)
        y_ord_val = torch.tensor(y_ord_val, dtype=torch.long)

        # DataLoader
        train_dataset = TensorDataset(X_train_mt, y_ord_train)
        train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, generator=torch.Generator().manual_seed(seed * 100 + fold))

        # 損失記録用リスト
        train_losses = []
        val_losses = []


        coral_model = CoralOrdinalNN(input_dim=X.shape[1], num_classes=num_ord_classes_coral)

        optimizer = torch.optim.Adam(coral_model.parameters(), lr=5e-4)


        num_epochs = 100

        # Early Stopping
        early_stop_patience = 10
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        

        for epoch in trange(num_epochs, desc="Training Epochs"):
            coral_model.train()
            total_train_loss = 0
            for xb, y_ord in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
                optimizer.zero_grad()
                probs_ord, _ = coral_model(xb)
                loss = coral_loss(probs_ord, y_ord, num_ord_classes_coral)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
            
            avg_train_loss = total_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # --- Validation損失の計算 ---
            coral_model.eval()
            with torch.no_grad():
                probs_ord_val, _ = coral_model(X_val_mt)
                val_loss = coral_loss(probs_ord_val, y_ord_val, num_ord_classes_coral)
                avg_val_loss = val_loss.item()
                val_losses.append(avg_val_loss)

            print(f"Epoch {epoch+1:3d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

            # Early Stopping チェック
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = copy.deepcopy(coral_model.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break


        # ベストモデルを復元
        coral_model.load_state_dict(best_model_state)
        
        # 保存ディレクトリを作成（なければ作る）
        learning_curve_dir = os.path.join(result_dir, "learning_curve_coral")
        os.makedirs(learning_curve_dir, exist_ok=True)

        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Learning Curve (Seed {seed+1}, Fold {fold+1}, CORAL)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(learning_curve_dir, f"learning_curve_coral_seed{seed+1}_fold{fold+1}.png"))
        plt.close()


        
        end_time = time.time()
        print(f"CORALの学習完了（経過時間: {end_time - start_time:.2f} 秒）")


        # ========== Step 1-2: マルチラベルNNの学習 ==========
        print("マルチラベルNNの学習を開始...")
        start_time = time.time()

        # 再登記されるものだけ使う（binary=0）
        mask_train = y_binary[train_idx] == 0
        X_train = X.iloc[train_idx].loc[mask_train]
        y_multi_train = y_multilabel[train_idx][mask_train]

        # 再登記されるものだけ使う（binary=0）
        mask_val = y_binary[val_idx] == 0
        X_val = X.iloc[val_idx].loc[mask_val]
        y_multi_val = y_multilabel[val_idx][mask_val]



        # torch tensor に変換
        X_train_mt = torch.tensor(X_train.values, dtype=torch.float32)
        y_multi_train = torch.tensor(y_multi_train, dtype=torch.float32)

        X_val_mt = torch.tensor(X_val.values, dtype=torch.float32)
        y_multi_val = torch.tensor(y_multi_val, dtype=torch.float32)

        # DataLoader
        train_dataset = TensorDataset(X_train_mt, y_multi_train)
        train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, generator=torch.Generator().manual_seed(seed * 100 + fold))

        # 損失記録用リスト
        train_losses = []
        val_losses = []


        multilabelNN_model = MultiLabelNN(input_dim=X.shape[1], num_labels=y_multilabel.shape[1])
        optimizer = torch.optim.Adam(multilabelNN_model.parameters(), lr=5e-4)
        loss_fn_multi = nn.BCELoss()


        num_epochs = 100

        # Early Stopping
        early_stop_patience = 10
        best_val_loss = float('inf') 
        patience_counter = 0
        best_model_state = None
        

        for epoch in trange(num_epochs, desc="Training Epochs"):
            multilabelNN_model.train()
            total_train_loss = 0
            for xb, y_multi in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
                optimizer.zero_grad()
                probs_multi = multilabelNN_model(xb)
                loss = loss_fn_multi(probs_multi, y_multi)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
            
            avg_train_loss = total_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # --- Validation損失の計算 ---
            multilabelNN_model.eval()
            with torch.no_grad():
                probs_multi_val = multilabelNN_model(X_val_mt)
                val_loss = loss_fn_multi(probs_multi_val, y_multi_val)
                avg_val_loss = val_loss.item()
                val_losses.append(avg_val_loss)

            print(f"Epoch {epoch+1:3d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

            # Early Stopping チェック
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = copy.deepcopy(multilabelNN_model.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break


        # ベストモデルを復元
        multilabelNN_model.load_state_dict(best_model_state)

        # 保存ディレクトリを作成（なければ作る）
        learning_curve_dir = os.path.join(result_dir, "learning_curve_multilabelNN")
        os.makedirs(learning_curve_dir, exist_ok=True)

        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Learning Curve (Seed {seed+1}, Fold {fold+1}, MultilabelNN)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(learning_curve_dir, f"learning_curve_multilabelNN_seed{seed+1}_fold{fold+1}.png"))
        plt.close()


        
        end_time = time.time()
        print(f"マルチラベルNNの学習完了（経過時間: {end_time - start_time:.2f} 秒）")

        print("学習フェーズ完了")
        print("--------------------------------")


        # ========== Step 2: 推論 ==========
        print("--------------------------------")
        print("推論フェーズ開始")

        # Step 1: テストデータを取得
        X_test = X.iloc[test_idx]
        y_bin_test = y_binary[test_idx]
        y_ord_test = y_ordinal[test_idx]
        y_multilabel_test = y_multilabel[test_idx]

        # Step 2: 二値分類予測
        y_bin_pred = clf_bin_lgb.predict(X_test)
        mask_bin0 = y_bin_pred == 0

        # Step 3-1: NNの対象だけ取り出して順序回帰の推論
        X_test_masked = torch.tensor(X_test[mask_bin0].values, dtype=torch.float32)

        with torch.no_grad():
            probs_ord_masked, _ = coral_model(X_test_masked)
            preds_ord_masked = predict_classes(probs_ord_masked)
        
        # 統合予測ラベルの作成（順序分類，再登記されないと予測されたものは全てno_re_registration_class、再登記されるものにはNNの予測を代入）
        no_re_registration_class = label_names.index('120- months')  # → 5
        preds_ord_final = np.full_like(y_ord_test, fill_value=no_re_registration_class)
        preds_ord_final[mask_bin0] = preds_ord_masked.numpy()


        # Step 3-2: NNの対象だけ取り出してマルチラベル分類の推論
        X_test_masked = torch.tensor(X_test[mask_bin0].values, dtype=torch.float32)

        with torch.no_grad():
            probs_multilabel_masked = multilabelNN_model(X_test_masked)
            preds_multilabel_masked = (probs_multilabel_masked > 0.5).int().numpy()
        
        # 統合予測ラベルの作成（マルチラベル分類，再登記されないと予測されたものは全て0、再登記されるものにはNNの予測を代入）
        preds_multilabel_final = np.zeros_like(y_multilabel_test)
        preds_multilabel_final[mask_bin0] = preds_multilabel_masked



        # if fold == 0 and seed == 0:
        #     print("SHAPの計算を開始...")
        #     # 保存ディレクトリを作成（なければ作る）
        #     shap_dir = os.path.join(result_dir, "shap_summary_plot")
        #     os.makedirs(shap_dir, exist_ok=True)

        #     print("LightGBMのSHAPの計算を開始...")
        #     explainer_lgb = shap.TreeExplainer(clf_bin_lgb)
        #     shap_values_lgb = explainer_lgb(X_test)
        #     plt.figure()
        #     shap.summary_plot(shap_values_lgb, X_test, show=False)
        #     plt.title(f"SHAP Summary LightGBM Seed {seed+1} Fold {fold+1}")
        #     plt.tight_layout()
        #     plt.savefig(os.path.join(shap_dir, f"shap_summary_lgb_seed{seed+1}_fold{fold+1}.png"))
        #     plt.close()
        #     print("LightGBMのSHAPの計算完了")

        #     print("CORALのSHAPの計算を開始...")

        #     predict_fn = lambda x: predict_midpoint(coral_model, pd.DataFrame(x, columns=X_test.columns))

        #     explainer_coral = shap.Explainer(predict_fn, X_test, model_output="raw")
        #     shap_values_coral = explainer_coral(X_test)

        #     shap.summary_plot(shap_values_coral, X_test, show=False)
        #     plt.title(f"SHAP Summary for Predicted Midpoint (months) Seed {seed+1} Fold {fold+1}")
        #     plt.tight_layout()
        #     plt.savefig(os.path.join(shap_dir, f"shap_summary_y_pred_mid_seed{seed+1}_fold{fold+1}.png"))
        #     plt.close()
        #     print("CORALのSHAPの計算完了（y_pred_midベース）")

        #     print("SHAPの計算完了")



        print("推論フェーズ完了")
        print("--------------------------------")


        # ---------- Step 3: 評価 ----------
        print("--------------------------------")
        print("評価フェーズ開始")

        # ★追加：売買ラベル真値（次回登記原因が売買かどうか）
        is_sale_next_true = (y_multilabel_test[:, sale_label_idx] == 1)

        # 二値分類の評価
        acc_bin = accuracy_score(y_bin_test, y_bin_pred)
        precision_bin = precision_score(y_bin_test, y_bin_pred, zero_division=0)
        recall_bin = recall_score(y_bin_test, y_bin_pred, zero_division=0)
        f1_bin = f1_score(y_bin_test, y_bin_pred, zero_division=0)
        auc_bin = roc_auc_score(y_bin_test, clf_bin_lgb.predict_proba(X_test)[:, 1])

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
        y_true_days = df.loc[test_idx, "days_until_next"].values
        y_pred_days = np.array([label_to_midpoint[y] for y in preds_ord_final])
        # acc = accuracy_score(y_ord_test, preds_ord_final)
        mae = mean_absolute_error(y_true_days, y_pred_days)
        mse = mean_squared_error(y_true_days, y_pred_days)
        rmse = np.sqrt(mse)
        corr, _ = pearsonr(y_true_days, y_pred_days)
        # spearman, _ = spearmanr(y_ord_test, preds_ord_final)
        # qwk = cohen_kappa_score(y_ord_test, preds_ord_final, weights='quadratic')

        # print(f"[最終的な順序分類評価指標] Accuracy: {acc:.4f}, MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, Spearman: {spearman:.4f}, QWK: {qwk:.4f}")
        print(f"[最終的な順序分類評価指標]  MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, Corr: {corr:.4f}")



        # ===== Final preds (argmax 後) に対する「各しきい値 y ≤ k」の二値分類指標（AUCなし） =====
        print("\n[Final (argmax) y ≤ k : Binary Classification Metrics]")
        for k in range(coral_output_dim):  # 0..K-2
            # 真値と予測を二値化
            y_bin_true_k = (y_ord_test <= k).astype(int)
            y_bin_pred_k = (preds_ord_final <= k).astype(int)

            acc_k = accuracy_score(y_bin_true_k, y_bin_pred_k)
            precision_k = precision_score(y_bin_true_k, y_bin_pred_k, zero_division=0)
            recall_k = recall_score(y_bin_true_k, y_bin_pred_k, zero_division=0)
            f1_k = f1_score(y_bin_true_k, y_bin_pred_k, zero_division=0)

            # 保存
            final_bincls_scores[k]["accuracy"].append(acc_k)
            final_bincls_scores[k]["precision"].append(precision_k)
            final_bincls_scores[k]["recall"].append(recall_k)
            final_bincls_scores[k]["f1"].append(f1_k)

            # 表示
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
        metrics = compute_ordered_metrics(y_true_days, y_pred_days)
        metrics["Fold"] = fold + 1
        all_metrics.append(metrics)


        # ======= 順序分類の散布図のための格納 =======
        # 順序分類の混同行列のための格納
        all_y_true.extend(y_ord_test.tolist())
        all_y_pred.extend(preds_ord_final.tolist())


        # 統合フェーズ（散布図/相関/ヒスト）用に蓄積
        all_true_days.extend(y_true_days.tolist())
        all_pred_days.extend(y_pred_days.tolist())




        # --------- Step 3-2: マルチラベル分類の評価 ----------
        report_dict = classification_report(
            y_multilabel_test,
            preds_multilabel_final,
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

        
        # ★ここから追加：DM送付施策（売買限定）の混同行列をHごとに計算
        # days_until_next（真の登記間隔日数）は既に y_true_days として取得済み
        true_days_test = y_true_days  # alias（分かりやすさのため）

        for H in H_MONTHS_LIST:
            max_cat = H_TO_MAX_CATEGORY[H]

            # 真値: Hヶ月以内に次の登記原因が売買
            threshold_days = 31 * H
            true_positive_mask = (is_sale_next_true) & (true_days_test <= threshold_days)

            # 予測: Hヶ月以内に再登記が起こると予測 ＆ 次の登記原因が売買と予測
            pred_within_H = (preds_ord_final <= max_cat)
            pred_sale = (preds_multilabel_final[:, sale_label_idx] == 1)
            dm_send = pred_within_H & pred_sale

            tp = np.sum(dm_send & true_positive_mask)
            fp = np.sum(dm_send & (~true_positive_mask))
            fn = np.sum((~dm_send) & true_positive_mask)
            tn = np.sum((~dm_send) & (~true_positive_mask))

            dm_confusion[H]["TP"] += int(tp)
            dm_confusion[H]["FP"] += int(fp)
            dm_confusion[H]["FN"] += int(fn)
            dm_confusion[H]["TN"] += int(tn)
        # ★追加ここまで




        print("評価フェーズ完了")

        print("--------------------------------")



print("--------------------------------")
print("統合評価の保存フェーズ開始")
# ========= 統合評価CSVの出力（完全版） =========
summary_all_path = os.path.join(result_dir, "metrics_all_summary_lgb_coral+multilabelNN.csv")

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
        writer.writerow(["Ordinal(days_until_next)", "-", name, mean, ci_lower, ci_upper])



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

    # 5. マルチラベル分類（Macro & Weighted）
    for avg_type, metrics in [("Macro", multilabel_macro_metrics),
                               ("Weighted", multilabel_weighted_metrics)]:
        for metric in ["precision", "recall", "f1-score"]:
            values = metrics[metric]
            mean, ci_lower, ci_upper = mean_ci(values)
            writer.writerow(["Multilabel", avg_type, metric, mean, ci_lower, ci_upper])

    # 6. マルチラベル分類（ラベルごと）
    for label in multilabel_colnames:
        for metric in ["precision", "recall", "f1-score"]:
            values = per_label_scores[label][metric]
            mean, ci_lower, ci_upper = mean_ci(values)
            writer.writerow(["Multilabel", label, metric, mean, ci_lower, ci_upper])
    
    # 7. 順序回帰（実測日数 vs 予測日数）の Pearson 相関係数
    true_days_np = np.asarray(all_true_days, dtype=float)
    pred_days_np = np.asarray(all_pred_days, dtype=float)
    corr, pval = pearsonr(true_days_np, pred_days_np)
    writer.writerow(["Ordinal", "-", "Pearson_corr_days", corr, "-", f"p={pval}"])






# ==== 二値分類 全fold統合の混同行列を作成・保存 ====
cm_bin_all = confusion_matrix(all_y_bin_true, all_y_bin_pred, labels=[0, 1])
cm_bin_all_df = pd.DataFrame(cm_bin_all, index=["True: 0", "True: 1"], columns=["Pred: 0", "Pred: 1"])

plt.figure(figsize=(8, 6))
sns.heatmap(cm_bin_all_df, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix (Binary Classification - All Folds)")
plt.tight_layout()
plt.savefig(os.path.join(result_dir, "confusion_matrix_all_folds_lgb.png"))
plt.close()


# ==== 順序分類 全fold統合の混同行列を作成・保存 ====
label_indices = list(range(len(label_names)))  # → [0, 1, 2, 3, 4, 5]

cm_all = confusion_matrix(all_y_true, all_y_pred, labels=label_indices)
cm_all_df = pd.DataFrame(cm_all, index=label_names, columns=label_names)

plt.figure(figsize=(10, 8))
sns.heatmap(cm_all_df, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix (Ordinal(days_until_next) - All Folds)")
plt.tight_layout()
plt.savefig(os.path.join(result_dir, "confusion_matrix_all_folds_lgb_coral+multilabel.png"))
plt.close()


# ==== 散布図（実測日数 vs 予測日数） ====
plt.figure(figsize=(8, 6))
plt.scatter(true_days_np, pred_days_np, alpha=0.3, s=20, edgecolor='none')
max_val = max(true_days_np.max(), pred_days_np.max())
plt.plot([0, max_val], [0, max_val], color='red', linestyle='--', linewidth=1.5, label="y=x")
plt.xlabel("True Days Until Next Registration")
plt.ylabel("Predicted Days Until Next Registration")
plt.title("True vs Predicted Days Until Next Registration (All Folds)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(result_dir, "scatter_true_vs_pred_days_until_next.png"))
plt.close()


# ==== 予測カテゴリごとの正解日数ヒストグラム ====
print("予測カテゴリごとの正解日数ヒストグラムを作成中...")

y_pred_cat_np = np.asarray(all_y_pred, dtype=int)  # 予測カテゴリ（0..5）
true_days_np = np.asarray(all_true_days, dtype=float)  # 実測日数

hist_dir = os.path.join(result_dir, "hist_true_days_by_predcat")
os.makedirs(hist_dir, exist_ok=True)

# 個別（カテゴリごと1枚）
for cat in range(len(label_names)):  # 0..5
    mask = (y_pred_cat_np == cat)
    true_days_cat = true_days_np[mask]
    if true_days_cat.size == 0:
        continue

    plt.figure(figsize=(8, 4))
    plt.hist(true_days_cat, bins=50, alpha=0.8, edgecolor="black")
    plt.xlabel("True Days Until Next Registration")
    plt.ylabel("Count")
    plt.title(f"True Days Histogram (Predicted Category = {cat}: {label_names[cat]})")
    plt.tight_layout()
    plt.savefig(os.path.join(hist_dir, f"hist_true_days_by_predcat_{cat}.png"))
    plt.close()

# まとめ図（2x3 グリッド）
fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
axes = axes.ravel()
for cat in range(len(label_names)):
    ax = axes[cat]
    mask = (y_pred_cat_np == cat)
    true_days_cat = true_days_np[mask]
    if true_days_cat.size > 0:
        ax.hist(true_days_cat, bins=50, alpha=0.85, edgecolor="black")
    ax.set_title(f"{cat}: {label_names[cat]}")

for i, ax in enumerate(axes):
    if i % 3 == 0:
        ax.set_ylabel("Count")
    if i >= 3:
        ax.set_xlabel("True Days Until Next Registration")

plt.suptitle("True Days per Predicted Category", y=1.02, fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(hist_dir, "hist_true_days_by_predcat_grid.png"))
plt.close()

print("予測カテゴリごとの正解日数ヒストグラムを保存しました。")




print("統合評価の保存フェーズ完了")
print("--------------------------------")


# ★追加：DM施策（売買限定）のシミュレーション結果を集計・保存
dm_sim_path = os.path.join(result_dir, "dm_simulation_sale_within_Hmonths.csv")

with open(dm_sim_path, "w", newline="", encoding="utf-8") as f:
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
        cnt = dm_confusion[H]
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

print(f"DMシミュレーション結果を保存しました: {dm_sim_path}")



