import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from lightgbm import LGBMClassifier
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
import shap


# 保存パスの指定と準備（変更するの忘れないように！！）
result_dir = r"D:\fujiwara\M\result\binary+ordinal_multilabel\multi\lgb+multitaskcoralNN"
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


# ---------- CORAL ----------
class CoralOrdinalNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.ord_specific_layer = nn.Linear(32, 16)  # 順序回帰専用のタスク固有層
        self.shared_weight = nn.Parameter(torch.randn(16))  # 重み共有用
        self.raw_bias = nn.Parameter(torch.zeros(num_classes - 1))
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x_shared = self.relu(self.fc2(x))
        x_ord = self.relu(self.ord_specific_layer(x_shared))
        logits = x_ord @ self.shared_weight
        logits = logits.unsqueeze(1)
        ordered_bias = torch.cumsum(F.softplus(self.raw_bias), dim=0)
        logits = logits + ordered_bias
        probs = torch.sigmoid(logits)
        return probs, x_shared  # x_sharedはマルチラベル分類側でも利用


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


# ---------- MultiTask CORAL ----------
class MultiTaskCORAL(nn.Module):
    def __init__(self, input_dim, num_ord_classes, num_multilabels):
        super().__init__()
        self.coral_module = CoralOrdinalNN(input_dim, num_ord_classes)
        self.multilabel_specific_layer = nn.Linear(32, 16)  # マルチラベル分類専用のタスク固有層
        self.multi_out = nn.Linear(16, num_multilabels)
        self.relu = nn.ReLU()

    def forward(self, x):
        probs_ord, x_shared = self.coral_module(x)
        x_multi = self.relu(self.multilabel_specific_layer(x_shared))
        probs_multi = torch.sigmoid(self.multi_out(x_multi))
        return probs_ord, probs_multi
    

def predict_expected_class(model, X_df):
    """
    CORALモデルの出力から予測されるクラス期待値（soft class）を返す関数
    SHAPによる可視化のために使用

    Parameters:
        model : torch.nn.Module
            学習済みのCORALモデル（foldごとに異なる）
        X_df : pd.DataFrame
            特徴量名付きDataFrame（.iloc[] で分割後のX_testなど）

    Returns:
        np.ndarray : shape (N, 1), Expected class per sample
    """
    model.eval()
    x_tensor = torch.tensor(X_df.values, dtype=torch.float32)
    with torch.no_grad():
        probs, _ = model(x_tensor)  # Output: (N, K−1)
        probs_ext = torch.cat([
            torch.zeros((probs.shape[0], 1)),
            probs,
            torch.ones((probs.shape[0], 1))
        ], dim=1)
        probs_exact = probs_ext[:, 1:] - probs_ext[:, :-1]  # shape: (N, K)
        expected_class = (probs_exact * torch.arange(0, probs_exact.shape[1])).sum(dim=1)
    return expected_class.cpu().numpy().reshape(-1, 1)



# ---------- データ読み込み ----------
print("データ読み込みを開始...")
df = pd.read_csv(r"D:\fujiwara\M\data\after_preprocess\land_data_for_prediction.csv")
print("データ読み込み完了")

# ラベル名の取得（マルチラベル分類用）
multilabel_colnames = [col for col in df.columns if col.startswith("on_day_reason_group_") and col.endswith("_next")]


X = df.drop(columns=['will_not_be_re_registered', 'days_until_next_category'] +
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


# 評価指標格納用
all_metrics = []
metric_names = ["Accuracy", "MAE", "MSE", "RMSE", "Spearman", "QWK"]

all_y_bin_true = []
all_y_bin_pred = []

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

# CORALの二値分類の評価指標格納用
coral_per_task_scores = {  # foldごとに保持
    k: {"accuracy": [], "precision": [], "recall": [], "f1": []}
    for k in range(coral_output_dim)
}


# CORALの二値分類のAUC格納用
coral_auc_per_task = {k: [] for k in range(coral_output_dim)}


label_names = [
    '-1 months', 
    '1-4 months', 
    '4-9 months', 
    '9-24 months', 
    '24-120 months', 
    '120- months'
]


# ラベルごとのスコア格納用（全fold分,マルチラベル分類）
per_label_scores = {label: {"precision": [], "recall": [], "f1-score": []} for label in multilabel_colnames}



seeds = list(range(5))

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

        # ========== Step 1-2: マルチタスクNNの学習 ==========
        print("マルチタスクNNの学習を開始...")
        start_time = time.time()

        # 再登記されるものだけ使う（binary=0）
        mask_train = y_binary[train_idx] == 0
        X_train = X.iloc[train_idx].loc[mask_train]
        y_ord_train = y_ordinal[train_idx][mask_train]
        y_multi_train = y_multilabel[train_idx][mask_train]

        # 再登記されるものだけ使う（binary=0）
        mask_val = y_binary[val_idx] == 0
        X_val = X.iloc[val_idx].loc[mask_val]
        y_ord_val = y_ordinal[val_idx][mask_val]
        y_multi_val = y_multilabel[val_idx][mask_val]


        # torch tensor に変換
        X_train_mt = torch.tensor(X_train.values, dtype=torch.float32)
        y_ord_train = torch.tensor(y_ord_train, dtype=torch.long)
        y_multi_train = torch.tensor(y_multi_train, dtype=torch.float32)

        X_val_mt = torch.tensor(X_val.values, dtype=torch.float32)
        y_ord_val = torch.tensor(y_ord_val, dtype=torch.long)
        y_multi_val = torch.tensor(y_multi_val, dtype=torch.float32)

        # DataLoader
        train_dataset = TensorDataset(X_train_mt, y_ord_train, y_multi_train)
        train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, generator=torch.Generator().manual_seed(seed * 100 + fold))

        # 損失記録用リスト
        train_losses = []
        val_losses = []


        model = MultiTaskCORAL(input_dim=X.shape[1], num_ord_classes=num_ord_classes_coral, num_multilabels=y_multilabel.shape[1])
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
            for xb, y_ord, y_multi in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
                optimizer.zero_grad()
                probs_ord, probs_multi = model(xb)
                loss1 = coral_loss(probs_ord, y_ord, num_ord_classes_coral)
                loss2 = loss_fn_multi(probs_multi, y_multi)
                loss = loss1 + 1.66 * loss2
                
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
            
            avg_train_loss = total_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # --- Validation損失の計算 ---
            model.eval()
            with torch.no_grad():
                probs_ord_val, probs_multi_val = model(X_val_mt)
                val_loss1 = coral_loss(probs_ord_val, y_ord_val, num_ord_classes_coral)
                val_loss2 = loss_fn_multi(probs_multi_val, y_multi_val)
                avg_val_loss = (val_loss1 + val_loss2).item()
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
        print(f"マルチタスクNNの学習完了（経過時間: {end_time - start_time:.2f} 秒）")
        print("学習フェーズ完了")
        print("--------------------------------")


        # ========== Step 2: 推論 ==========
        print("--------------------------------")
        print("推論フェーズ開始")

        # Step 1: テストデータを取得
        X_test = X.iloc[test_idx]
        y_bin_test = y_binary[test_idx]
        y_ord_test = y_ordinal[test_idx]
        y_multi_test = y_multilabel[test_idx]

        # Step 2: 二値分類予測
        y_bin_pred = clf_bin_lgb.predict(X_test)
        mask_bin0 = y_bin_pred == 0

        # Step 3: NNの対象だけ取り出して推論
        X_test_masked = torch.tensor(X_test[mask_bin0].values, dtype=torch.float32)

        with torch.no_grad():
            probs_ord_masked, probs_multi_masked = model(X_test_masked)
            preds_ord_masked = predict_classes(probs_ord_masked)
            preds_multi_masked = (probs_multi_masked > 0.5).int().numpy()

        # Step 4: 統合予測ラベルの作成（順序 + マルチラベル）
        no_re_registration_class = label_names.index('120- months')  # → 5
        preds_ord_final = np.full_like(y_ord_test, fill_value=no_re_registration_class)
        preds_ord_final[mask_bin0] = preds_ord_masked.numpy()

        preds_multi_final = np.zeros_like(y_multi_test)
        preds_multi_final[mask_bin0] = preds_multi_masked
        
        if fold == 0 and seed == 0:
            print("SHAPの計算を開始...")
            # 保存ディレクトリを作成（なければ作る）
            shap_dir = os.path.join(result_dir, "shap_summary_plot")
            os.makedirs(shap_dir, exist_ok=True)

            print("LightGBMのSHAPの計算を開始...")
            explainer_lgb = shap.TreeExplainer(clf_bin_lgb)
            shap_values_lgb = explainer_lgb(X_test)
            plt.figure()
            shap.summary_plot(shap_values_lgb, X_test, show=False)
            plt.title(f"SHAP Summary LightGBM Seed {seed+1} Fold {fold+1}")
            plt.tight_layout()
            plt.savefig(os.path.join(shap_dir, f"shap_summary_lgb_seed{seed+1}_fold{fold+1}.png"))
            plt.close()
            print("LightGBMのSHAPの計算完了")

            print("CORALのSHAPの計算を開始...")
            
            predict_fn = lambda x: predict_expected_class(model, pd.DataFrame(x, columns=X_test.columns))

            explainer_coral = shap.Explainer(predict_fn, X_test, model_output="raw")
            shap_values_coral = explainer_coral(X_test)

            # 可視化
            shap.summary_plot(shap_values_coral, X_test, show=False)
            plt.title(f"SHAP Summary for Expected Class Seed {seed+1} Fold {fold+1}")
            plt.tight_layout()
            plt.savefig(os.path.join(shap_dir, f"shap_summary_expected_class_seed{seed+1}_fold{fold+1}.png"))
            plt.close()
            print("CORALのSHAPの計算完了")
            print("SHAPの計算完了")



        print("推論フェーズ完了")
        print("--------------------------------")


        # ---------- Step 3: 評価 ----------
        print("--------------------------------")
        print("評価フェーズ開始")

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
        acc = accuracy_score(y_ord_test, preds_ord_final)
        mae = mean_absolute_error(y_ord_test, preds_ord_final)
        mse = mean_squared_error(y_ord_test, preds_ord_final)
        rmse = np.sqrt(mse)
        spearman, _ = spearmanr(y_ord_test, preds_ord_final)
        qwk = cohen_kappa_score(y_ord_test, preds_ord_final, weights='quadratic')

        print(f"[最終的な順序分類評価指標] Accuracy: {acc:.4f}, MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, Spearman: {spearman:.4f}, QWK: {qwk:.4f}")

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

                # --- AUC の追加 ---
                try:
                    auc_k = roc_auc_score(y_ord_bin_targets[:, k], probs_ord_np[:, k])
                except ValueError:
                    auc_k = np.nan  # 正例 or 負例が1クラスしかないと計算できない
                coral_auc_per_task[k].append(auc_k)

                print(f"Task y <= '{label_names[k]}': Acc={acc_k:.4f}, Precision={precision_k:.4f}, Recall={recall_k:.4f}, F1={f1_k:.4f}, AUC={auc_k:.4f}")



        # 順序分類の指標をまとめて記録（1〜6分類）
        metrics = compute_ordered_metrics(y_ord_test, preds_ord_final)
        metrics["Fold"] = fold + 1
        all_metrics.append(metrics)


        # ======= 順序分類の混同行列のための格納 =======
        all_y_true.extend(y_ord_test.tolist())
        all_y_pred.extend(preds_ord_final.tolist())


        # --------- Step 3-2: マルチラベル分類の評価 ----------
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
summary_all_path = os.path.join(result_dir, "metrics_all_summary_lgb_multitaskcoralNN.csv")

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
        writer.writerow(["Ordinal", "-", name, mean, ci_lower, ci_upper])

    # 3. CORAL 順序回帰の各二値タスクごとのスコア
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


    # 4. マルチラベル分類（Macro & Weighted）
    for avg_type, metrics in [("Macro", multilabel_macro_metrics),
                               ("Weighted", multilabel_weighted_metrics)]:
        for metric in ["precision", "recall", "f1-score"]:
            values = metrics[metric]
            mean, ci_lower, ci_upper = mean_ci(values)
            writer.writerow(["Multilabel", avg_type, metric, mean, ci_lower, ci_upper])

    # 5. マルチラベル分類（ラベルごと）
    for label in multilabel_colnames:
        for metric in ["precision", "recall", "f1-score"]:
            values = per_label_scores[label][metric]
            mean, ci_lower, ci_upper = mean_ci(values)
            writer.writerow(["Multilabel", label, metric, mean, ci_lower, ci_upper])


# ==== 二値分類 全fold統合の混同行列を作成・保存 ====
cm_bin_all = confusion_matrix(all_y_bin_true, all_y_bin_pred, labels=[0, 1])
cm_bin_all_df = pd.DataFrame(cm_bin_all, index=["True: 0", "True: 1"], columns=["Pred: 0", "Pred: 1"])

plt.figure(figsize=(8, 6))
sns.heatmap(cm_bin_all_df, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix (Binary Classification - All Folds)")
plt.tight_layout()
plt.savefig(os.path.join(result_dir, "confusion_matrix_binary_lgb_multitaskcoralNN.png"))
plt.close()


# ==== 順序分類 全fold統合の混同行列を作成・保存 ====
label_indices = list(range(len(label_names)))  # → [0, 1, 2, 3, 4, 5]

cm_all = confusion_matrix(all_y_true, all_y_pred, labels=label_indices)
cm_all_df = pd.DataFrame(cm_all, index=label_names, columns=label_names)

plt.figure(figsize=(10, 8))
sns.heatmap(cm_all_df, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix (All Folds)")
plt.tight_layout()
plt.savefig(os.path.join(result_dir, "confusion_matrix_ordinal_lgb_multitaskcoralNN.png"))
plt.close()

print("統合評価の保存フェーズ完了")
print("--------------------------------")
