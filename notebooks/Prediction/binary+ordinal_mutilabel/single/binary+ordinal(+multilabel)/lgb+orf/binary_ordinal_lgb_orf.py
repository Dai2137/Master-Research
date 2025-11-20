import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_absolute_error, mean_squared_error, cohen_kappa_score, confusion_matrix, classification_report
from scipy.stats import spearmanr, t, pearsonr
import time
import os
import matplotlib
matplotlib.use("Agg")  # GUIなしで描画できるバックエンドに変更
import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns
import csv
import shap
from sklearn.ensemble import RandomForestRegressor

# 保存パスの指定と準備（変更するの忘れないように！！）
result_dir = r"D:\fujiwara\M\result\binary+ordinal_multilabel\single\binary+ordinal(+multilabel)\lgb+orf"
os.makedirs(result_dir, exist_ok=True)

# ===== Ordered Forest 実装 =====
class OrderedForest:
    def __init__(self, n_estimators=100, min_samples_leaf=5, max_features="auto"):
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.forest = None

    def __xcheck(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")

    def fit(self, X, y, verbose=True):
        self.__xcheck(X)
        if isinstance(y, pd.Series):
            if y.empty:
                raise ValueError("y Series is empty.")
            y = y.astype(int)
        else:
            raise ValueError("y is not a Pandas Series.")

        nclass = len(y.unique())
        labels = ['Class ' + str(c_idx) for c_idx in range(nclass)]
        forests = {}
        probs = {}

        for class_idx in range(nclass - 1):
            outcome_ind = (y <= class_idx) * 1
            forests[class_idx] = RandomForestRegressor(
                n_estimators=self.n_estimators,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                oob_score=True,
                random_state=42
            )
            forests[class_idx].fit(X=X, y=outcome_ind)
            probs[class_idx] = pd.Series(forests[class_idx].oob_prediction_,
                                         name=labels[class_idx],
                                         index=X.index)

        probs = pd.DataFrame(probs)
        probs_0 = pd.concat([pd.Series(np.zeros(probs.shape[0]), index=probs.index, name=0), probs], axis=1)
        probs_1 = pd.concat([probs, pd.Series(np.ones(probs.shape[0]), index=probs.index, name=nclass)], axis=1)
        class_probs = probs_1 - probs_0.values
        class_probs[class_probs < 0] = 0
        class_probs = class_probs.divide(class_probs.sum(axis=1), axis=0)
        class_probs.columns = labels

        self.nclass = nclass
        self.forest = {'forests': forests, 'probs': class_probs}
        if verbose:
            print("Ordered Forest training complete.")
        return self

    def predict(self, X):
        class_probs = self.predict_proba(X)
        pred_class = pd.Series(class_probs.values.argmax(axis=1), index=X.index)
        return pred_class

    def predict_proba(self, X):
        self.__xcheck(X)
        forests = self.forest['forests']
        labels = list(self.forest['probs'].columns)
        nclass = len(labels)

        probs = {}
        for class_idx in range(nclass - 1):
            probs[class_idx] = pd.Series(forests[class_idx].predict(X=X), name=labels[class_idx], index=X.index)

        probs = pd.DataFrame(probs)
        probs_0 = pd.concat([pd.Series(np.zeros(probs.shape[0]), index=probs.index, name=0), probs], axis=1)
        probs_1 = pd.concat([probs, pd.Series(np.ones(probs.shape[0]), index=probs.index, name=nclass)], axis=1)
        class_probs = probs_1 - probs_0.values
        class_probs[class_probs < 0] = 0
        class_probs = class_probs.divide(class_probs.sum(axis=1), axis=0)
        class_probs.columns = labels
        return class_probs


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


# def predict_expected_class(ord_forest, X_df):
#     """
#     ORFモデルの出力から予測されるクラス期待値（soft class）を返す関数
#     SHAP可視化用
#     """
#     class_probs = ord_forest.predict_proba(X_df).values  # shape: (N, K)
#     expected_class = (class_probs * np.arange(class_probs.shape[1])).sum(axis=1)
#     return expected_class.reshape(-1, 1)


def predict_midpoint(model, X_df):
    """
    ORFモデルの出力から「各クラスの範囲中央値ベース」の予測値を返す関数。
    SHAP可視化に用いる。

    Parameters
    ----------
    model : OrderedForest
        学習済みのORFモデル（foldごとに異なる）
    X_df : pd.DataFrame
        特徴量名付きDataFrame（.iloc[]で分割後のX_testなど）

    Returns
    -------
    np.ndarray : shape (N, 1)
        各サンプルの予測中央値（月単位）
    """
    probs = model.predict_proba(X_df).values
    label_to_midpoint = np.array([31 * 0.5, 31 * 2.5, 31 * 6.5, 31 * 16.5, 31 * 72.0])  # 各クラスの中央値

    y_pred_mid = (probs * label_to_midpoint).sum(axis=1)
    return y_pred_mid.reshape(-1, 1).astype(np.float32)


# ---------- データ読み込み ----------
print("データ読み込みを開始...")
df = pd.read_csv(r"D:\fujiwara\M\data\after_preprocess\land_data_for_prediction.csv")
print("データ読み込み完了")


X = df.drop(columns=['will_not_be_re_registered', 'days_until_next_category', 'days_until_next'] +
            [col for col in df.columns if col.startswith("on_day_reason_group_") and col.endswith("_next")]).astype(np.float32)

y_binary = df['will_not_be_re_registered'].values
y_ordinal = df['days_until_next_category'].values

# 順序クラスの数を決定
num_ord_classes = len(np.unique(y_ordinal))  # ← これで「元のラベルの個数 = 6」が得られる
num_ord_classes_orf = num_ord_classes - 1 # 0〜4の5クラス
orf_output_dim = num_ord_classes_orf - 1 # 0以下〜3以下の4本のランダムフォレスト


binary_metrics = {
    "accuracy": [],
    "precision": [],
    "recall": [],
    "f1": [],
    "auc": []
}


# 評価指標格納用
all_metrics = []
# metric_names = ["Accuracy", "MAE", "MSE", "RMSE", "Spearman", "QWK"]
metric_names = ["MAE", "MSE", "RMSE", "Corr"]

all_y_bin_true = []
all_y_bin_pred = []

# ループ外で定義（foldループの前）
all_y_true = []
all_y_pred = []

# 追加（統合フェーズで実測/予測“日数”を使うため）
all_true_days = []   # 実測日数（days_until_next）
all_pred_days = []   # 予測カテゴリを日数換算した値



# ORFの二値分類の評価指標格納用
orf_per_task_scores = {  # foldごとに保持
    k: {"accuracy": [], "precision": [], "recall": [], "f1": []}
    for k in range(orf_output_dim)
}


# ORFの二値分類のAUC格納用
orf_auc_per_task = {k: [] for k in range(orf_output_dim)}

# 最終予測カテゴリに対する「y ≤ k」二値化の評価（AUCなし）
final_bincls_scores = {
    k: {"accuracy": [], "precision": [], "recall": [], "f1": []}
    for k in range(orf_output_dim)  # 最後は全て1になるので除外
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
    1: 31 *2.5,     # 1–4 months
    2: 31 *6.5,     # 4–9 months
    3: 31 *16.5,    # 9–24 months
    4: 31 *72.0,      # 24–120 months
    5: 31 *120.0      # >120 months（再登記なし）
}


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

        
        # ===== アンダーサンプリング処理 (train_idxのみ適用, 0-4対象) =====
        train_df = df.iloc[train_idx].copy()

        print("Before undersampling (train only, categories 0-4):")
        print(train_df['days_until_next_category'].value_counts().sort_index())

        counts_0to4 = train_df[train_df['days_until_next_category'].between(0, 4)]['days_until_next_category'].value_counts()
        min_cat = counts_0to4.idxmin()
        target_counts = counts_0to4.min()

        target_categories = [c for c in counts_0to4.index if c != min_cat]

        sampled_dfs = []
        for cat in target_categories:
            cat_df = train_df[train_df['days_until_next_category'] == cat]
            sampled_df = cat_df.sample(n=target_counts, random_state=42)
            sampled_dfs.append(sampled_df)

        other_df = train_df[~train_df['days_until_next_category'].isin(target_categories)]
        balanced_train_df = pd.concat(sampled_dfs + [other_df], axis=0).sample(frac=1, random_state=42)

        train_idx = balanced_train_df.index.values

        print("After undersampling (train only, categories 0-4):")
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

        # ========== Step 1-2: ORFの学習 ==========
        print("ORFの学習（グリッドサーチ）を開始...")
        print("Seed:", seed+1, "Fold:", fold+1)
        start_time = time.time()
        print("学習開始時刻：", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)))

        # 再登記されるものだけ使う（binary=0）
        mask_train = y_binary[train_idx] == 0
        X_train = X.iloc[train_idx].loc[mask_train]
        y_ord_train = pd.Series(y_ordinal[train_idx][mask_train], index=X_train.index)

        # 再登記されるものだけ使う（binary=0）
        mask_val = y_binary[val_idx] == 0
        X_val = X.iloc[val_idx].loc[mask_val]
        y_ord_val = pd.Series(y_ordinal[val_idx][mask_val], index=X_val.index)

        # グリッドサーチ設定
        param_grid_orf = {
            "n_estimators": [200, 300],
            "min_samples_leaf": [5],
            "max_features": [0.5]
        }
        # 8+1で1fold6時間かかる

        # 各クラスに対応する中央値（midpoint）を定義
        # 例：<1, 1-4, 4-9, 9-24, 24-120, not re-registered(>120)
        midpoints = np.array([31 * 0.5, 31 * 2.5, 31 * 6.5, 31 * 16.5, 31 * 72.0, 31 * 120.0])

        best_score_orf = np.inf
        best_params_orf = None

        for n_est in param_grid_orf["n_estimators"]:
            for min_leaf in param_grid_orf["min_samples_leaf"]:
                for max_feat in param_grid_orf["max_features"]:
                    # モデル作成・学習
                    orf_tmp = OrderedForest(
                        n_estimators=n_est,
                        min_samples_leaf=min_leaf,
                        max_features=max_feat
                    )
                    orf_tmp.fit(X_train, y_ord_train, verbose=False)
                    
                    # valデータで予測
                    preds_val = orf_tmp.predict(X_val)

                    # midpointsに変換してRMSEを計算
                    y_true_val_mid = midpoints[y_ord_val]
                    y_pred_val_mid = midpoints[preds_val]
                    rmse_val = np.sqrt(mean_squared_error(y_true_val_mid, y_pred_val_mid))
                    
                    # RMSEが小さいほど良いので小さい値をbestとする
                    if rmse_val < best_score_orf:
                        best_score_orf = rmse_val
                        best_params_orf = {
                            "n_estimators": n_est,
                            "min_samples_leaf": min_leaf,
                            "max_features": max_feat
                        }

        print("ORF 最適ハイパーパラメータ:", best_params_orf, "Val RMSE:", best_score_orf)

        # ベストパラメータで再学習（trainのみ）
        ord_forest = OrderedForest(**best_params_orf)
        ord_forest.fit(X_train, y_ord_train, verbose=True)

        end_time = time.time()
        print("学習終了時刻：", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)))
        print(f"ORFの学習完了（経過時間: {end_time - start_time:.2f} 秒）")
        print("学習フェーズ完了")
        print("--------------------------------")



        # ========== Step 2: 推論 ==========
        print("--------------------------------")
        print("推論フェーズ開始")

        # Step 1: テストデータを取得
        X_test = X.iloc[test_idx]
        y_bin_test = y_binary[test_idx]
        y_ord_test = y_ordinal[test_idx]

        # Step 2: 二値分類予測
        y_bin_pred = clf_bin_lgb.predict(X_test)
        mask_bin0 = y_bin_pred == 0

        # Step 3: ORF対象だけ推論（確率も取得）
        probs_ord_masked_df = ord_forest.predict_proba(X_test.loc[mask_bin0])
        preds_ord_masked = probs_ord_masked_df.values.argmax(axis=1)

        # Step 4: 統合予測ラベルの作成（順序分類）
        no_re_registration_class = label_names.index('120- months')  # → 5
        preds_ord_final = np.full_like(y_ord_test, fill_value=no_re_registration_class)
        preds_ord_final[mask_bin0] = preds_ord_masked

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

            # print("ORFのSHAPの計算を開始...")
            
            # predict_fn = lambda x: predict_midpoint(ord_forest, pd.DataFrame(x, columns=X_test.columns))

            # explainer_orf = shap.Explainer(predict_fn, X_test, model_output="raw")
            # shap_values_orf = explainer_orf(X_test)

            # # 可視化
            # shap.summary_plot(shap_values_orf, X_test, show=False)
            # plt.title(f"SHAP Summary for Expected Class Seed {seed+1} Fold {fold+1}")
            # plt.tight_layout()
            # plt.savefig(os.path.join(shap_dir, f"shap_summary_expected_class_seed{seed+1}_fold{fold+1}.png"))
            # plt.close()
            # print("ORFのSHAPの計算完了")
            # 10日かかるからmarginal effectで計算する
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
        # 実測は days_until_next（“日数”）
        y_true_days = df.loc[test_idx, "days_until_next"].values

        # 予測は「カテゴリ → 中央日数」
        y_pred_days = np.array([label_to_midpoint[y] for y in preds_ord_final])

        # メトリクス（日数ベース）
        mae = mean_absolute_error(y_true_days, y_pred_days)
        mse = mean_squared_error(y_true_days, y_pred_days)
        rmse = np.sqrt(mse)
        corr, _ = pearsonr(y_true_days, y_pred_days)
        print(f"[最終的な順序分類評価指標] MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, Corr: {corr:.4f}")

        # 統合フェーズ用に蓄積

        # ======= 順序分類の混同行列のための格納 =======
        all_y_true.extend(y_ord_test.tolist())
        all_y_pred.extend(preds_ord_final.tolist())
        # ======= 順序分類の散布図のための格納 =======
        all_true_days.extend(y_true_days.tolist())
        all_pred_days.extend(y_pred_days.tolist())

        



        # ===== Final preds (argmax後) に対する「各しきい値 y ≤ k」の二値分類指標（AUCなし） =====
        print("\n[Final (argmax) y ≤ k : Binary Classification Metrics]")
        for k in range(orf_output_dim):
            y_bin_true_k = (y_ord_test <= k).astype(int)
            y_bin_pred_k = (preds_ord_final <= k).astype(int)

            acc_k = accuracy_score(y_bin_true_k, y_bin_pred_k)
            precision_k = precision_score(y_bin_true_k, y_bin_pred_k, zero_division=0)
            recall_k = recall_score(y_bin_true_k, y_bin_pred_k, zero_division=0)
            f1_k = f1_score(y_bin_true_k, y_bin_pred_k, zero_division=0)

            final_bincls_scores[k]["accuracy"].append(acc_k)
            final_bincls_scores[k]["precision"].append(precision_k)
            final_bincls_scores[k]["recall"].append(recall_k)
            final_bincls_scores[k]["f1"].append(f1_k)

            print(f"y ≤ '{label_names[k]}' : "
                  f"Acc={acc_k:.4f}, Precision={precision_k:.4f}, "
                  f"Recall={recall_k:.4f}, F1={f1_k:.4f}")
            

        # ORF の出力（二値タスクごとの評価）
        # 累積確率（K-1列）を作成
        class_probs = probs_ord_masked_df.values                     # shape: (n_samples, K)
        cum_probs = np.cumsum(class_probs[:, :-1], axis=1)           # shape: (n_samples, K-1)
        probs_ord_masked = cum_probs  # 二値タスク評価で使用
        if mask_bin0.sum() > 0:
            y_ord_bin_targets = (np.arange(num_ord_classes_orf)[None, :] >= (y_ord_test[mask_bin0][:, None])).astype(int)
            y_ord_bin_preds = (probs_ord_masked > 0.5).astype(int)

            print("\n[ORF内部の二値タスクごとの評価指標]")
            for k in range(orf_output_dim):
                acc_k = accuracy_score(y_ord_bin_targets[:, k], y_ord_bin_preds[:, k])
                precision_k = precision_score(y_ord_bin_targets[:, k], y_ord_bin_preds[:, k], zero_division=0)
                recall_k = recall_score(y_ord_bin_targets[:, k], y_ord_bin_preds[:, k], zero_division=0)
                f1_k = f1_score(y_ord_bin_targets[:, k], y_ord_bin_preds[:, k], zero_division=0)

                orf_per_task_scores[k]["accuracy"].append(acc_k)
                orf_per_task_scores[k]["precision"].append(precision_k)
                orf_per_task_scores[k]["recall"].append(recall_k)
                orf_per_task_scores[k]["f1"].append(f1_k)

                # --- AUC の追加 ---er
                try:
                    auc_k = roc_auc_score(y_ord_bin_targets[:, k], probs_ord_masked[:, k])
                except ValueError:
                    auc_k = np.nan  # 正例 or 負例が1クラスしかないと計算できない
                orf_auc_per_task[k].append(auc_k)

                print(f"Task y <= '{label_names[k]}': Acc={acc_k:.4f}, Precision={precision_k:.4f}, Recall={recall_k:.4f}, F1={f1_k:.4f}, AUC={auc_k:.4f}")





        # 順序分類の指標をまとめて記録（1〜6分類）
        metrics = compute_ordered_metrics(y_true_days, y_pred_days)
        metrics["Fold"] = fold + 1
        all_metrics.append(metrics)





        print("評価フェーズ完了")

        print("--------------------------------")



print("--------------------------------")
print("統合評価の保存フェーズ開始")
# ========= 統合評価CSVの出力（完全版） =========
summary_all_path = os.path.join(result_dir, "metrics_all_summary_lgb_orf.csv")

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


    # 3. Final (argmax) に対する各しきい値 y ≤ k の二値指標（AUCなし）
    for k in range(orf_output_dim):
        label = label_names[k]
        for metric in ["accuracy", "precision", "recall", "f1"]:
            values = final_bincls_scores[k][metric]
            mean, ci_lower, ci_upper = mean_ci(values)
            writer.writerow(["OrdinalBinary(Final-Argmax)", f"y <= '{label}'", metric, mean, ci_lower, ci_upper])

    # 4. ORF 順序回帰の各二値タスクごとのスコア
    for k in range(orf_output_dim):
        label = label_names[k]  # 例: '〜1 month'
        for metric in ["accuracy", "precision", "recall", "f1"]:
            values = orf_per_task_scores[k][metric]
            mean, ci_lower, ci_upper = mean_ci(values)
            writer.writerow(["OrdinalBinary", f"y <= '{label}'", metric, mean, ci_lower, ci_upper])

        # --- AUC の統合評価 ---
        auc_values = [v for v in orf_auc_per_task[k] if not np.isnan(v)]
        if len(auc_values) > 0:
            mean, ci_lower, ci_upper = mean_ci(auc_values)
            writer.writerow(["OrdinalBinary", f"y <= '{label}'", "AUC", mean, ci_lower, ci_upper])
    
    # 5. 順序回帰（“日数”ベース）の相関係数
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
plt.title("Confusion Matrix (All Folds)")
plt.tight_layout()
plt.savefig(os.path.join(result_dir, "confusion_matrix_all_folds_lgb_orf.png"))
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
plt.savefig(os.path.join(result_dir, "scatter_true_vs_pred_days.png"))
plt.close()



# ==== 予測カテゴリごとの正解日数ヒストグラム ====
print("予測カテゴリごとの正解日数ヒストグラムを作成中...")

y_pred_cat_np = np.asarray(all_y_pred, dtype=int)   # 予測カテゴリ（0..5）
true_days_np = np.asarray(all_true_days, dtype=float)  # 実測日数

hist_dir = os.path.join(result_dir, "hist_true_days_by_predcat")
os.makedirs(hist_dir, exist_ok=True)

# カテゴリごと（個別）
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

# まとめ図（2x3）
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
