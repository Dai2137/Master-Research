
# Real Estate Re-registration Prediction

本リポジトリは，日本の不動産登記データ（2014–2023）を用いて，
**再登記が発生するか否か**，
**再登記までの期間**，
**再登記原因**
を予測する機械学習モデルを実装・評価するための研究用コードである．

---

## Design Policy

- `train.py` に集約せず，**1つの `.py` = 1つの手法・実験設定**とする
- 比較手法・提案手法を**コード構造レベルで明確に分離**
- 実験の再現性・可読性を最優先
- 将来的なデータ追加（年次データ）に耐える前処理設計

---

## Directory Structure

```text

├─ .ipynb_checkpoints/
├─ .vscode/
│
├─ data/
│  ├─ raw/
│  │  ├─ 2014_受付帳.csv
│  │  ├─ 2015_受付帳.csv
│  │  ├─ ...
│  │  ├─ 2021_受付帳.csv
│  │  └─ 2022-2023_受付帳.csv
│  │
│  └─ after_preprocess/
│     └─ land_data_for_prediction.csv
│
├─ notebooks/
│  ├─ dm_simulation/
│  │  ├─ dm_topN_simulation_binary_coral.py
│  │  └─ dm_topN_simulation_binaryH.py
│  │
│  └─ Prediction/
│     ├─ binary+ordinal_multilabel/
│     │  └─ single/
│     │     └─ binary+ordinal(+multilabel)/
│     │        ├─ lgb+coral+multilabelNN/
│     │        │  └─ binary_ordinal_lgb_multilabelNN.py
│     │        ├─ lgb+olr/
│     │        │  └─ binary_ordinal_lgb_olr.py
│     │        └─ lgb+orf/
│     │           └─ binary_ordinal_lgb_orf.py
│     │
│     └─ ordinal_multilabel/
│        └─ single/
│           ├─ multilabel/
│           │  └─ multilabelNN/
│           │     └─ multilabel_multilabelNN.py
│           └─ ordinal/
│              ├─ coral/
│              │  └─ ordinal_coral.py
│              ├─ olr/
│              │  └─ ordinal_olr.py
│              └─ orf/
│                 └─ ordinal_orf.py
│
├─ Preprocess_ipynb/
│  ├─ 14-23_受付帳_各登記原因_*.ipynb
│  └─ typedata_daily_to_type_data_*.ipynb
│
├─ preprocess.py
├─ results/
├─ requirements.txt
└─ .gitignore
````

---

## Execution

### 1. Raw Data Placement

以下のパスに，年次ごとの登記受付帳データを配置する：

```text
./data/raw/
```

ファイル名は以下を想定する：

* `YYYY_受付帳.csv`（単年）
* `YYYY-YYYY_受付帳.csv`（複数年まとめ）

例：

* `2014_受付帳.csv`
* `2015_受付帳.csv`
* `2022-2023_受付帳.csv`

---

### 2. Preprocessing

前処理スクリプトを実行する：

```bash
python preprocess.py
```

`preprocess.py` は `data/raw/` 配下の
**すべての `*_受付帳.csv` を自動で結合**し，以下を行う：

* 登記原因の正規化・グルーピング
* 土地データの抽出
* 同日登記の集約
* 次回登記までの日数算出
* 再登記原因（マルチラベル）の生成
* 特徴量・目的変数の作成

処理後，以下のファイルが生成される：

```text
./data/after_preprocess/land_data_for_prediction.csv
```

以降のすべての学習・評価スクリプトはこの CSV を入力として使用する．

---

## Model Training and Evaluation

本プロジェクトでは，**モデル構成ごとに `.py` を完全分離**している．
各スクリプトは単体で実行可能であり，
学習・評価・結果保存までを内部で完結させている．

---

### Binary + Ordinal + Multi-label

（`binary+ordinal_multilabel/`）

二段階構成：

1. 再登記が発生するか否かを二値分類
2. 再登記が発生すると判定された物件に対して

   * 再登記までの日数：順序回帰
   * 再登記原因：マルチラベル分類

#### 実装手法

* `lgb+coral+multilabelNN`

  * Binary: LightGBM
  * Ordinal: CORAL
  * Multi-label: Neural Network

* `lgb+olr`

  * Binary: LightGBM
  * Ordinal: Ordered Logistic Regression

* `lgb+orf`

  * Binary: LightGBM
  * Ordinal: Ordered Random Forest

---

### Ordinal + Multi-label（Binary なし）

（`ordinal_multilabel/`）

二値分類を用いず，
「再登記しない」クラスを含めて直接学習する構成．

#### Multi-label only

* `multilabel_multilabelNN.py`
  再登記原因のみを予測するベースライン．

#### Ordinal only

* `ordinal_coral.py`
* `ordinal_olr.py`
* `ordinal_orf.py`

順序回帰手法の違いのみを比較するための構成である．

---

## DM Simulation

`notebooks/dm_simulation/` では，
学習済みモデルの出力を用いて **DM 施策を想定したシミュレーション**を行う．

主な内容：

* 物件スコアリング
* Top-N 抽出
* 再登記発生率・案件化数の評価
* ベースラインとの比較

### Files

* `dm_topN_simulation_binary_coral.py`
  提案手法

* `dm_topN_simulation_binaryH.py`
  比較手法

予測精度ではなく，
**実運用上のビジネス価値を定量評価すること**を目的としている．

---

## Environment

* OS: Windows
* Python: 3.x

主なライブラリ：

* numpy / pandas
* scikit-learn
* LightGBM
* PyTorch
* SHAP

---

## Notes


* 特徴量名・評価指標・図表は英語表記を前提
* 各 `.py` は論文中の「比較手法」「提案手法」と1対1で対応する設計

---

## License

Research use only.

```