import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas()  # これがないと .progress_apply は使えない

from pathlib import Path


print("=== Step 0: Load raw data ===")
data_path = Path("data") / "raw" / "2014-2023_受付帳.csv"
data = pd.read_csv(data_path)
print(f"Loaded data: {len(data):,} rows")


# 主要な登記原因のリスト　25個
valid_reasons = [
    "所有権移転売買", "所有権移転相続・法人合併", "抹消登記", 
    "登記名義人の氏名等についての変更・更正", "所有権の保存(申請)", "所有権の保存(職権)", "滅失", 
    "地目変更・更正", "所有権移転遺贈・贈与その他", "分筆", "権利の変更・更正", 
    "根抵当権の設定", "所有権移転その他の原因", "権利の移転(所有権を除く)", 
    "抵当権の設定", "地積変更・更正", "区分建物の表題", "表題", "仮登記(その他)", "処分の制限に関する登記",
    "合筆", "仮登記(所有権)", "共同担保変更通知", "共同担保追加通知", "信託に関する登記"
]

other_dict = {
    "登記名義人の氏名等についての変更・更正": [
        "登記名義人の氏名等",
        '登記名義の人名氏等についての変更更正',
        '登記名義の大名氏についての変更更正',
        '登記名義人のの変更更正',
        '登記名義入の氏名等についての変更更正',
        '登記名義の人名氏についての変更更正',
        '登記名義大の氏名等についての変更更正',
        '登記名義人の変更更',
        
    ],
    
    "所有権移転相続・法人合併": [
        "所有権移転相続法人合併",
        "所有権移転・相続",
        "所有権移転/相続",
        "所有権移転相続法大合併",
        "所有権移転相続法人台併",
        "所有権移転相"
    ],
   
    "所有権移転遺贈・贈与その他": [
        "所有権移転遺贈贈与その他無償名義",
        "所有権移転/贈与",
        "無償名義",
        "遺贈",
        "贈与",
        "死因贈与",
        "所有権移転遺贈"
    ],


    "所有権の保存(申請)": [
        "所有権の保存申請",
        "所有権の保存（申請）",
        '所有権の保存(申讀)',
        '所有権の保存、(申請)',
        '所有権の保存(申',
        '所有権の保存請',
        '所有権の保存(請)',
        '所有権の保存(巾睛)',
        '所有権の保存(巾請)',
        '所有権の保存:(申請)',
        '所所権の保存(申請)',
        '所有権の保存(中請)',
        '所有権の保存中請',
        '所有権の保存_申請',
        '所有権の保存-(申請)',
        
        
        
    ],

    "所有権の保存(職権)": [
        "所有権の保存（職権）",
        "所有権の保存職権",
        '所有権の保存(戦権)'
        '所有権の保存(職權)',
        '所有権の保存',
        '所有権の保',
    
    ],
    
    "権利の移転(所有権を除く)": [
        "権利の移転",
        '権利移転所有権を除く'
        
    ],
    "仮登記(その他)": [
        "仮登記その他",
        "仮登記（その他）",
        "仮登記:(その他)",
        "仮登記:（その他）",
        '仮登記(ぞの他)' ,
        '収登記(その他)',
        
        
    ],
    "仮登記(所有権)": [
        "仮登記",
    ],
    
    "地積変更・更正": [
        "地積変更",
        '地変更更正',
        '地獄変更更正',
        '地震変更更正',
        
    ],

    "地目変更・更正": [
        "地自変更",
        "地目変更",
        '地且変更',
        '地日変更',
        '地月変更更正',
        '地変変更更正',
    ],

    "権利の変更・更正": [
        "権利の変更",
        "権利変更更正",
        'の変更・更正',
        'の変更更正',
        '権利変変更正',



    ],
    
    "信託に関する登記": [
        "託に関する登記"
    ],

    "滅失": [
        "鍼失",
        "失",
        "減滅",
        "誠夫",
        '滅',
        '誠矢',
        
        

        
    ],
    
    "所有権移転その他の原因": [
        "所有権移転/その他",
        '所有権移転ぞの他の原',
        '所有権移転その',
        '所所権移転その他の原因',
        '所有物移転その他の原因',

    ],
    
    
    "所有権移転売買": [
        '所有権移転',

        
        
    ],

    "分筆": [
        "分",
        '台筆',

    ],
    "合筆": [
        '合',
        '含筆',
        
        
        
    ],

    "抹消登記": [
        "抹茶登記",
        "抹登記",
        '消',
        '抹清登記',
        '枋寮登記',
        '末梢登記',

    ],

    "表題": [
        "裏題",
        "夜題",
        "裴題",
        '衷題',
        '表告',
        '喪題',
        '変題',
        '表明',
        '寝題',
        '表麗',
        '表選',
        '襄題',
        '裹題',
        '表',
        '題',
        '衰題',
          
    ],
    
    "根抵当権の設定": [
        '根',
    ],

    "抵当権の設定": [
        '抵抵権権設定',
        '抵抵権の設定',
        '抵抵権権の設定',
        '抵尚権の設定',
        '当当の設定',
        '抵',
        '抵当権の設',
        


    ],
    
    "共同担保変更通知": [
        "共同担保変更道知",

    ]
    
}


# 無視したい "どうでもいい理由"
ignore_reasons = [
    '表示に関するその他', '買戻権', '権利に関するその他', '配偶者居住権の設定', '分割・区分', '地上権の設定', '床面積の変更', '附属建物の新築', '敷地権の表示', '建物所在図訂正', 
    '地役権の設定', '賃借権の設定', '土地改良区画整理', '合併', '合体', '先取特権の保存', '質権の設定', '移記', '敷地権たる旨の登記', '#NAME?', 'nan', '筆界特定に伴う地図番号欄への記載',
    '地図訂正', '採石権の設定', '郵送', '予告登記', '地上欄の設定', '永小作権の設定', '買房権', '飯豊記(所有権)', '地上榴の設定', '衰期', '増役権の設定', '取下' '却下', 
]


# 登記原因の名寄せ
reverse_dict = {}
for norm, variants in other_dict.items():
    for variant in variants:
        reverse_dict[variant] = norm

def map_reason(value):
    for variant in reverse_dict:
        if variant in value:
            return reverse_dict[variant]
    for reason in valid_reasons:
        if reason in value:
            return reason
    return value

print("=== Step 1: Filter valid registration reasons ===")
before = len(data)
data = data[data['reason'].isin(valid_reasons)]
after = len(data)
print(f"Filtered reasons: {before:,} → {after:,}")


# 登記原因のグルーピング
reason_map = {
    '所有権移転売買': 'sale',
    '所有権移転相続・法人合併': "inheritance_or_gift_transfer",
    '所有権移転遺贈・贈与その他': "inheritance_or_gift_transfer",
    '所有権移転その他の原因': "other_causes_transfer",
    '抵当権の設定': 'collateral',
    '根抵当権の設定': 'collateral',
    '仮登記(その他)': 'collateral',
    '仮登記(所有権)': 'collateral',
    '共同担保追加通知': 'collateral',
    '共同担保変更通知': 'collateral',
    '所有権の保存(申請)': 'ownership_origin',
    '所有権の保存(職権)': 'ownership_origin',
    '地目変更・更正': 'physical_change',
    '地積変更・更正': 'physical_change',
    '分筆': 'physical_change',
    '合筆': 'physical_change',
    '滅失': 'physical_change',
    '表題': 'title_registration',
    '処分の制限に関する登記': 'restriction',
    '信託に関する登記': 'restriction',
    '登記名義人の氏名等についての変更・更正': 'title_or_right_correction',
    '権利の変更・更正': 'title_or_right_correction',
    '権利の移転(所有権を除く)': 'title_or_right_correction',
    '抹消登記': 'cancellation',
}

data['reason_group'] = data['reason'].map(reason_map).fillna('other')

# datetime型への変換
import datetime
data['register_date'] = pd.to_datetime(data['register_date'])

# 用途地域を英語に変換
district_translation = {
    "第二種住居地域": "category_ii_residential_district",
    "第二種中高層住居専用地域": "category_ii_mid_high_rise_residential_district",
    "第一種住居地域": "category_i_residential_district",
    "準住居地域": "semi_residential_district",
    "工業地域": "industrial_district",
    "近隣商業地域": "neighborhood_commercial_district",
    "第一種低層住居専用地域": "category_i_low_rise_exclusive_residential_district",
    "第一種中高層住居専用地域": "category_i_mid_high_rise_residential_district",
    "工業専用地域": "exclusively_industrial_district",
    "準工業地域": "quasi_industrial_district",
    "第二種低層住居専用地域": "category_ii_low_rise_exclusive_residential_district",
    "商業地域": "commercial_district"
}
# 用途地域をsnake_caseの英語ラベルに変換（列を上書き）
data['use_district'] = data['use_district'].map(district_translation)

# chiban と land_num を統合した新しい列 parcel_num を作成
data["parcel_num"] = data["chiban"].fillna(data["land_num"])

# NaNの除去
data = data[~data["pref"].isna()]
data = data[~data["city"].isna()]
data = data[~data["location"].isna()]

# "不明" や 空文字列の除去（.any(axis=1) は不要）
data = data[~data["pref"].isin(["不明", ""])]
data = data[~data["city"].isin(["不明", ""])]
data = data[~data["location"].isin(["不明", ""])]

# 住所キー
data["address_group"] = data["pref"] + "_" + data["city"] + "_" + data["location"]

# 地番キー（parcel_num）も使って group_key 作成
data["group_key"] = data["address_group"] + "_" + data["parcel_num"].astype(str)

# 同日の登記はセット．その日あった登記原因の1-hotの算出，直後にあった登記原因の1-hotを算出，直後の登記までの日数も算出する

# 以下土地データのみ
# 土地データを取得
print("=== Step 2: Extract land data ===")
land_data = data[data["type"] == "土地"].copy()
print(f"Land records: {len(land_data):,}")


# 登記回数（その group_key・日付の件数）
land_data['same_day_count'] = land_data.groupby(['group_key', 'register_date'])['reason'].transform('count')

print("=== Step 3: Create group keys ===")
print(f"Unique group_key: {land_data['group_key'].nunique():,}")


print("=== Step 4: Same-day aggregation & one-hot encoding ===")

land_data.sort_values(by=["group_key", "register_date"], inplace=True)
land_data.reset_index(drop=True, inplace=True)

reason_dummies = pd.get_dummies(
    land_data['reason_group'],
    prefix='on_day_reason_group'
)

print(f"Reason group dummy cols: {reason_dummies.shape[1]}")


# one-hot付きデータ
land_data_with_dummies = pd.concat([land_data[['group_key', 'register_date']], reason_dummies], axis=1)

# group_key × 登記日で1レコードに統合（複数の登記原因が同日あればmaxで1）
dummies_grouped = land_data_with_dummies.groupby(['group_key', 'register_date'], as_index=False).max()

# land_dataから代表行（group_key × 登記日で1件）を取得（他の列を残す）
# ここでは先頭の行を代表として抽出
meta_cols = [col for col in land_data.columns if col not in ['reason', 'reason_group', 'register_type', 'register_num']]
meta_grouped = land_data[meta_cols].drop_duplicates(subset=['group_key', 'register_date'])

# 結合：one-hotと他の列を統合
land_data_daily = pd.merge(meta_grouped, dummies_grouped, on=['group_key', 'register_date'], how='left')

# on_day_reason_group_ で始まる列を抽出して int に変換
reason_group_cols = [col for col in land_data_daily.columns if col.startswith('on_day_reason_group_')]
land_data_daily[reason_group_cols] = land_data_daily[reason_group_cols].astype(int)

# ステップ 1: 登記日ソート
land_data_daily.sort_values(by=['group_key', 'register_date'], inplace=True)

# ステップ 2: 次の登記日の取得
land_data_daily['next_date'] = land_data_daily.groupby('group_key')['register_date'].shift(-1)

# ステップ 3: 登記日その日の one-hot を lookup 用に保存
reason_group_cols = [col for col in land_data_daily.columns if col.startswith('on_day_reason_group_')]

lookup_onehot = land_data_daily[['group_key', 'register_date'] + reason_group_cols].copy()
lookup_onehot = lookup_onehot.rename(columns={'register_date': 'next_date'})

# ステップ 4: merge によって次の登記日の one-hot を付与
land_data_daily = pd.merge(
    land_data_daily,
    lookup_onehot,
    on=['group_key', 'next_date'],
    how='left',
    suffixes=('', '_next')
)

print("=== Step 5: Compute next registration info ===")
print(f"Records before next-date merge: {len(land_data_daily):,}")

land_data_daily['days_until_next'] = (
    land_data_daily['next_date'] - land_data_daily['register_date']
).dt.days

print("Next registration computed")

import datetime
land_data_daily['register_date'] = pd.to_datetime(land_data_daily['register_date'])

# 特徴量の作成

# monthを抽出する
land_data_daily['month'] = land_data_daily['register_date'].dt.month
land_data_daily['month_sin'] = np.sin(2 * np.pi * land_data_daily['month'] / 12)

# 各都道府県の人口密度データを辞書形式で用意(令和2年度，最新版)　1km2あたりの人口
prefecture_density = {
    '北海道': 66.6, '青森県': 128.3, '岩手県': 79.2, '宮城県': 316.1, '秋田県': 82.4,
    '山形県': 114.6, '福島県': 133.0, '茨城県': 470.2, '栃木県': 301.7, '群馬県': 304.8,
    '埼玉県': 1934.0, '千葉県': 1218.5, '東京都': 6402.6, '神奈川県': 3823.2, '新潟県': 174.9,
    '富山県': 243.6, '石川県': 270.5, '福井県': 183.0, '山梨県': 181.4, '長野県': 151.0,
    '岐阜県': 186.3, '静岡県': 467.2, '愛知県': 1458.0, '三重県': 306.6, '滋賀県': 351.9,
    '京都府': 559.0, '大阪府': 4638.4, '兵庫県': 650.5, '奈良県': 358.8, '和歌山県': 195.3,
    '鳥取県': 157.8, '島根県': 100.1, '岡山県': 265.4, '広島県': 330.2, '山口県': 219.6,
    '徳島県': 173.5, '香川県': 506.3, '愛媛県': 235.2, '高知県': 97.3, '福岡県': 1029.8,
    '佐賀県': 332.5, '長崎県': 317.7, '熊本県': 234.6, '大分県': 177.2, '宮崎県': 138.3,
    '鹿児島県': 172.9, '沖縄県': 642.9
}

# DataFrameに人口密度カラムを追加
land_data_daily['population_density'] = land_data_daily['pref'].map(prefecture_density)

# 用途地域をダミー変数に変換
land_data_daily = pd.get_dummies(land_data_daily, columns=['use_district'], prefix='dummy', drop_first=True, dtype=float)

# 登記日が複数ある物件のうち、最後の登記日であるレコードを削除する
# 各 group_key ごとの登記日数をカウント → 登記日数フラグ列を作成（再登記される場合は0）
date_counts = land_data_daily.groupby('group_key')['register_date'].transform('nunique')
land_data_daily['will_not_be_re_registered'] = (date_counts <= 1).astype(int)

# 最終登記日を取得（すべての group_key 対象でOK）
last_date = land_data_daily.groupby('group_key')['register_date'].transform('max')

# フィルタ適用：再登記がある場合は最終日を除く、再登記がないものはそのまま残す
land_data_daily = land_data_daily[
    (land_data_daily['will_not_be_re_registered'] == 1) |
    (land_data_daily['register_date'] < last_date)
]

print("=== Step 6: Handle non-re-registered properties ===")

cnt_not = land_data_daily['will_not_be_re_registered'].sum()
cnt_all = len(land_data_daily)

print(f"Will-not-be-re-registered: {cnt_not:,} / {cnt_all:,}")


# 特徴量と目的変数のカラム名を定義（_nextを除外）
feature_cols = (
    ['month_sin', 'same_day_count', 'size', 'official_price', 'population_density',
     'building_coverage_ratio', 'floor_area_ratio', 'on_foot']
    + [col for col in land_data_daily.columns
       if col.startswith('on_day_reason_group_') and not col.endswith('_next')]
    + [col for col in land_data_daily.columns if col.startswith('dummy_')]
)

target_cols = [col for col in land_data_daily.columns
               if col.startswith('on_day_reason_group_') and col.endswith('_next')] + ['days_until_next']


# 特徴量の欠損を除去（目的変数には影響しない）
filtered_data = land_data_daily.dropna(subset=feature_cols)



# 登記間隔日数予測の目的変数の作成
# 閾値を1,4,9,24,120カ月とする
def categorize_period(days):
    # 再登記されないデータはそれだけでカテゴリmaxにする
    if pd.isnull(days):
        return 5
    elif days <= 31:
        return 0
    elif days <= 31 * 4:
        return 1
    elif days <= 31 * 9:
        return 2
    elif days <= 31 * 24:
        return 3
    else:
        return 4
    

# ====== 1. filtered_dataにsales1_categoryを追加 ======
print("=== Step 7: Categorize days_until_next ===")

filtered_data['days_until_next_category'] = (
    filtered_data['days_until_next']
    .progress_apply(categorize_period)
)


# 再登記されなかったデータの日数を31*120で埋める
filtered_data['days_until_next'] = filtered_data['days_until_next'].fillna(31 * 120)

# 再登記されなかったデータの登記原因_nextを0で埋める
# 対象の列名リストを取得
reason_next_cols = [
    col for col in filtered_data.columns
    if col.startswith('on_day_reason_group_') and col.endswith('_next')
]

# 条件に合う行に対して 0.0 を代入
filtered_data.loc[filtered_data['will_not_be_re_registered'] == 1, reason_next_cols] = 0.0

# 列を特徴量と目的変数だけにする
feature_cols = (
    ['month_sin', 'same_day_count', 'size', 'official_price', 'population_density',
     'building_coverage_ratio', 'floor_area_ratio', 'on_foot']
    + [col for col in land_data_daily.columns
       if col.startswith('on_day_reason_group_') and not col.endswith('_next')]
    + [col for col in land_data_daily.columns if col.startswith('dummy_')]
)

target_cols = ['will_not_be_re_registered'] + [col for col in land_data_daily.columns
               if col.startswith('on_day_reason_group_') and col.endswith('_next')] + ['days_until_next_category'] + ['days_until_next']

land_data_for_prediction = filtered_data[feature_cols + target_cols]

feature_cols = (
    ['month_sin', 'same_day_count', 'size', 'official_price', 'population_density',
     'building_coverage_ratio', 'floor_area_ratio', 'on_foot']
    + [col for col in land_data_daily.columns
       if col.startswith('on_day_reason_group_') and not col.endswith('_next')]
    + [col for col in land_data_daily.columns if col.startswith('dummy_')]
)

target_cols = ['will_not_be_re_registered'] + [col for col in land_data_daily.columns
               if col.startswith('on_day_reason_group_') and col.endswith('_next')] + ['days_until_next_category'] + ['days_until_next']

land_data_for_prediction = filtered_data[feature_cols + target_cols]

print("=== Step 8: Final dataset ===")
print(f"Feature columns: {len(feature_cols)}")
print(f"Target columns: {len(target_cols)}")
print(f"Final rows: {len(land_data_for_prediction):,}")


print("=== Step 9: Save preprocessed data ===")

out_dir = Path("data") / "after_preprocess"
out_dir.mkdir(parents=True, exist_ok=True)

out_path = out_dir / "land_data_for_prediction.csv"
land_data_for_prediction.to_csv(out_path, index=False)

print(f"Saved to: {out_path.resolve()}")
