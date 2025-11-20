import pandas as pd
from tqdm import tqdm
tqdm.pandas()  # tqdmをpandasで使用する設定

# CSVデータを読み込む
file_path = "./data/登記受付帳_築年数追加.csv"
data = pd.read_csv(file_path)

# 主要な登記原因のリスト
valid_reasons = [
    "所有権移転売買", "所有権移転相続・法人合併", "抹消登記", "抵当権の設定", 
    "登記名義人の氏名等についての変更・更正", "所有権の保存(申請)", "表題", "滅失", 
    "地目変更・更正", "所有権移転遺贈・贈与その他無償名義", "分筆", "権利", 
    "変更・更正", "根抵当権の設定", "所有権移転その他の原因", "権利の移転(所有権を除く)", 
    "地積変更・更正", "区分建物の表題", "仮登記(その他)", "処分の制限に関する登記", 
    "合筆", "仮登記(所有権)", "所有権移転遺贈・贈与その他", "共同担保変更通知", 
    "抹消登記/嘱託", "処分の制限に関する登記/嘱託"
]

other_dict = {"登記名義人の氏名等についての変更更正": "登記名義人の氏名等についての変更・更正", "所有権移転相続法人合併": "所有権移転相続・法人合併",
               "地目変更更正": "地目変更・更正", "所有権移転遺贈贈与その他無償名義": "所有権移転遺贈・贈与その他無償名義", "抹消登記嘱託": "抹消登記/嘱託",
                 "処分の制限に関する登記嘱託": "処分の制限に関する登記/嘱託", "所有権の保存申請": "所有権の保存(申請)", "権利の移転所有権を除く": "権利の移転(所有権を除く)", "仮登記その他": "仮登記(その他)",
                 "仮登記所有権": "仮登記(所有権)", "地積変更更正": "地積変更・更正"}

# `reason`列を主要な登記原因で置き換える処理
def map_reason(value):
    for pre_reason in other_dict:
        if pre_reason in value:
            return other_dict[pre_reason]
        
    for reason in valid_reasons:
        if reason in value:
            return reason
    return value  # その他の場合は元の値をそのまま保持

# tqdmを使って途中経過を表示しながら適用
data['reason'] = data['reason'].astype(str).progress_apply(map_reason)

# 前処理されたデータを保存する場合
data.to_csv("./data/登記受付帳_築年数追加_前処理済み.csv", index=False)

# 結果確認用
# print(data['reason'].value_counts())
