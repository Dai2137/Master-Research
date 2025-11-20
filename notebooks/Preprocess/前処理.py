import pandas as pd
from tqdm import tqdm
import re
tqdm.pandas()  # tqdmをpandasで使用する設定

# CSVデータを読み込む
file_path = ".\data\登記受付帳_22-23_築年数追加.csv"
data = pd.read_csv(file_path)

# 登記原因の名寄せ

# 主要な登記原因のリスト　27個
valid_reasons = [
    "所有権移転売買", "所有権移転相続・法人合併", "抹消登記", "抵当権の設定", 
    "登記名義人の氏名等についての変更・更正", "所有権の保存(申請)", "表題", "滅失", 
    "地目変更・更正", "所有権移転遺贈・贈与その他無償名義", "分筆", "権利の変更・更正", 
    "根抵当権の設定", "所有権移転その他の原因", "権利の移転(所有権を除く)", 
    "地積変更・更正", "区分建物の表題", "仮登記(その他)", "処分の制限に関する登記",
    "合筆", "仮登記(所有権)", "共同担保変更通知", "共同担保追加通知", "所有権移転遺贈・贈与その他", 
    "抹消登記/嘱託", "処分の制限に関する登記/嘱託", "信託に関する登記"
]

other_dict = {
    "登記名義人の氏名等についての変更更正": "登記名義人の氏名等についての変更・更正", "所有権移転相続法人合併": "所有権移転相続・法人合併",
    "地目変更更正": "地目変更・更正", "所有権移転遺贈贈与その他無償名義": "所有権移転遺贈・贈与その他無償名義", "抹消登記嘱託": "抹消登記/嘱託",
    "処分の制限に関する登記嘱託": "処分の制限に関する登記/嘱託", "所有権の保存申請": "所有権の保存(申請)", "権利の移転所有権を除く": "権利の移転(所有権を除く)", "仮登記その他": "仮登記(その他)",
    "仮登記所有権": "仮登記(所有権)", "地積変更更正": "地積変更・更正", "権利の変更更正": "権利の変更・更正"
}


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


# 住所の名寄せ

HYPHEN: str = "-"
def replace_hyphen_like_characters_after_digits(addr: str) -> str:
    """
    数字の後にあるハイフンのような文字をハイフンに置換する
    """
    if pd.isna(addr):  # NaN値のチェックを追加
        return addr
        
    hyphen_iter = re.finditer(
        "([0-9０-９一二三四五六七八九〇十百千][-－﹣−‐⁃‑‒–—﹘―⎯⏤ーｰ─━])|([-－﹣−‐⁃‑‒–—﹘―⎯⏤ーｰ─━])[0-9０-９一二三四五六七八九〇十]",
        str(addr),  # 文字列型に変換
    )
    
    for m in hyphen_iter:
        from_value = m.group()
        replace_value = re.sub("[-－﹣−‐⁃‑‒–—﹘―⎯⏤ーｰ─━]", HYPHEN, from_value)
        addr = addr.replace(from_value, replace_value)
    return addr

# データフレームに適用
from tqdm import tqdm
tqdm.pandas(desc="Replacing hyphen-like characters")
data_all['chiban'] = data_all['chiban'].progress_apply(replace_hyphen_like_characters_after_digits)
data_all['land_num'] = data_all['land_num'].progress_apply(replace_hyphen_like_characters_after_digits)




# location内にある半角スペースと全角スペースを消去
def remove_spaces_from_location(addr: str) -> str:
    """
    location内にある半角スペースを消去する
    """
    if pd.isna(addr):  # NaN値のチェックを追加
        return addr
    
    return addr.replace(' ', '')

# データフレームに適用
data_all['location'] = data_all['location'].apply(remove_spaces_from_location)



# locationに全角のアラビア数字がある場合，半角に変換
def convert_fullwidth_to_halfwidth(addr: str) -> str:
    """
    location内にある全角のアラビア数字を半角に変換する
    """
    if pd.isna(addr):  # NaN値のチェックを追加
        return addr
    
    return addr.translate(str.maketrans('０１２３４５６７８９', '0123456789'))

# データフレームに適用
data_all['location'] = data_all['location'].apply(convert_fullwidth_to_halfwidth)

# # land_num列のハイフンを全角から半角に変換する処理を追加
# data['land_num'] = data['land_num'].astype(str).str.replace('－', '-', regex=False)

# # 漢数字をアラビア数字に変換する辞書
# kanji_to_num = {
#     '一': '1', '二': '2', '三': '3', '四': '4', '五': '5', 
#     '六': '6', '七': '7', '八': '8', '九': '9', '十': '10'
# }

# # 漢数字をアラビア数字に変換する関数
# def convert_kanji_to_number(location):
#     if pd.isnull(location):
#         return location  # NaNはそのまま返す

#     # 丁目の直前にある漢数字を変換
#     def replace_kanji(match):
#         kanji = match.group(1)  # キャプチャグループ1を取得
#         return kanji_to_num.get(kanji, kanji) + '丁目'  # 漢数字をアラビア数字に変換して返す

#     # 正規表現で "漢数字+丁目" を探し出して変換
#     return re.sub(r'([一二三四五六七八九十])丁目', replace_kanji, location)

# # location列に適用
# data['location'] = data['location'].apply(convert_kanji_to_number)



# # 地番部分を抽出して、丁目・番地（番も含む）・号も処理する関数
# def extract_land_num(location):
#     match = re.search(
#         r'(\d+)丁目(\d+[-－]\d+(?:[-－]\d+)?)(?:番地|番)?(?:号)?|'
#         r'(\d+)丁目(\d+)(?:番地|番)(\d+)号|'
#         r'(\d+)丁目(\d+)(?:番地|番)|'
#         r'(\d+)丁目|'
#         r'(\d+[-－]\d+(?:[-－]\d+)?)$', 
#         str(location)
#     )
#     if match:
#         if match.group(1) and match.group(2):  # "1丁目326-402" のような形式
#             return f"{match.group(1)}-{match.group(2)}"
#         elif match.group(3) and match.group(4) and match.group(5):  # "4丁目20番地8号" or "4丁目20番8号"
#             return f"{match.group(3)}-{match.group(4)}-{match.group(5)}"
#         elif match.group(6) and match.group(7):  # "4丁目20番地" or "4丁目20番"
#             return f"{match.group(6)}-{match.group(7)}"
#         elif match.group(8):  # "3丁目" のみ
#             return f"{match.group(8)}"
#         elif match.group(9):  # "30-1" のような形式
#             return match.group(9)
#     return None

# # land_numに移動し、locationから削除
# data['land_num_update'] = data['location'].apply(extract_land_num)
# data['land_num'] = data['land_num'].fillna(data['land_num_update'])

# # location列から地番部分を削除（番地、番、号、〇丁目も含めて削除）
# data['location'] = data['location'].apply(
#     lambda x: re.sub(
#         r'(\d+)丁目(\d+[-－]\d+(?:[-－]\d+)?)(?:番地|番)?(?:号)?|'
#         r'(\d+)丁目(\d+)(?:番地|番)(\d+)号|'
#         r'(\d+)丁目(\d+)(?:番地|番)|'
#         r'(\d+)丁目|'
#         r'(\d+[-－]\d+(?:[-－]\d+)?)$', 
#         '', 
#         str(x)
#     ).strip()
# )

# # # land_num_update列は確認用なので削除
# data.drop(columns=['land_num_update'], inplace=True)



# register_date列をdatetime型に変換
data['register_date'] = pd.to_datetime(data['register_date'], errors='coerce')

# register_dateでソート
data = data.sort_values(by='register_date')

# 

# 同一住所，同一日時，同一登記原因の登記を重複除去

# 前処理されたデータを保存する場合
data.to_csv(".\data\登記受付帳_22-23_築年数追加_前処理済み.csv", index=False)

# 結果確認用
# print(data['reason'].value_counts())
