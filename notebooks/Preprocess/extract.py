import pandas as pd
file_path = r"D:\fujiwara\M\data\after_preprocess\2022-2023_受付帳_名寄せ_登記原因_所有権移転売買.csv"
df_reg26 = pd.read_csv(file_path)

fraction = 1 / 200
# 指定された割合でランダムにサンプリング
sampled_df = df_reg26.sample(frac=fraction)

# サンプルデータを新しいCSVファイルとして保存
sampled_df.to_csv(r"D:\fujiwara\M\data\after_preprocess\2022-2023_受付帳_名寄せ_登記原因_所有権移転売買_200分の1.csv", index=False)