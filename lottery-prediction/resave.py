from datetime import datetime, timedelta

import pandas as pd

import fc3d_model_v1

df = pd.read_excel('./data/fc3dh.xlsx')
# 拆分出每个中奖号码
df[['no1', 'no2', 'no3']] = df['jianghaoStr'].str.split(' ', expand=True)

def num_to_date(num):
    num = str(num)
    year = int(num[:4])
    days = int(num[-3:])
    date = datetime(year, 1, 1)
    date = date + timedelta(days=(days-1))
    return date.strftime('%Y%m%d')
df['date'] = df['qihao'].apply(num_to_date)


# print(df)
# df.drop(columns=['jianghaoStr'], inplace=True)
df['nos'] = df['jianghaoStr'].str.replace(' ', '')
df['num'] = df['qihao']

def isnnn(v):
    return len(fc3d_model_v1.value_to_list(v)) == 2


df['err'] = df["nos"].apply(isnnn)
# err = df[df["err"]]
# print(err)
# 保留列
df = df[['num', 'date', 'nos','no1', 'no2', 'no3','err']]
df.to_csv("./data/fc3d.csv", index=False)