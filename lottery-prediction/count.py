import matplotlib.pyplot as plt
import pandas as pd

# df = pd.read_excel('./data/fc3dh.xlsx',nrows=500)
df = pd.read_excel('./data/fc3dh.xlsx')
# 拆分出每个中奖号码
df[['no1', 'no2', 'no3']] = df['jianghaoStr'].str.split(' ', expand=True)

no1_counts = df['no1'].value_counts()

# 计算总号码数
no1_total = no1_counts.values.sum()
# 计算每个号码的占比
no1_percentages = no1_counts.values / no1_total
# 将结果转换为 DataFrame
no_df = pd.DataFrame({
    'No1': no1_counts.index,
    'Count': no1_counts.values,
    'Percentage': no1_percentages
})
# 按照号码排序
no_df = no_df.sort_values(by='No1', ascending=True).reset_index(drop=True)

# 可视化每个号码占比
plt.figure(figsize=(10, 6))
plt.bar(no_df['No1'], no_df['Percentage'], color='skyblue')
plt.title('每个号码占比')
plt.xlabel('号码')
plt.ylabel('比例')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()