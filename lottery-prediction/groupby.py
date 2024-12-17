import numpy as np
import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv('./data/fc3d.csv')

    # 分组 十个一组
    grouped_data = df.groupby(np.arange(len(df)) // 10)

    # 打印结果
    for i, (_, group) in enumerate(grouped_data):
        print(f"Group {i + 1}:\n{group}\n")
