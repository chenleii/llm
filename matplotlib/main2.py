from itertools import combinations, product

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib

# matplotlib.use('TkAgg')

# 创建一个3D坐标轴
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')


# 创建一个3D正方体的数据点
r = [-1, 1]
for s, e in combinations(np.array(list(product(r, r, r))), 2):
    if np.sum(np.abs(s-e)) == r[1]-r[0]:
        ax.plot3D(*zip(s, e), color="b")

# 设置坐标轴范围
ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
ax.set_zlim([-2, 2])

# 显示图形
plt.show()