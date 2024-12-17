import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

X = np.arange(-4, 4, 0.25)
Y = np.arange(-4, 4, 0.25)
X, Y, = np.meshgrid(X, Y)
R = np.sqrt(X ** 2 + Y ** 2)
Z = np.sin(R)

fig = plt.figure()

# 使用 subplot_mosaic 创建布局，其中 'right' 是 3D axes
layout = [
    ['top', 'top'],
    ['left', 'right'],
]
mosaic = fig.subplot_mosaic(layout, per_subplot_kw={('left', 'right'): {'projection': '3d'}},
                            gridspec_kw={'width_ratios': [1, 1],
                                         'wspace': 0.01, 'hspace': 0.01}, )

# 绘制其他 2D 图
mosaic['top'].plot(X, Y)
mosaic['left'].scatter(X, Y,Z)
# 绘制 3D 图
mosaic['right'].scatter(X, Y, Z)

plt.show()

