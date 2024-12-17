import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np

if __name__ == __name__:
    # 创建一个新的图像
    fig, ax = plt.subplots(figsize=(5, 5))

    # 生成100个在0到1之间的随机数
    x = np.random.rand(100)*2
    y = np.random.rand(100)*2
    # 在图像中随机撒下100个点
    ax.scatter(x, y,c="r", cmap='viridis')

    # 创建一个正方形，左下角位于 (0.2, 0.2)，宽度和高度为 0.6
    square = Rectangle((0, 0), 2, 2,facecolor='blue', edgecolor='black', linewidth=0)

    # 创建一个圆形，其中心位于 (0.5, 0.5)，半径为 0.2
    circle = Circle((1, 1), 1, facecolor='red', edgecolor='black', linewidth=0)

    # 将正方形和圆形添加到图像中
    # ax.add_patch(square)
    # ax.add_patch(circle)

    # 设置图像的x轴和y轴范围
    ax.set_xticks([0, 2])
    # ax.set_ylim(0, 2)
    # 设置axes的纵横比为1:1
    ax.set_aspect('equal')


    # 显示图像
    plt.show()
