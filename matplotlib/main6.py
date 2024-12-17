import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# 创建网格数据
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
u, v = np.meshgrid(u, v)
x = 10 * np.cos(u) * np.sin(v)
y = 10 * np.sin(u) * np.sin(v)
z = 10 * np.cos(v)

# 创建一个3D的figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# ax = fig.add_subplot(111)

# 使用 plot_surface() 方法
ax.plot_surface(x, y, z, color='r')
# ax.plot(x,y)
ax.set_aspect('equal')
plt.show()