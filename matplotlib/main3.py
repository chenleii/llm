from itertools import combinations, product

import numpy as np
import matplotlib.pyplot as plt
import torch
from mpl_toolkits.mplot3d import Axes3D
import matplotlib

# 创建一个全0的28x28矩阵
image = torch.zeros(28, 28)
image[2:5, 5:23] = 255
image[5:26, 20:23] = 255
image[23:26, 5:23] = 255
image[12:15, 5:23] = 255

# 显示或保存图像
# 注意：这里使用了matplotlib来显示图像，确保已经安装了matplotlib
import matplotlib.pyplot as plt

plt.figure(figsize=(28,28))
plt.imshow(image, cmap='gray')
plt.axis('off')
plt.show()
