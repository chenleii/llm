import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import torch.nn.functional as F
import matplotlib.pyplot as plt

import model

if __name__ == '__main__':
    # transforms_compose = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor(), ])
    # img = Image.open("./mnist_test/2/mnist_test_1.png")
    # img = transforms_compose(img)[0]

    # 数字3形状
    img = torch.zeros(28, 28)
    img[2:5, 5:23] = 255
    img[5:26, 20:23] = 255
    img[23:26, 5:23] = 255
    img[12:15, 5:23] = 255

    model = model.Model()
    model.load_state_dict(torch.load("cc.pth"))
    model.eval()

    with torch.no_grad():
        output = model(img)
        output = F.layer_norm(output, normalized_shape=(output.size(-2), output.size(-1)))
        output = F.softmax(input=output, dim=-1)
        # 随机，偏向概率大的。
        random = torch.multinomial(input=output, num_samples=1)[0][0].item()
        # 概率最大的
        predict = output.argmax(dim=1).item()

        figure = plt.figure()
        # layout = """A
        #             B"""
        layout = [['A'],['B']]
        axes_mosaic = figure.subplot_mosaic(layout)
        axes_mosaic['A'].imshow(img.cpu().numpy(), cmap='gray')
        axes_mosaic['A'].axis('off')
        axes_mosaic['A'].set_title(f"recognition randomMax:{random}, predictMax:{predict}")
        axes_mosaic['B'].plot(output.squeeze(0) .cpu().numpy(),'r--')
        axes_mosaic['B'].set_xlim(0, 9)
        # axes_mosaic['B'].set_xticks(torch.range(0,9,1).cpu().numpy())
        axes_mosaic['B'].set_title(f"recognition probability distribution")
        plt.show()
