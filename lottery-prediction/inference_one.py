import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

import fc3d_model_v1

# 预测一次
if __name__ == '__main__':
    model = fc3d_model_v1.Model()
    model.load_state_dict(torch.load("cc.pth"))
    model.eval()

    with torch.no_grad():
        data = torch.tensor(fc3d_model_v1.value_to_list("2024329")).unsqueeze(0)
        output = model(data)
        output = output.squeeze(0)
        # output = F.layer_norm(output, normalized_shape=(output.size(-2), output.size(-1)))
        output = F.softmax(input=output, dim=-1)
        # 概率最大的
        predict = output.argmax(dim=1)

        print(predict)
        figure = plt.figure()
        layout = """A
                    B
                    C"""
        # layout = [['A'],['B'],['C']]
        y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        axes_mosaic = figure.subplot_mosaic(layout)
        axes_mosaic['A'].bar(y, output[0], color='skyblue')
        axes_mosaic['A'].set_title(f'predict:{predict.numpy()} \n\nSingle digits')
        axes_mosaic['B'].bar(y, output[1], color='skyblue')
        axes_mosaic['B'].set_title('Tens digits')
        axes_mosaic['C'].bar(y, output[2], color='skyblue')
        axes_mosaic['C'].set_title('Hundreds digit')

        # plt.tight_layout()
        plt.subplots_adjust(hspace=0.6)
        plt.show()
