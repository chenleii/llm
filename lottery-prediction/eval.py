import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision import datasets
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import fc3d_model_v1


class Fc3DDataset(Dataset):
    def __init__(self):
        df = pd.read_csv('./data/fc3d.csv',dtype={'num':str,'nos':str})
        self._features = df['num'].values
        self._labels = df['nos'].values

    @property
    def features(self):
        return self._features

    @property
    def labels(self):
        return self._labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        feature = fc3d_model_v1.value_to_list(feature)
        label = fc3d_model_v1.value_to_list(label)
        return torch.tensor(feature), torch.tensor(label)


if __name__ == '__main__':
    model = fc3d_model_v1.Model()
    model.load_state_dict(torch.load("cc.pth"))
    dataset = Fc3DDataset()
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # 评估模型
    all_preds = []
    all_labels = []
    # 评估模型
    correct = 0
    rough_correct = 0
    random_correct = 0
    random_rough_correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(dataloader):
            output = model(data)
            output = F.softmax(input=output, dim=-1)
            # 概率最大的
            predict = output.argmax(dim=-1)
            # 完全随机预测
            random_predict = torch.randint(low=0, high=10, size=label.shape)

            # predict = predict.view(-1)
            # label = label.view(-1)
            # all_preds.extend(predict.cpu().numpy())
            # all_labels.extend(label.cpu().numpy())

            correct += (predict == label).all(dim=-1).sum().item()
            rough_correct += (predict.sort(dim=-1)[0] == label.sort(dim=-1)[0]).all(dim=-1).sum().item()
            random_correct += (random_predict == label).all(dim=-1).sum().item()
            random_rough_correct += (random_predict.sort(dim=-1)[0] == label.sort(dim=-1)[0]).all(dim=-1).sum().item()
            total += label.size(0)

        # 计算评估指标
        # accuracy = accuracy_score(all_labels, all_preds)
        # precision = precision_score(all_labels, all_preds, average='binary')
        # recall = recall_score(all_labels, all_preds, average='binary')
        # f1 = f1_score(all_labels, all_preds, average='binary')

        # print(f'Accuracy: {accuracy:.4f}')
        # print(f'Precision: {precision:.4f}')
        # print(f'Recall: {recall:.4f}')
        # print(f'F1 Score: {f1:.4f}')

        # 计算准确率
        accuracy = correct / total
        rough_accuracy = rough_correct / total
        random_accuracy = random_correct / total
        random_rough_accuracy = random_rough_correct / total
        print(f'Accuracy: {accuracy:.4f}')
        print(f'Rough Accuracy: {rough_accuracy:.4f}')
        print(f'Random Accuracy: {random_accuracy:.4f}')
        print(f'Random Rough Accuracy: {random_rough_accuracy:.4f}')