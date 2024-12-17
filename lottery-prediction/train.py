import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

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
    dataset = Fc3DDataset()
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(1000):
        for batch_idx, (data, label) in enumerate(dataloader):
            output = model(data)
            output = output.view(-1,10)
            label = label.view(-1)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch_idx % 100 == 0:
                print(f"epoch: {epoch}/100, batch_idx: {batch_idx}/{len(dataloader)}, loss: {loss.item()}")

    torch.save(model.state_dict(), './cc.pth')

