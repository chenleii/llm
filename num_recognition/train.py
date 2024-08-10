import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets

import model

if __name__ == '__main__':
    transforms_compose = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor(), ])

    train_datasets = datasets.ImageFolder(root='./mnist_train', transform=transforms_compose)
    test_datasets = datasets.ImageFolder(root='./mnist_test', transform=transforms_compose)

    print("train_datasets len:", len(train_datasets))
    print("test_datasets len:", len(test_datasets))

    train_dataloader = DataLoader(train_datasets, batch_size=64, shuffle=True)
    print("train_dataloader len:", len(train_dataloader))

    for batch_idx, (data, label) in enumerate(train_dataloader):
        if batch_idx == 1:
            break
        print("batch_idx:", batch_idx)
        print("data:", data.shape)
        print("label:", label.shape)

    model = model.Model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(10):
        for batch_idx, (data, label) in enumerate(train_dataloader):
            output = model(data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch_idx % 100 == 0:
                print(f"epoch: {epoch}/10, batch_idx: {batch_idx}/{len(train_dataloader)}, loss: {loss.item()}")

    torch.save(model.state_dict(), './cc.pth')

