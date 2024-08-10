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
    model.load_state_dict(torch.load("cc.pth"))
    model.eval()

    right_total = 0
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(test_datasets):
            output = model(data)
            predict = output.argmax(dim=1).item()
            if predict == label:
                right_total += 1
            else:
                print(f"wrong case: predict {predict}, label {label}, path:{test_datasets.samples[batch_idx][0]}")

    test_sample_total = len(test_datasets)
    accuracy = right_total / test_sample_total
    print("test accuracy = %d / %d = %.3f" % (right_total, test_sample_total, accuracy))

