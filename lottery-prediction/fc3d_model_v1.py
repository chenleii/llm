from collections.abc import Iterable

import torch
from torch import nn


def value_to_list(input_value):
    # 定义一个辅助函数，用于处理单个输入
    def single_value_to_list(value):
        if not isinstance(value, str):
            value = str(value)
        return [int(digit) for digit in value]

    # 如果输入是列表，则对列表中的每个元素进行处理
    if isinstance(input_value, Iterable) and not isinstance(input_value, (str, bytes)):
        return [single_value_to_list(item) for item in input_value]
    else:
        return single_value_to_list(input_value)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=10, embedding_dim=512)
        self.linear1 = nn.Linear(512, 1024)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(1024, 30)

    def forward(self, x):
        x = self.embedding(x)  # (7,64)
        x = self.linear1(x)  # (7,512)
        x = self.activation(x)  # (7,512)
        x = self.linear2(x)  # (7,30)
        x = torch.mean(x, dim=-2)  # (1,30)
        x = x.squeeze(-2)  # (30,)
        x = x.view(-1, 3, 10)  # (3,10)
        return x
