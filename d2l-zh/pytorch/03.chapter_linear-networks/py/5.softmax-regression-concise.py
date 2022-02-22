# -*- coding: utf-8 -*-
import torch
from torch import nn
from IPython import display
from d2l import torch as d2l
import os

from  matplotlib import pyplot as plt


if __name__ == "__main__":
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, root="../../data")
    # print(len(train_iter))
    # print(len(test_iter))
    # print(os.getcwd())

    # 初始化模型参数
    # PyTorch不会隐式地调整输入的形状。因此，
    # 我们在线性层前定义了展平层（flatten），来调整网络输入的形状
    net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.01)

    net.apply(init_weights)

    loss = nn.CrossEntropyLoss()

    # 优化算法
    trainer = torch.optim.SGD(net.parameters(), lr=0.1)

    # 训练
    num_epochs = 10
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
