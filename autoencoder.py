#!/usr/bin/env python3

import pandas as pd

import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim


class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Autoencoder, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Threshold(1e-5, 0),
            nn.ReLU()
        )
        self.output = nn.Sequential(
            nn.Linear(hidden_size, input_size)
            )

    def forward(self, x):
        x = self.layer1(x)
        x = self.output(x)
        return x


if __name__ == "__main__":
    print("# Loading data...", end=" ", flush=True)
    X = pd.read_csv("data/train.data.csv")
    y = pd.read_csv("data/train.labels.csv")
    test_data = pd.read_csv("data/test.data.csv")
    X = X.values
    y = y.values.ravel()
    test_data = test_data.values
    print("done.")

    trainset = TensorDataset(torch.Tensor(X), torch.Tensor(X))
    trainloader = DataLoader(trainset, batch_size=256,
                             shuffle=True, num_workers=2)

    net = Autoencoder(X.shape[1], 32)
    criterion = nn.MSELoss()
    optimizer = optim.Adadelta(net.parameters(),
                               lr=1.0, rho=0.95, weight_decay=1e-5)

    for epoch in range(50):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]
            if i % 10 == 9:
                print(f"[{epoch:3},{i+1:3}] Loss: {running_loss/10:.3f}")
                running_loss = 0.0

    print(net.layer1(Variable(torch.Tensor(X))))
