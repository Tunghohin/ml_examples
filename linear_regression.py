import numpy as np
import torch
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from torch.utils import data
from torch import nn

def load_data(data_input, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_input)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    dataset = pd.read_csv(os.path.join(".", "dataset", "1000_Companies.csv"))
    dataset.drop("State", axis=1, inplace=True)
    dataset_norm = (dataset - dataset.mean()) / dataset.std()

    inputs, outputs = torch.from_numpy(dataset_norm.iloc[:, :-1].values), torch.from_numpy(dataset_norm.iloc[:, -1].values).reshape(1000, 1)

    features_train, features_test, labels_train, labels_test = train_test_split(inputs, outputs, test_size=0.2, random_state=0)

    batch_size = 20
    data_iter = load_data((features_train, labels_train), batch_size, True)
    net = nn.Sequential(nn.Linear(features_test.shape[1], 1))
    loss = nn.MSELoss()
    trainer = torch.optim.SGD(net.parameters(), lr=0.01)

    num_epochs = 20
    for epoch in range(num_epochs):
        for X, y in data_iter:
            l = loss(net(X) ,y)
            trainer.zero_grad()
            l.backward()
            trainer.step()
        l = loss(net(features_test), labels_test)
        print(f'epoch {epoch + 1}, loss {l:f}')