import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import matplotlib.axes as axes
import torch
import random
import os

def linear_regression(X, w, b):
    return torch.matmul(X, w) + b

def squared_loss(y_predict, y):
    return (y_predict - y.reshape(y_predict.shape)) ** 2 / 2

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices, :], labels[batch_indices]

def sgd(params, learning_rate, batch_size):
    with torch.no_grad():
        for param in params:
            param -= learning_rate * param.grad / batch_size
            param.grad.zero_()

if __name__ == "__main__":
    dataset = pd.read_csv(os.path.join(".", "dataset", "1000_Companies.csv"))
    dataset.drop("State", axis=1, inplace=True)
    dataset_norm = (dataset - dataset.mean()) / dataset.std()

    inputs, outputs = dataset_norm.iloc[:, :-1], dataset_norm.iloc[:, -1]

    features_train, features_test, labels_train, labels_test = train_test_split(inputs, outputs, test_size=0.2, random_state=0)

    w = torch.normal(0, 0.01, size=(3, ), requires_grad=True, dtype=torch.double)
    b = torch.zeros(1, requires_grad=True)
    batch_size = 80
    lr = 0.01
    num_epoch = 20
    net = linear_regression
    loss = squared_loss

    for i in range(num_epoch):
        for X, y in data_iter(batch_size, torch.from_numpy(features_train.values), torch.from_numpy(labels_train.values)):
            l = loss(net(X, w, b), y)
            l.sum().backward()
            sgd([w, b], lr, batch_size)
        train_l = loss(net(torch.from_numpy(features_test.values), w, b), torch.from_numpy(labels_test.values))
        print(f"batch {i + 1}, lost: {train_l.sum():f}")
        predict_test = net(torch.from_numpy(features_test.values), w, b).detach().numpy()
        print("R2 score: ", r2_score(labels_test.values, predict_test))


