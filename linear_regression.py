import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import matplotlib.axes as axes
import torch
import os

def linear_regression(X, w, b):
    return torch.matmul(X, w) + b

def loss_function(y_predict, y):
    return (y_predict - y.reshape(y_predict.shape)) ** 2 / 2

if __name__ == "__main__":
    dataset = pd.read_csv(os.path.join(".", "dataset", "1000_Companies.csv"))
    dataset.drop("State", axis=1, inplace=True)
    inputs, outputs = dataset.iloc[:, :-1], dataset.iloc[:, -1]

    features_train, features_test, labels_train, labels_test = train_test_split(inputs, outputs, test_size=0.2, random_state=0)
    plt.scatter(features_train.iloc[:, 0], labels_train.iloc[:], s=1)
    plt.scatter(features_train.iloc[:, 1], labels_train.iloc[:], s=1)
    plt.scatter(features_train.iloc[:, 2], labels_train.iloc[:], s=1)
    plt.show()
