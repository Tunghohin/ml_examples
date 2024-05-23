import torch
import torchvision
import matplotlib.pyplot as plt
import os
import time
import random
from torch.utils import data
from torchvision import transforms

W = torch.normal(0, 0.01, size=(784, 10), requires_grad=True)
b = torch.zeros(10, requires_grad=True)
batch_size = 1000


class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    def __getitem__(self, idx) -> float: 
        return self.data[idx]
    
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)
    
def softmax_classification(X):
    return softmax(torch.matmul(X.reshape(-1, W.shape[0]), W) + b)

def softmax(X):
    X_exp = torch.exp(X)
    return X_exp / X_exp.sum(1, keepdim=True)

def get_dataloader_workers():
    return os.cpu_count()

def get_fashion_mnist_textlabel(i):
    return torchvision.datasets.FashionMNIST.classes[i]

def mnist_random_sample_img(dataset, nrows=5, ncols=5, cmap='hot'):
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 6))
    for i, j in zip(range(ncols * nrows), random.sample(range(len(dataset)), ncols * nrows)):
        image, label = dataset[j]
        axes[i//ncols, i%ncols].imshow(image.squeeze(), cmap=cmap)
        axes[i//ncols, i%ncols].set_title(f'Label: {get_fashion_mnist_textlabel(label)}')
        axes[i//ncols, i%ncols].axis('off')
    plt.tight_layout()
    plt.show()

def cross_entropy(y_pred, y):
    return -torch.log(y_pred[range(len(y_pred)), y])

def accuracy(y_hat, y) -> float:
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    return float((y_hat == y).sum())

def evaluate_accuracy(net, data_iter):
    if isinstance(net, torch.nn.Module):
        net.eval()
    matric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            matric.add(accuracy(net(X), y), y.numel())
    return matric[0] / matric[1]

def train_epoch(net, train_iter, loss, updater):
    if isinstance(net, torch.nn.Module):
        net.train()
    metric = Accumulator(3) #
    for X, y in train_iter:
        y_pred = net(X)
        l = loss(y_pred, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            l.sum().backward()
            updater(X.shape[0])
        
    

if __name__ == "__main__":
    net = softmax_classification
    loss = cross_entropy

    data_train = torchvision.datasets.FashionMNIST(
        root="./dataset", 
        train=True, 
        transform=transforms.ToTensor(), 
        download=True,  
    )
    data_test = torchvision.datasets.FashionMNIST(
        root="./dataset", 
        train=False, 
        transform=transforms.ToTensor(), 
        download=True
    )

    train_loader = data.DataLoader(
        data_train, 
        batch_size, 
        shuffle=True,
    )
    test_loader = data.DataLoader(
        data_test, 
        batch_size, 
        shuffle=False,
    )

    start_time = time.time()
    for i, (X, y_true) in enumerate(train_loader):
        print(accuracy(net(X), y_true) / y_true.numel())
        break
    end_time = time.time()

    print(evaluate_accuracy(net, train_loader))

    print(end_time - start_time)

    # mnist_random_sample_img(data_test, 2, 5)