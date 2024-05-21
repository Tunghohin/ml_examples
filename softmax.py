import torch
import torchvision
import matplotlib.pyplot as plt
import os
import time
import random
from torch.utils import data
from torchvision import transforms

def softmax_classification(X, W, b):
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
    return - torch.log(y_pred[range(len(y_pred)), y])

if __name__ == "__main__":
    dim_input = 784
    dim_output = 10
    net = softmax_classification
    loss = cross_entropy
    w = torch.normal(0, 0.01, size=(dim_input, dim_output), requires_grad=True)
    b = torch.zeros(dim_output, requires_grad=True)
    batch_size = 1000

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
        print(net(X, w, b))
    end_time = time.time()

    mnist_random_sample_img(data_test, 5, 10)

    print(end_time - start_time)