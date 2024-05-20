import torch
import torchvision
from torch.utils import data
from torchvision import transforms

def softmax(X):
    0

if __name__ == "__main__":
    mnist_train = torchvision.datasets.FashionMNIST(
        root="./dataset", 
        train=True, 
        transform=transforms.ToTensor, 
        download=True
    )
    mnist_test = torchvision.datasets.FashionMNIST(
        root="./dataset", 
        train=False, 
        transform=transforms.ToTensor, 
        download=True
    )