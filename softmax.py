import torch
import torchvision
import matplotlib.pyplot as plt
import os
import time
from torch.utils import data
from torchvision import transforms

def softmax(X):
    0

def get_dataloader_workers():
    return os.cpu_count()

def fashion_mnist_get_textlabel(i):
    text_lables = [
        "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    ]
    return text_lables[i]

if __name__ == "__main__":
    start_time = time.time()
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

    end_time = time.time()

    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(12, 6))
    for i in range(10):
        image, label = data_train[i]
        axes[i//5, i%5].imshow(image.squeeze(), cmap='hot')
        axes[i//5, i%5].set_title(f'Label: {fashion_mnist_get_textlabel(label)}')
        axes[i//5, i%5].axis('off')
    plt.tight_layout()
    plt.show()

    print(end_time - start_time)