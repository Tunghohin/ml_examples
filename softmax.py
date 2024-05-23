import torchvision
import random
from itertools import starmap
import matplotlib.pyplot as plt
from torch.utils import data
from torchvision import transforms
from torch import nn
from torch import optim

def model_train(dataloader, model, loss_fn, optimizer):
    model.train()
    for i, (X, y) in enumerate(dataloader):
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y).mean()
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            print(f"loss: {loss}")
            
def evaluate_accuracy(y_hat, y) -> float:
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    return float((y_hat == y).sum()) / y.numel()

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

if __name__ == "__main__":
    learning_rate = 0.02
    batch_size = 500
    net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
    loss = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.SGD(net.parameters(), lr=learning_rate)
    num_epoch = 20

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

    for i in range(num_epoch):
        print(f"epoch: {i + 1}/{num_epoch}")
        model_train(train_loader, net, loss, optimizer)

    avg = sum(
        starmap(lambda X, y: evaluate_accuracy(net(X), y), test_loader)
    ) / len(test_loader)
    print(f"Average precision: {avg}")

    mnist_random_sample_img(data_test, 2, 5)
