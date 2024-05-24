import torchvision
import random
import torch
import time
from itertools import starmap
import matplotlib.pyplot as plt
from torch.utils import data
from torchvision import transforms
from torch import nn
from torch import optim
from sklearn.manifold import TSNE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def model_train(dataloader, model, loss_fn, optimizer):
    model.train()
    for i, (X, y) in enumerate(dataloader):
        optimizer.zero_grad()
        pred = model(X.to(device)).to(device)
        loss = loss_fn(pred, y.to(device)).mean()
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            print(f"iter: {i + 1}/{len(dataloader)}, loss: {loss}")
            
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
    learning_rate = 0.002
    batch_size = 4096
    net = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=5),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(32, 64, kernel_size=5),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(64 * 4 * 4, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    # net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
    # optimizer = optim.SGD(net.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    num_epoch = 40
    net.to(device=device)
    loss_fn.to(device=device)

    data_train = torchvision.datasets.FashionMNIST(
        root="./dataset", 
        train=True, 
        transform=transforms.Compose([
            transforms.ToTensor(),
        ]), 
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

    # num_samples = 1000
    # train_data = data_train.data[:num_samples].numpy().reshape(num_samples, -1)
    # train_labels = data_train.targets[:num_samples].numpy()

    # tsne = TSNE(n_components=3, random_state=42)
    # train_tsne = tsne.fit_transform(train_data)

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # scatter = ax.scatter(train_tsne[:,0], train_tsne[:,1], train_tsne[:,2], c=train_labels, cmap='tab10')
    # legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
    # ax.add_artist(legend1)
    # plt.show()

    start_time = time.time()
    for i in range(num_epoch):
        print(f"epoch: {i + 1}/{num_epoch}")
        model_train(train_loader, net, loss_fn, optimizer)
    end_time = time.time()

    avg = sum(
        starmap(lambda X, y: evaluate_accuracy(net(X.to(device)).to("cpu"), y.to("cpu")), test_loader)
    ) / len(test_loader)
    print(f"Average precision: {avg}")
    print(f"Train cost: {end_time - start_time}s")

    mnist_random_sample_img(data_train, 2, 5)
