# %%
import torch
import torchvision
from torch import nn
from torchvision import transforms
from torch.utils import data


# %%
def load_data(batch_size, resize=None):
    trans = [transforms.ToTensor()]

    if resize:
        trans.insert(0, transforms.Resize(resize))

    trans = transforms.Compose(trans)

    data_train = torchvision.datasets.FashionMNIST(
        root="data", train=True, transform=trans, download=True
    )
    data_test = torchvision.datasets.FashionMNIST(
        root="data", train=False, transform=trans, download=True
    )

    return (data.DataLoader(data_train, batch_size, shuffle=True,
                            num_workers=4),
            data.DataLoader(data_train, batch_size, shuffle=False,
                            num_workers=4))
# %%
def train(net, train_iter, test_iter, loss, num_epochs, updater):
    for epoch in range(num_epochs):
        train_matrics = train_epoch()

# %%
batch_size = 256

train_iter, test_iter = load_data(batch_size)


# %%
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=.01)

net.apply(init_weights)

# %%
loss = nn.CrossEntropyLoss(reduction='none')

trainer = torch.optim.SGD(net.parameters(), lr=.1)
# %%


# %%
num_epochs = 10
