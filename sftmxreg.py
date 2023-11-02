# %%
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np

from torch.utils import data
from torchvision import transforms

# %%
trans = transforms.ToTensor()
data_train = torchvision.datasets.FashionMNIST(
    root="data", train=True, transform=trans, download=True
)
data_test = torchvision.datasets.FashionMNIST(
    root="data", train=False, transform=trans, download=True
)
# %%
data_train
# %%
len(data_train)
# %%
data_test
# %%
data_train[0][0].shape
# %%


def get_fashionmnist_labels(labels):
    test_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [test_labels[int(i)] for i in labels]


# %%
def show_img(imgs, num_rows, num_cols, titles=None, scale=1.5):
    figsize = (num_cols * scale, num_rows * scale)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()

    for i, (ax, img) in enumerate(zip(axes, imgs)):
        ax.imshow(img.numpy())
        # ax.axes.get_xaxis().set_visible(False)
        # ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])

    return axes


# %%
X, y = next(iter(data.DataLoader(data_train, batch_size=18)))

X.shape
# %%
show_img(X.reshape(18, 28, 28), 2, 9, titles=get_fashionmnist_labels(y))
# %%
batch_size = 256

train_iter = data.DataLoader(
    data_train, batch_size, shuffle=True, num_workers=4)
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
train_iter, test_iter = load_data(32, resize=64)

for X, y in train_iter:
    print(X.shape, X.dtype, y.shape, y.dtype)
    break


# %%
# implementation
batch_size = 256
train_iter, test_iter = load_data(batch_size)


# %%
# init weights
n_inputs = 784
n_outputs = 10

W = torch.normal(0, 0.01, size=(n_inputs, n_outputs,), requires_grad=True)
b = torch.zeros(n_outputs, requires_grad=True)


# %%

X = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
X = torch.ones((2, 5))
X.sum(0, keepdim=True), X.sum(1, keepdim=True)

# %%
# softmax

X_exp = torch.exp(X)
X, X_exp, X_exp.shape
#%%
partition = X_exp.sum(1, keepdim=True)
X, partition, partition.shape

# %%
sftmx = X / partition # broadcasting 
sftmx, sftmx.shape
# %%
def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition


# %%
X = torch.normal(0, 1, (2, 5))

X_prob = softmax(X)
X, X_prob, X_prob.sub(1)


# %%
# a = torch.tensor([[12]*5, [12]*5])
# a, a.shape
# # %%
# b = torch.tensor([[2], [6]])
# b, b.shape
# # %%
# a / b



# %%
def net(X):
    X = X.reshape([-1, W.shape[0]])
    linear = torch.matmul(X, W) + b
    return softmax(linear)
    # return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)
# %%
y = torch.tensor([0, 2])
y_hat = torch.tensor([[.1,.3,.6], [.3,.2,.5]])
y_hat, y, y_hat[0:2, y]
# %%
y_hat, y_hat[:, [0, 2]]

# %%
def cross_entropy(y_hat, y):
    return - np.log(y_hat[range(len(y_hat)), y])

cross_entropy(y_hat, y)
# %%
def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1 :
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

accuracy(y_hat, y) / len(y)
# %%
class Accumulator:
    def __init__(self, n_vars) -> None:
        self.data = [.0] * n_vars

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [.0] * len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    


# %%
def evaluate_accuracy(net, data_iter):
    if isinstance(net, torch.nn.Module):
        net.eval() # evaluation model
    metric = Accumulator(256) # num of 
    with torch.no_grad():
        for x, y in data_iter:
            metric.add(accuracy(net(x), y), y.numel())
    return metric[0] / metric[1]


# %%
next(iter(test_iter))[0].shape

# %%
evaluate_accuracy(net, test_iter)

# %%
def train(net, train_iter, loss, updater):
    if isinstance(net, torch.nn.Module):
        net.train()

    metric = Accumulator(3)

    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, )