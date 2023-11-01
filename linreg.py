# %%
import math 
import time 
import numpy as np
import torch
import matplotlib.pyplot as plt
import random

from torch.utils import data
from torch import nn



# %%
n = 10000
a = torch.ones(n)
b = torch.ones(n)
a, b
# %%
# squred loss is the simple version of maximum likelihood 
def normal(x, mu, sigma):
    p = 1 / math.sqrt(2 * math.pi * sigma ** 2)
    return p * np.exp(-.5 / sigma**2 * (x - mu) ** 2)


# %%
def synthetic_data(w, b, n_samples):
    dim = len(w)
    X = torch.normal(0, 1, (n_samples, dim)) # generate x
    y = torch.matmul(X, w) + b # mm without broadcasting, matmul with broadcasting 
    
    y += torch.normal(0, 0.01, y.shape) # add perturbation 
    return X, y.reshape((-1, 1)) 

true_w = torch.tensor([2, -3.4])
true_b = 4.2

feats, labels = synthetic_data(true_w, true_b, 1000)

feats, labels, feats[0], labels[0]
# %%
plt.scatter(feats[:, 1], labels)
# %%
plt.scatter(feats[:, 0], labels)
# %%
def data_iter(batch_size, feats, labels):
    n_samples = len(feats)
    indices = range(n_samples)

    random.shuffle(list(indices))

    for i in range(0, n_samples, batch_size):
        batch_indices = torch.tensor(
            indices[i : min(i + batch_size, n_samples)]
        )
        yield feats[batch_indices], labels[batch_indices]
# %%
batch_size = 10 

data_loader = data_iter(batch_size, feats, labels)
feat, lab = next (data_loader)
feat.shape, lab.shape, feat, lab
# %%
# init 

w = torch.normal(0, 0.01, size = [2,1], requires_grad=True)
b = torch.zeros(1, requires_grad=True)


# %%
def linreg(X, w, b):
    return torch.matmul(X, w) + b


# %%
def squared_loss(y_hat, y):
    y = y.reshape(y_hat.shape)
    return (y_hat - y) ** 2 / 2
# %%
# min batch sgd
def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


# %%
###### training #####

lr = .03
n_epochs = 4
net = linreg
loss = squared_loss

for epoch in range(n_epochs):
    for X, y in data_iter(batch_size, feats, labels):
        y_hat = net(X, w, b)
        delta = loss(y_hat, y)
        delta.sum().backward()

        sgd([w, b], lr, batch_size)

    with torch.no_grad():
        train_loss = loss(net(feats, w, b), labels)
        print(f"epoch  {epoch + 1}, loss {float(train_loss.mean()):f}")
# %%
print(f'estimate error of w: {true_w - w.reshape(true_w.shape)}')
print(f'estimate error of b: {true_b - b}')
# %%
true_w = torch.tensor([2, -3.4])
true_b = 4.2
feats, labels = synthetic_data(true_w, true_b, 1000)


# %%
def load_array(data_array, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_array)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10

data_iter = load_array((feats, labels), batch_size)
# %%
next(iter(data_iter))
# %%
net = nn.Sequential(nn.Linear(2,1))

# %%
# init params

net[0].weight.data.normal_(0, .01)
net[0].bias.data.fill_(0)


# %%
loss = nn.MSELoss()

# %%
trainer = torch.optim.SGD(net.parameters(), lr=.03)
# %%
n_epochs = 3
for epoch in range(n_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(feats), labels)
    print(f"epoch:{epoch + 1}, loss:{l:f}")
# %%
net[0] == net[-1], net
# %%
w = net[0].weight.data
n = net[0].bias.data
w, true_w
# %%
b, true_b
# %%
print(f"estimate error of w:{true_w - w.reshape(true_w.shape)}")
print(f"estimate error of b:{true_b - b}")
# %%
