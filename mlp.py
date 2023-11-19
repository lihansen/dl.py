# %% 
import torch 
from torch import nn
from utils.load_fshnmnist import load_data
from training import train_ch3, train, cal_acc
from d2l import torch as d2l 

def gpu():
    return torch.device('cuda')
# %%
net = nn.Sequential(nn.Flatten(), 
                    nn.Linear(784, 256), 
                    nn.ReLU(), 
                    nn.Linear(256, 10))

net = net.to(device=gpu())



# %%
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=.01)

net.apply(init_weights)


# %%
batch_size = 265
lr = .1
n_epochs = 20
loss = nn.CrossEntropyLoss(reduction="none")

trainer = torch.optim.SGD(net.parameters(), lr = lr)

train_iter, test_iter = load_data(batch_size)

# %%
# for x, y in train_iter:
#     print(x.shape, y.shape)
#     break


# %%

# train_ch3(net, train_iter, test_iter, loss, n_epochs, trainer)
# %%
train(net, train_iter, loss, n_epochs, trainer)

# net.state_dict

# %%
acc = cal_acc(net, test_iter)
acc