# %%

import torch
from torch import nn
from d2l import torch as d2l

# load utils module, add parent dir to sys.path needed
import sys
sys.path.append("..")
from utils.load_fshnmnist import load_data
from utils.Animator import Animator


def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#%%
####################### model definition #######################

# class defined lenet
class LeNet(nn.Module):
    def __init__(self) -> None:
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.sigmoid1 = nn.Sigmoid()
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.sigmoid2 = nn.Sigmoid()
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(16 * 5 * 5, 120)
        self.sigmoid3 = nn.Sigmoid()

        self.linear2 = nn.Linear(120, 84)
        self.sigmoid4 = nn.Sigmoid()
        self.linear3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.sigmoid1(x)
        x = self.avgpool1(x)

        x = self.conv2(x)
        x = self.sigmoid2(x)
        x = self.avgpool2(x)

        x = self.flatten(x)

        x = self.linear1(x)
        x = self.sigmoid3(x)
        x = self.linear2(x)
        x = self.sigmoid4(x)
        x = self.linear3(x)
        return x


lenet = LeNet()
print(lenet)
# lenet.train()

# %%
total_params = 0
for p in lenet.parameters():
    total_params += p.numel()
    print(total_params, p.shape)


# %%

# sequential defined lenet model
lenet_seq = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, 10))


print(lenet_seq)

# %%
# check output shape from each layer, use dummy input, 
x = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)

for layer in lenet_seq:
    x = layer(x)
    print(layer.__class__.__name__, 'output shape:\t', x.shape)

# %%
x = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)

for layer in lenet.children(): # return layers of the model as an iterator
    x = layer(x)
    print(layer.__class__.__name__, 'output shape:\t', x.shape)


# %%

batch_size = 256
train_iter, test_iter = load_data(batch_size=batch_size)

# %%
####################### model training #######################
def init_weights(layer):
    if type(layer) == nn.Linear or type(layer) == nn.Conv2d:
        nn.init.xavier_uniform_(layer.weight)


def train_lenet(net, train_iter, test_iter, num_epochs, lr, device):
    net.apply(init_weights) # initialize weights

    print("training on ", device) # show device info
    net.to(device)

    optimizer = torch.optim.SGD(net.parameters(), lr=lr) # optimizer
    criterion = nn.CrossEntropyLoss() # loss function


    timer, num_batches = d2l.Timer(), len(train_iter)
    animator = Animator(xlabel='epoch', 
                        xlim=[1, num_epochs], 
                        title="self-defined lenet training",
                        legend=['train loss', 'train acc', 'test acc'])


    for epoch in range(num_epochs):
        # metric = d2l.Accumulator(3) # metric has 3 items: loss, train acc, num of samples
        accumulated_loss = 0
        accumulated_acc = 0
        accumulated_num_samples = 0
        net.train() # set model to train mode, dropout and batchnorm will be used

        for i, (x, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            
            x, y = x.to(device), y.to(device)

            output = net(x)
            loss = criterion(output, y)
            loss.to(device)

            loss.backward() # compute gradient

            optimizer.step() # update parameters

            with torch.no_grad():
                # metric.add(loss * x.shape[0], d2l.accuracy(output, y), x.shape[0])
                accumulated_loss += loss * x.shape[0]
                accumulated_acc += d2l.accuracy(output, y)
                accumulated_num_samples += x.shape[0]

            timer.stop()

            train_loss = accumulated_loss / accumulated_num_samples
            train_acc = accumulated_acc / accumulated_loss

            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches, (train_loss.cpu(), train_acc.cpu(), None))
        
        # evaluate on test set
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))

    print(f'loss {train_loss:.3f}, train acc {train_acc:.3f}, test acc {test_acc:.3f}')
    print(f'{accumulated_num_samples * num_epochs / timer.sum():.1f} examples/sec on {str(device)}')


# %%
lr, num_epochs = 0.9, 10
# %%
# d2l.train_ch6(lenet, train_iter, test_iter, num_epochs, lr, device())

# %%
train_lenet(lenet, train_iter, test_iter, num_epochs, lr, device())

# %%
