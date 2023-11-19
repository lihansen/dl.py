# %%
import torch 
from torch import nn
from d2l import torch as d2l

import sys
sys.path.append('..')
from utils.load_fshnmnist import load_data
from utils.device import device, try_gpu


# %%
# model definition
alexnet = nn.Sequential(

    nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),

    nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),

    nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),

    nn.MaxPool2d(kernel_size=3, stride=2), 
    nn.Flatten(),

    nn.Linear(6400, 4096), nn.ReLU(), 
    nn.Dropout(p=0.5),

    nn.Linear(4096, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),

    nn.Linear(4096, 10)
)


# %%

x = torch.randn(size=(1, 1, 224, 224))
for layer in alexnet:
    x = layer(x)
    print(layer.__class__.__name__,'output shape:\t', x.shape)

    
# %%
total_params = 0
for p in alexnet.parameters():
    total_params += p.numel()
    print(total_params, p.shape)

# %%
batch_size = 128
train_iter, test_iter = load_data(batch_size, resize=224)
len(train_iter), next(iter(train_iter))[0].shape


# %%
# training

lr = 0.09
num_epochs = 10


# %%
d2l.train_ch6(alexnet, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())


# %%
weights = [(name, param.shape) for name, param in alexnet.named_parameters()]
for w in weights:
    print(w)
# %%
print(alexnet)

# %%
