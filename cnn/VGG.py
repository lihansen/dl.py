# %%
import torch
from torch import nn
from d2l import torch as d2l

import sys
sys.path.append('..')
from utils.load_fshnmnist import load_data
# %%


def VGG_block(num_convs, in_channels, out_channels):
    layers = []
    for i in range(num_convs):
        layers.append(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )
        layers.append(nn.ReLU())
        in_channels = out_channels

    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)

# %%


def VGG(conv_arch):
    conv_blks = []
    in_channels = 1

    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(VGG_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(
        *conv_blks,

        nn.Flatten(),

        nn.Linear(out_channels * 7 * 7, 4096),
        nn.ReLU(),
        nn.Dropout(p=0.5),

        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Dropout(p=0.5),

        nn.Linear(4096, 10)
    )


# %%
conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
vgg = VGG(conv_arch)


# %%
total_params = 0
for p in vgg.parameters():
    total_params += p.numel()
    print(total_params, p.shape)


# %%
x = torch.randn(size=(1, 1, 224, 224))
for block in vgg:
    x = block(x)
    print(block.__class__.__name__, 'output shape:\t', x.shape)
# %%
ratio = 4

small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
small_vgg = VGG(small_conv_arch)
# %%
total_params = 0
for p in small_vgg.parameters():
    total_params += p.numel()
    print(total_params, p.shape)
# %%
lr, num_epochs, batch_size = 0.05, 10, 128
train_iter, test_iter = load_data(batch_size, resize=224)
d2l.train_ch6(small_vgg, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
# %%
lr, num_epochs, batch_size = 0.05, 10, 128
train_iter, test_iter = load_data(batch_size, resize=224)
d2l.train_ch6(vgg, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
# %%
