# %%
import torch 
import torchvision

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
    _, axis = 