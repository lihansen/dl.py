# %%
import torch 
from torch import nn 

def dropout_layer(x, dropout):
    assert 0 <= dropout <= 1

    if dropout == 1:
        return torch.zeros_like(x)
    
    if dropout == 0:
        return x
    
    mask = (torch.rand(x.shape) > dropout).float()
    return mask * x / (1.0 - dropout)




# %%
x = torch.arange(16, dtype=torch.float32).reshape((2, 8))
x
# %%
torch.rand(x.shape)
# %%
mask = torch.rand(x.shape) > .5
mask
# %%
mask.float() * x / (1.0 - .5)

# %%
dropout_layer(x, 0)

# %%
dropout_layer(x, .5)
# %%
dropout_layer(x, .6)
# %%
dropout_layer(x, 1)

# %%
n_inputs = 784
n_outputs = 10
n_hiddens1 = 256
n_hiddens2 = 256

dropout1 = .2
dropout2 = .5

# %%
class Net(nn.Module):
    def __init__(self, n_inputs, n_outputs, n_hiddens1, n_hiddens2, is_training=True) -> None:
        super(Net, self).__init__()

        nn.Dropout