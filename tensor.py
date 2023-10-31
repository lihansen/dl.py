# %%
import numpy as np
import torch

# %%
x = torch.arange(12)
x
# %%
x.shape
# %%
x.numel()
# %%
X = x.reshape(3, 4)
X
# %%
torch.zeros([2, 3, 4])

# %%
torch.ones((2, 3, 4))
# %%
a = torch.randn(3, 4)
a.reshape([2, 6])
a.reshape([1, 12])
a.reshape(12)
# %%
x = torch.tensor([1, 2, 4, 8])
y = torch.tensor((2, 2, 2, 2))
x + y

# %%
x - y
# %%
x * y
# %%
x / y
# %%
x ** y
# %%
torch.exp(x)
# %%
X = torch.arange(12, dtype=torch.int8).reshape([3, 4])
X
# %%
Y = torch.tensor([[1, 2, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
Y
# %%
torch.cat((X, Y), dim=0)  # concate with axis row
# %%
torch.cat((X, Y), dim=1)  # concate with axis col

# %%
X == Y
# %%
X.sum()

# %%
# broadcasting

a = torch.arange(3).reshape([3, 1])
b = torch.arange(2).reshape([1, 2])

a

# %%
b
# %%
a + b
# %%
X
# %%
X[1][2]
# %%
X[0, 2]
# %%
X[0:2, :] = 12
X
# %%
X[-1, 0]




# %%
before = id(Y)
Y = Y + X
print(before)
id(Y)
# %%
Z = torch.zeros_like(Y)
print(id(Z))
Z[:] = X + Y
print(id(Z))
# %%
before = id (X)
X += Y
print(before)
print(id(X))
# %%
A = X.numpy()
A
# %%
B = torch.tensor(A)
print(type(A))
type(B)
# %%
