# %%
import torch
# %%
x = torch.tensor(3)
y = torch.tensor(2)

x + y, x - y, x / y, x ** y
# %%
# vectors
x = torch.arange(3)
x
# %%
x.shape
# %%
# matrix
A = torch.arange(6).reshape(3, 2)
A
# %%
A.T
# %%
A == A.T.T


# %%
# tensors
X = torch.arange(24).reshape([2, 3, 4])
X.reshape(2, 3, 4)  # same thing when using reshape
X
# %%
A = torch.arange(20).reshape(5, 4)
B = A.clone()
A
# %%
A + B, A + B == 2 * A
# %%
# Hadamard product
# The elementwise product of two matrices 
A * B
# %%
a = 2
X = torch.arange(24).reshape(2,3,4)
a + X, (a * X).shape
# %%
