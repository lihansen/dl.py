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

x = torch.arange(4)
x, x.sum()

# %%
A, A.shape, A.sum()
# %%
A.sum(axis=0)
# %%
A.sum(axis=1).T
# %%
A.sum(axis=[0,1])
# %%
A = A.type(torch.float) # casting data type must copy memory
A.dtype

# %%
A.mean()

# %%
A.sum() / A.numel()
# %%
A.mean(axis=0)
# %%
A.sum(axis=0) / A.shape[0]
# %%
reduction_A = A.sum(axis = 1)
reduction_A, reduction_A.shape

# %%
keepdim_A = A.sum(axis=1, keepdim=True)
keepdim_A, keepdim_A.shape

# %%
A / keepdim_A # broad casting 

# %%
# the cumulative sum of elements of A 
A, A.cumsum(axis=0)
# %%
# Dot product
y = torch.ones(4)
y, y.dtype
# %%
x = x.type(dtype=y.dtype)
x.dtype, y.dtype
# %%
torch.dot(x, y)
# %%
x * y
# %%
A.shape, x.shape
# %%
torch.mv(A, x)
# %%
A, x, A@x
# %%
B = torch.ones(4,5)
B.dtype
# %%
torch.mm(A, B), A@ B
# %%
# NORM

u = torch.tensor([3., -4.])
u, torch.norm(u) # L2 Norm

# %%
torch.abs(u).sum() # L1 norm
# %%
torch.norm(torch.ones((4, 9))) # Frobenius norm
# %%
# Frobenius norm is elementwise square, then sum and square root
# L2 norm is vector wise 