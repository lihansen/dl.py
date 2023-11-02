# %%
import torch 
import matplotlib.pyplot as plt


figsize = (5, 2.5)

def plot(x, y, xlabel, ylabel, figsize):
    plt.figure(figsize=figsize)
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.show()


x = torch.arange(-8., 8., .1, requires_grad=True)
# %% relu
y = torch.relu(x)
plot(x.detach(), y.detach(), "x", "relu(x)", figsize=figsize)



# %% grad of relu figure
y.backward(torch.ones_like(x), retain_graph=True)
plot(x.detach(), x.grad, "x", "grad of relu(x)", figsize)



# %% sigmoid
y = torch.sigmoid(x)
plot(x.detach(), y.detach(), "x", "sigmoid(x)", figsize)
# plt.figure(figsize=figsize)
# plt.plot(x.detach(), y.detach())
# plt.xlabel("x")
# plt.ylabel("sigmoid(x)")
# plt.grid()
# plt.show()


# %% grad of sigmoid
x.grad.data.zero_()
y.backward(torch.ones_like(x), retain_graph=True)
plot(x.detach(), x.grad, "x", "grad of sigmoid(x)", figsize)
# plt.figure(figsize=figsize)
# plt.plot(x.detach(), x.grad)
# plt.xlabel("x")
# plt.ylabel("grad of sigmoid")
# plt.grid()
# plt.show()


# %%
y = torch.tanh(x)
plot(x.detach(), y.detach(), "x", "tanh(x)", figsize)
# plt.figure(figsize=figsize)
# plt.plot(x.detach(), x.grad)
# plt.xlabel("x")
# plt.ylabel("tanh(x)")
# plt.grid()
# plt.show()

# %%
x.grad.data.zero_()
y.backward(torch.ones_like(x), retain_graph=True) # keep computational graph
plot(x.detach(), x.grad, "x", "grad of tanh(x)", figsize)
# %%
