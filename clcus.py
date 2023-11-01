# %%
import numpy as np
import torch 

# %%
def f(x):
    return 3 * x ** 2 - 4 * x

# %%
def numerical_lim(f, x, h):
    return (f(x + h) - f(x)) / h # derivative


h = 0.1
for i in range(5):
    print(f"h={h:.5f}, numerical limit={numerical_lim(f, 1, h):.5f}")
    h *= .1




# %%
x = torch.arange(4, dtype=torch.float)
x.type()
# %%
x.requires_grad_(True)
x
# %%
x.grad == None 
# %%
y = 2 * torch.dot(x, x)
# %%
y
# %%
y == 2*(x@x)
# %%
y
# %%
y.backward()
# %%
y
# %%
x.grad
# %%
x.grad == 4 * x
# %%




x.grad.zero_()
# %%
y = x.sum()
# %%
y
# %%
y.backward()
# %%
x.grad

# %%
x.grad.zero_()
y = x * x

y.sum().backward() # == y.backward(torch.ones(len(x)))
x.grad

# %%
x.grad.zero_()
y = x * x
u = y.detach()
u, y, x.grad

# %%
z = u * x

z.sum().backward()
x, u, y, x.grad == u
# %%
x.grad.zero_()
y.sum().backward()
x.grad == 2 * x
# %%
def f(a):
    b = a * 2

    while b.norm() < 1000:
        b = b*2
    
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    
    return c

a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()
a.grad
# %%
a.grad == d / a
# %%
d, a, d/a, a.grad
# %%
