# %%
import math 
import time 
import numpy as np
import torch
import matplotlib.pyplot as plt


# %%
n = 10000
a = torch.ones(n)
b = torch.ones(n)
a, b
# %%
def normal(x, mu, sigma):
    p = 1 / math.sqrt(2 * math.pi * sigma ** 2)
    return p * np.exp(-.5 / sigma**2 * (x - mu) ** 2)

