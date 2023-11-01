# %%
from matplotlib import cm 
from matplotlib.pyplot import figure
from matplotlib import pyplot as plt
import math
import numpy as np
figure(figsize=(8,6), dpi=100)
fig, ax = plt.subplots(subplot_kw={"projection":"3d"})

X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, .25)

X, Y = np.meshgrid(X, Y)
print(X.shape, Y.shape)

R = (6*X + 5*math.e**X + 5*Y*math.e**Y)

surf = ax.plot_surface(X, Y, R, cmap=cm.coolwarm)

