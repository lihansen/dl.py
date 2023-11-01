# %%
import random
import torch
from torch.distributions import multinomial

# %%
fair_probs = torch.ones([6]) / 6
print(fair_probs)
# %%
multinomial.Multinomial(10, fair_probs).sample()
# %%
counts = multinomial.Multinomial(1000, fair_probs).sample()
counts
# %%
(counts / 1000)* 6
# %%
