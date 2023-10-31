# %%
import os
import pandas as pd

# %%
os.makedirs(os.path.join(".", "data"), exist_ok=True)

# %%
data_file = os.path.join("data", "test.csv")

lines = [
    'NumRooms,Alley,Price',
    'NA,Pave,127500',
    '2,NA,106000',
    '4,NA,178100',
    'NA,NA,140000',
]

for i, line in enumerate(lines):
    lines[i] = line + "\n"
print(lines)
# %%
with open(data_file, "w") as f:
    f.writelines(lines)

# %%
data = pd.read_csv(data_file)
data
# %%
inputs, outputs = data.iloc[:, 0:2] , data.iloc[:, 2]
inputs
# %%
outputs
# %%
inputs = inputs.fillna(inputs.mean())
inputs
# %%
