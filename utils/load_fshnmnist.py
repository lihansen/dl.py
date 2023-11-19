
import torchvision
from torch.utils import data
from torchvision import transforms

# make project root dir
import os 
proj_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
data_dir = os.path.join(proj_dir, "data")


# load fashion mnist data, return train and test data loader,
# transform data to tensor
def load_data(batch_size, resize=None):
    trans = []

    if resize:
        trans.append(transforms.Resize(resize))
    trans.append(transforms.ToTensor())

    trans = transforms.Compose(trans)

    data_train = torchvision.datasets.FashionMNIST(
        root=data_dir, train=True, transform=trans, download=True
    )
    data_test = torchvision.datasets.FashionMNIST(
        root=data_dir, train=False, transform=trans, download=True
    )

    return (data.DataLoader(data_train, batch_size, shuffle=True,
                            num_workers=4),
            data.DataLoader(data_train, batch_size, shuffle=False,
                            num_workers=4))


def load_array(data_array, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_array)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10