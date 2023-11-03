import torchvision
from torch.utils import data
from torchvision import transforms

def load_data(batch_size, resize=None):
    trans = [transforms.ToTensor()]

    if resize:
        trans.insert(0, transforms.Resize(resize))

    trans = transforms.Compose(trans)

    data_train = torchvision.datasets.FashionMNIST(
        root="data", train=True, transform=trans, download=True
    )
    data_test = torchvision.datasets.FashionMNIST(
        root="data", train=False, transform=trans, download=True
    )

    return (data.DataLoader(data_train, batch_size, shuffle=True,
                            num_workers=4),
            data.DataLoader(data_train, batch_size, shuffle=False,
                            num_workers=4))
