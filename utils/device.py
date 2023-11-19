import torch 


def device():
    """Return cuda device if available, otherwise return cpu.

    Defined in :numref:`sec_use_gpu`"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu().

    Defined in :numref:`sec_use_gpu`"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')