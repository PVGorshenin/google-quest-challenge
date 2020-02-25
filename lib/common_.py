import torch
import pandas as pd


def ts(x):
    return torch.Tensor(x)


def to_numpy(x_tensor):
    return x_tensor.cpu().detach().numpy()


def df_(x):
    return pd.DataFrame(x)


def vc_(x):
    return x.value_counts()


