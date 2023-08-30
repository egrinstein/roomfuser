import torch
from tqdm import trange


class MinMaxScaler:
    def __init__(self, path):
        self.scaler = torch.load(path)
    
    def scale(self, data):
        return (data - self.scaler["min"]) / (self.scaler["max"] - self.scaler["min"])

    def descale(self, data):
        return data * (self.scaler["max"] - self.scaler["min"]) + self.scaler["min"]


def get_dataset_min_max_scaler(dataset, key="rir"):
    """Get a scaler for the dataset which scales the data to have zero mean and unit variance.

    Args:
        dataset (Dataset): Torch dataset where samples contain a field named key.
    """

    min_vals = None
    max_vals = None

    n_dataset = len(dataset)
    for i in trange(n_dataset):
        sample = dataset[i]
        data = sample[key]
        if min_vals is None:
            min_vals = data
            max_vals = data
        else:
            min_vals = torch.min(min_vals, data)
            max_vals = torch.max(max_vals, data)

    scaler = {"min": min_vals, "max": max_vals}
    return scaler
