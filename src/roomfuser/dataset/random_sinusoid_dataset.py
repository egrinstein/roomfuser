import numpy as np
import torch

from torch.utils.data import Dataset


class RandomSinusoidDataset(Dataset):
    """
    Random Sinusoid Dataset
    """
    def __init__(self, n_sample, n_samples_per_epoch, n_min_cycle=1, n_max_cycle=4):
        self.n_sample = n_sample
        self.n_samples_per_epoch = n_samples_per_epoch
        self.n_min_cycle = n_min_cycle
        self.n_max_cycle = n_max_cycle
    
    def __len__(self):
        return self.n_samples_per_epoch
    
    def __getitem__(self, idx):
        """
        Returns a random sinusoid function
        """
        # Generate random amplitude, phase, and number of cycles
        
        amplitude, phase, n_cycles = get_random_sinusoid_config(
                                      self.n_max_cycle, self.n_min_cycle)

        # Generate x values
        x = torch.arange(0, self.n_sample)

        # Generate y values
        y = amplitude * torch.sin(2 * torch.tensor(np.pi) * n_cycles * x / self.n_sample + phase)
        
        # Concatenate amplitude, phase, and number of cycles
        labels = torch.cat((amplitude, phase, n_cycles))

        return {
            'audio': y,
            'conditioner': labels,
        }


def get_random_sinusoid_config(n_max_cycle=4, n_min_cycle=1, cat=False):
    # Amplitude is uniform in [0.1, 1]
    #amplitude = torch.rand(1) * 0.9 + 0.1
    amplitude = torch.Tensor([0.5])

    # Phase is uniform in [0, 2*pi]
    phase = torch.rand(1) * 2 * torch.tensor(np.pi)
    
    # Number of cycles is uniform in [n_min_cycle, n_max_cycle]
    n_cycles = torch.rand(1)*(n_max_cycle - 1) + n_min_cycle

    if cat:
      return torch.cat((amplitude, phase, n_cycles))
    else:
      return amplitude, phase, n_cycles
