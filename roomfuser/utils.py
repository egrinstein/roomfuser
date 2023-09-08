# Acoustics-related functions

import numpy as np
import torch

from tqdm import trange


def get_exponential_envelope(n_envelope, rt60, sr=16000):
    """Get Room Impulse Response envelope. The envelope is modeled as a
    decaying exponential which reaches -60 dB (10^(-6)) after rt60 seconds.
    The exponential starts at the time of arrival of the direct path,
    which is the distance between the source and the microphone divided
    by the speed of sound.
    
    Args:
        n_envelope (int): Number of samples in the envelope.
        source_pos (np.array): Source position in meters.
        mic_pos (np.array): Microphone position in meters.
        sr (float): Sampling rate.
        rt60 (float): Reverberation time in seconds.
        c (float): Speed of sound in meters per second.
        start_at_direct_path (bool): If True, the envelope starts at the
            time of arrival of the direct path. If False, the envelope
            starts at t=0.
    """

    # Get envelope centered at t=0
    t_envelope = torch.arange(n_envelope) / sr
    envelope = torch.exp(-6 * np.log(10) * t_envelope / rt60)
    
    return envelope


def get_direct_path_idx(labels, sr=16000, c=343):
    source_pos = labels["source_pos"]
    mic_pos = labels["mic_pos"]


    # Get distance between source and microphone
    distance = torch.linalg.norm(source_pos - mic_pos)

    # Get time of arrival of direct path, in samples
    t_direct = distance * sr / c

    return int(t_direct)


def format_rir(rir, labels, n_rir, trim_direct_path=False):
    # 1. Trim direct path
    if trim_direct_path:
        t = get_direct_path_idx(labels)
        rir = rir[t:]

    # 4. Pad or truncate RIR
    if n_rir:
        max_len = min(rir.shape[-1], n_rir)
        # Pad with zeros in the end if the RIR is too short, or truncate if it's too long
        rir = torch.nn.functional.pad(rir, (0, n_rir - max_len))[:n_rir]
    
    return rir


# Misc utils

def dict_to_device(d, device):
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            d[k] = v.to(device, dtype=torch.float32)
    return d


class MinMaxScaler:
    """Class for min-max scaling of data. to the range [-1, 1]."""
    def __init__(self, path):
        self.scaler = torch.load(path)
    
    def scale(self, data):
        dict_to_device(self.scaler, data.device)
        # Scale to [0, 1]
        scaled = (data - self.scaler["min"]) / (self.scaler["max"] - self.scaler["min"])
        # Scale to [-1, 1]
        scaled = scaled * 2 - 1
        return scaled
    
    def descale(self, data):
        dict_to_device(self.scaler, data.device)
        # Scale to [0, 1]
        descaled = (data + 1) / 2
        # Scale to [min, max]
        descaled = descaled * (self.scaler["max"] - self.scaler["min"]) + self.scaler["min"]
        return descaled


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
