import os
import numpy as np
import pickle
import soundfile as sf
import torch

from tqdm import tqdm
from torch.utils.data import Dataset


class FastRirDataset(Dataset):
    """Room impulse response dataset. Loads RIRs provided by 
    the FAST RIR paper (https://github.com/anton-jeran/FAST-RIR).
    """

    def __init__(
        self,
        dataset_path: str,
        n_rir: int = None,
    ):
        """
        dataset_path: Path to the dataset folder
        n_rir: Number of RIRs to load. If None, load all RIRs in the dataset folder
        cat_labels: whether to concatenate labels into a torch tensor or return them as a dict
        normalize: whether to normalize the RIRs
        """

        self.dataset_path = dataset_path
        self.n_rir = n_rir
        
        self.rir_path = os.path.join(self.dataset_path, "RIR")
        self.labels_path = os.path.join(self.dataset_path, "train")
        
        self.file_names_pickle = os.path.join(self.labels_path, "filenames.pickle") 
        self.conditioners_pickle = os.path.join(self.labels_path, "embeddings.pickle")

        self.file_names = pickle.load(open(self.file_names_pickle, "rb"))
        self.conditioners = pickle.load(open(self.conditioners_pickle, "rb"))

        # Convert conditioners to torch tensors
        for key in self.conditioners:
            self.conditioners[key] = torch.from_numpy(self.conditioners[key]).float()

        super().__init__()
    
    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):
        id = self.file_names[idx]
        rir_path = os.path.join(self.rir_path, f"{id}.wav")

        rir, sr = sf.read(rir_path)

        # 1. Load RIR
        rir = torch.from_numpy(rir).float()

        if self.n_rir:
            max_len = min(rir.shape[-1], self.n_rir)
            # Pad with zeros in the end if the RIR is too short, or truncate if it's too long
            rir = torch.nn.functional.pad(rir, (0, self.n_rir - max_len))[:self.n_rir]

        # 2. Load conditioner (aka label)
        conditioner = self.conditioners[id]

        return {"audio": rir, "conditioner": conditioner}
