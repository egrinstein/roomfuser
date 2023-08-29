import os
import soundfile as sf
import torch

from torch.utils.data import Dataset

from roomfuser.utils import format_rir


class RirDataset(Dataset):
    """Room impulse response dataset. Unlike RandomRirDataset, this dataset
    loads RIRs from disk.
    """

    def __init__(
        self,
        dataset_path: str,
        n_rir: int = None,
        normalize: bool = True,
        sr=16000,
        trim_direct_path: bool = False,
    ):
        """
        dataset_path: Path to the dataset folder
        n_rir: Number of RIRs to load. If None, load all RIRs in the dataset folder
        cat_labels: whether to concatenate labels into a torch tensor or return them as a dict
        normalize: whether to normalize the RIRs
        """

        self.dataset_path = dataset_path
        self.n_rir = n_rir
        self.normalize = normalize

        self.sr = sr
        self.trim_direct_path = trim_direct_path

        self
        self.rir_files = sorted(
            [
                f
                for f in os.listdir(dataset_path)
                if os.path.isfile(os.path.join(dataset_path, f))
                and f.startswith("rir")
                and f.endswith(".wav")
            ]
        )

        super().__init__()
    
    def __len__(self):
        return len(self.rir_files)
    
    def __getitem__(self, idx):
        rir_path = os.path.join(self.dataset_path, self.rir_files[idx])
        rir, sr = sf.read(rir_path)

        # 1. Load RIR
        rir = torch.from_numpy(rir).float()
        if self.normalize:
            # Normalize the RIR using the maximum absolute value
            rir = rir / torch.max(torch.abs(rir))
        

        # 2. Load conditioner (aka label)
        label_filename = self.rir_files[idx].replace("rir", "label").replace(".wav", ".pt")
        label_path = os.path.join(self.dataset_path, label_filename)
        labels = torch.load(label_path)

        rir = format_rir(rir, labels, self.n_rir, self.trim_direct_path)

        conditioner = torch.cat((
            labels["room_dims"].float(),
            labels["source_pos"].float(),
            labels["mic_pos"].float(),
            torch.Tensor([labels["rt60"]]).float(),
        ), dim=0)


        out = {
            "rir": rir, "conditioner": conditioner,
            "labels": labels
        }

        return out
