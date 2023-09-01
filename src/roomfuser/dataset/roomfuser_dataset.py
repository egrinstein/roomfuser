import os
import soundfile as sf
import torch

from torch.utils.data import Dataset

from roomfuser.utils import MinMaxScaler, format_rir, get_dataset_min_max_scaler


class RirDataset(Dataset):
    """Room impulse response dataset. Unlike RandomRirDataset, this dataset
    loads RIRs from disk.
    """

    def __init__(
        self,
        dataset_path: str,
        n_rir: int = None,
        normalize: bool = False,
        sr=16000,
        trim_direct_path: bool = False,
        scaler_path: str = "",
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

        self.scaler = None
        if scaler_path != "":
            self.scaler = MinMaxScaler(scaler_path)

        super().__init__()
    
    def __len__(self):
        return len(self.rir_files)
    
    def __getitem__(self, idx):
        rir_path = os.path.join(self.dataset_path, self.rir_files[idx])
        rir, sr = sf.read(rir_path)

        # 1. Load RIR
        rir = torch.from_numpy(rir)
        if self.normalize:
            # Normalize the RIR using the maximum absolute value
            rir = rir / torch.max(torch.abs(rir))
        

        # 2. Load conditioner (aka label)
        label_filename = self.rir_files[idx].replace("rir", "label").replace(".wav", ".pt")
        label_path = os.path.join(self.dataset_path, label_filename)
        labels = torch.load(label_path)

        if isinstance(labels["rt60"], torch.Tensor):
            labels["rt60"] = labels["rt60"].item()

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

        if self.scaler is not None:
            # Apply min-max scaling
            out["rir"] = self.scaler.scale(out["rir"])


        return out


if __name__ == "__main__":
    from roomfuser.params import params
    dataset = RirDataset(params.roomfuser_dataset_path, n_rir=params.rir_len,
                         trim_direct_path=params.trim_direct_path)

    scaler = get_dataset_min_max_scaler(dataset)
    torch.save(scaler, params.roomfuser_scaler_path)
    print("Saved scaler")
    print(scaler)
