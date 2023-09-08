import os
import pickle
import soundfile as sf
import torch

from torch.utils.data import Dataset

from roomfuser.utils import MinMaxScaler, get_dataset_min_max_scaler
from roomfuser.utils import format_rir


class FastRirDataset(Dataset):
    """Room impulse response dataset. Loads RIRs provided by 
    the FAST RIR paper (https://github.com/anton-jeran/FAST-RIR).
    """

    def __init__(
        self,
        dataset_path: str,
        n_rir: int = None,
        trim_direct_path: bool = True,
        scaler_path: str = "",
        frequency_response=False,
    ):
        """
        dataset_path: Path to the dataset folder
        n_rir: Number of RIRs to load. If None, load all RIRs in the dataset folder
        cat_labels: whether to concatenate labels into a torch tensor or return them as a dict
        normalize: whether to normalize the RIRs
        """

        self.dataset_path = dataset_path
        self.n_rir = n_rir
        self.trim_direct_path = trim_direct_path
        
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

        self.scaler = None
        if scaler_path != "":
            self.scaler = MinMaxScaler(scaler_path)

        self.frequency_response = frequency_response
    
    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):
        id = self.file_names[idx]
        rir_path = os.path.join(self.rir_path, f"{id}.wav")

        rir, sr = sf.read(rir_path)

        # 1. Load RIR
        rir = torch.from_numpy(rir).float()

        # 2. Load conditioner (aka label)
        conditioner = self.conditioners[id]
        labels = self.decode_conditioner(conditioner)

        # 3. Format (trim and truncate) RIR
        rir = format_rir(rir, labels, self.n_rir, self.trim_direct_path)
        
        out = {"rir": rir, "conditioner": conditioner,
                  "labels": labels}

        if self.scaler is not None:
            # Apply min-max scaling
            out["rir"] = self.scaler.scale(out["rir"])

        if self.frequency_response:
            # Compute the frequency response
            out["rir"] = torch.fft.rfft(out["rir"])

            # Normalize the RIR using the magnitude
            out["rir"] = out["rir"] / torch.max(torch.abs(out["rir"]))
            
            # Convert complex to 2 channels
            out["rir"] = torch.stack((out["rir"].real, out["rir"].imag), dim=0)

        return out

    def decode_conditioner(self, conditioner):
        """According to https://github.com/anton-jeran/FAST-RIR:
        
        Each normalized embedding is created as follows:
        If you are using our trained model, you may need to use extra parameter Correction(CRR).
            Listener Position = LP
            Source Position = SP
            Room Dimension = RD
            Reverberation Time = T60
            Correction = CRR

            CRR = 0.1 if 0.5<T60<0.6
            CRR = 0.2 if T60>0.6
            CRR = 0 otherwise

            Embedding = ([LP_X,LP_Y,LP_Z,SP_X,SP_Y,SP_Z,RD_X,RD_Y,RD_Z,(T60+CRR)] /5) - 1
        """
        
        conditioner = (conditioner + 1) * 5
        mic_pos = conditioner[:3]
        source_pos = conditioner[3:6]
        room_dims = conditioner[6:9]
        rt60 = min(conditioner[9], 0.7) # Don't apply the correction, as the original is unknown.
        # Limit to 0.7 as the original dataset has a maximum of 0.7s for RT60.
        return {
            "mic_pos": mic_pos,
            "source_pos": source_pos,
            "room_dims": room_dims,
            "rt60": rt60,
        }


if __name__ == "__main__":
    from roomfuser.params import params
    dataset = FastRirDataset(params.fast_rir_dataset_path, n_rir=params.rir_len,
                             trim_direct_path=params.trim_direct_path)

    scaler = get_dataset_min_max_scaler(dataset)
    torch.save(scaler, params.fast_rir_scaler_path)
    print("Saved scaler")
    print(scaler)