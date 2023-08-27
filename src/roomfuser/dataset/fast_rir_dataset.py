import os
import pickle
import soundfile as sf
import torch

from torch.utils.data import Dataset


class FastRirDataset(Dataset):
    """Room impulse response dataset. Loads RIRs provided by 
    the FAST RIR paper (https://github.com/anton-jeran/FAST-RIR).
    """

    def __init__(
        self,
        dataset_path: str,
        n_rir: int = None,
        trim_direct_path: bool = True,
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

        # 3. Trim direct path
        if self.trim_direct_path:
            t = get_direct_path_idx(labels)
            rir = rir[t:]

        # 4. Pad or truncate RIR
        if self.n_rir:
            max_len = min(rir.shape[-1], self.n_rir)
            # Pad with zeros in the end if the RIR is too short, or truncate if it's too long
            rir = torch.nn.functional.pad(rir, (0, self.n_rir - max_len))[:self.n_rir]
        
        
        output = {"audio": rir, "conditioner": conditioner,
                  "labels": labels}

        return output

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
        room_dimensions = conditioner[6:9]
        rt60 = min(conditioner[9], 0.7) # Don't apply the correction, as the original is unknown.
        # Limit to 0.7 as the original dataset has a maximum of 0.7s for RT60.
        return {
            "mic_pos": mic_pos,
            "source_pos": source_pos,
            "room_dimensions": room_dimensions,
            "rt60": rt60,
        }


def get_direct_path_idx(labels, sr=16000, c=343):
    source_pos = labels["source_pos"]
    mic_pos = labels["mic_pos"]


    # Get distance between source and microphone
    distance = torch.linalg.norm(source_pos - mic_pos)

    # Get time of arrival of direct path, in samples
    t_direct = distance * sr / c

    return int(t_direct)
