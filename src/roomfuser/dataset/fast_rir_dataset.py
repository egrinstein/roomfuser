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
        return_envelope: bool = True,
    ):
        """
        dataset_path: Path to the dataset folder
        n_rir: Number of RIRs to load. If None, load all RIRs in the dataset folder
        cat_labels: whether to concatenate labels into a torch tensor or return them as a dict
        normalize: whether to normalize the RIRs
        """

        self.dataset_path = dataset_path
        self.n_rir = n_rir
        self.return_envelope = return_envelope
        
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

        output = {"audio": rir, "conditioner": conditioner}


        if self.return_envelope:
            conditioner = decode_conditioner(conditioner)
            envelope = get_rir_envelope(
                n_envelope=rir.shape[-1],
                source_pos=conditioner["source_position"],
                mic_pos=conditioner["microphone_position"],
                rt60=conditioner["rt60"],
                sr=sr,
            )
            output["envelope"] = envelope

        return output


def decode_conditioner(conditioner):
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
    microphone_position = conditioner[:3]
    source_position = conditioner[3:6]
    room_dimensions = conditioner[6:9]
    rt60 = min(conditioner[9], 0.7) # Don't apply the correction, as the original is unknown.
    # Limit to 0.7 as the original dataset has a maximum of 0.7s for RT60.

    return {
        "microphone_position": microphone_position,
        "source_position": source_position,
        "room_dimensions": room_dimensions,
        "rt60": rt60,
    }


def get_rir_envelope(n_envelope, source_pos, mic_pos, rt60, sr=16000, c=343.0):
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
    """

    # Get distance between source and microphone
    distance = torch.linalg.norm(source_pos - mic_pos)

    # Get time of arrival of direct path, in samples
    t_direct = distance * sr / c

    # Get envelope centered at t=0
    t_envelope = torch.arange(n_envelope) / sr
    envelope = torch.exp(-6 * np.log(10) * t_envelope / rt60)
    # Shift envelope to the time of arrival of the direct path
    envelope = torch.roll(envelope, int(t_direct))

    # Zero pad the envelope
    # envelope = np.pad(envelope, (0, n_envelope - len(envelope)))
    return envelope
