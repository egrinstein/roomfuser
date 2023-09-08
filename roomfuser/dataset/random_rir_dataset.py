import os
import numpy as np
import soundfile as sf
import torch

from tqdm import tqdm

from roomfuser.dataset.simulator import RirSimulator
from roomfuser.utils import format_rir


class RandomRirDataset(torch.utils.data.Dataset):
    """Generate a random room impulse response dataset."""

    def __init__(
        self,
        n_rir: int,
        n_samples_per_epoch: int,
        room_dims_range: tuple = ((3, 10), (3, 10), (2, 4)),
        rt60_range: tuple = (0.2, 1.2),
        absorption_range: tuple = (0.5, 1),
        sr: float = 16000,
        backend: str = "auto",
        normalize: bool = True,
        trim_direct_path: bool = False,
        n_order_reflections = None,
        mic_pos=None,
        source_pos=None,
    ):
        """
        n_rir: Size of each dataset sample
        n_samples_per_epoch: Number of samples to generate per epoch
        room_dims_range: 3d tuple of tuples with the range of the room dimensions in meters
        rt60_range: Reverberation time range of the room in seconds
        absorption_range: wall absorption coefficients
        sr: Sample rate of the generated RIRs
        """

        self.n_rir = n_rir
        self.room_dims_range = np.array(room_dims_range)
        self.rt60_range = np.array(rt60_range)
        self.n_samples_per_epoch = n_samples_per_epoch
        self.absorption_range = np.array(absorption_range)
        self.sr = sr
        self.trim_direct_path = trim_direct_path
        self.n_order_reflections = n_order_reflections

        self.normalize = normalize

        self.mic_pos = mic_pos
        self.source_pos = source_pos

        self.simulator = RirSimulator(self.sr, backend=backend, n_order_reflections=self.n_order_reflections)

    def __len__(self):
        return self.n_samples_per_epoch

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError("Index out of range")

        labels = get_random_room_config(
            self.room_dims_range, self.rt60_range, self.absorption_range
        )

        for key in labels:
            if isinstance(labels[key], np.ndarray):
                labels[key] = torch.from_numpy(labels[key]).float()

        rir = self.simulator(labels)

        conditioner = torch.cat([
            labels["room_dims"],
            labels["source_pos"],
            labels["mic_pos"],
            torch.Tensor([labels["rt60"]])
        ])
        
        rir = format_rir(rir, labels, self.n_rir, self.trim_direct_path)

        if self.normalize:
            # Normalize the RIR using the maximum absolute value
            rir = rir / torch.max(torch.abs(rir))


        out = {"rir": rir, "conditioner": conditioner,
               "labels": labels}

        return out


def get_random_room_config(
    room_dims_range: tuple = ((3, 10), (3, 10), (2, 4)),
    rt60_range: tuple = (0.2, 1.2),
    absorption_range: tuple = (0.5, 1),
    mic_pos=None, source_pos=None,
    cat=False
    ):    
    # 1. Generate random room

    if isinstance(room_dims_range[0], tuple):
        room_dims = np.array([
            np.random.uniform(*room_dims_range[i])
            for i in range(len(room_dims_range))
        ])
    else:
        # If the room_dims_range is not a tuple of tuples, then it is an array.
        room_dims = np.array(room_dims_range)

    rt60 = np.array([np.random.uniform(*rt60_range)])
    
    # 2. Generate random source and mic positions
    if mic_pos is not None:
        mic_pos = np.array(mic_pos)
    else:
        mic_pos = np.random.uniform(0, 1, 3) * room_dims
    
    if source_pos is not None:
        source_pos = np.array(source_pos)
    else:
        source_pos = np.random.uniform(0, 1, 3) * room_dims

    # 2.1. Make sure the mic is not too close to the source
    while np.linalg.norm(mic_pos - source_pos) < 0.3 * np.linalg.norm(room_dims):
        mic_pos = np.random.uniform(0, 1, 3) * room_dims

    out = [room_dims, rt60, source_pos, mic_pos]

    if cat:
        out = np.concatenate(out, axis=0)
        out = torch.from_numpy(out).float()
    else:
        # return a dictionary
        out = {
            "room_dims": room_dims,
            "rt60": rt60[0],
            "source_pos": source_pos,
            "mic_pos": mic_pos,
        }

    return out


def save_rir_dataset(
    dataset_path: str,
    n_dataset: int,
    room_dims_range: tuple = ((3, 10), (3, 10), (2, 4)),
    rt60_range: tuple = (0.2, 1.2),
    absorption_range: tuple = (0.5, 1),
    sr: float = 16000,
    backend: str = "auto",
    n_order_reflections = None,
    c=343,
    mic_pos=None,
    source_pos=None,):
    """Save a random room impulse response dataset to disk.

    params:
        dataset_path: Path to the dataset folder
        n_dataset: Dataset size
        room_dims_range: 3d tuple of tuples with the range of the room dimensions in meters
        rt60_range: Reverberation time range of the room in seconds
        absorption_range: wall absorption coefficients
        sr: Sample rate of the generated RIRs
        backend: "gpuRIR" or "pyroomacoustics",
        n_order_reflections: Number of order reflections to simulate
        c: Speed of sound in m/s
    """

    if isinstance(room_dims_range[0], tuple):
        max_room = np.array([r[1] for r in room_dims_range])
    else:
        # If the room_dims_range is not a tuple of tuples, then it is an array.
        max_room = np.array(room_dims_range)

    max_room_diag = np.linalg.norm(max_room)

    n_rir = rt60_range[-1] + max_room_diag/c
    n_rir = int(n_rir * sr)

    # Number of samples is the RT60 in seconds times the sample rate plus the propagation from the room diagonal

    dataset = RandomRirDataset(
        n_rir=n_rir,
        n_samples_per_epoch=n_dataset,
        room_dims_range=room_dims_range,
        rt60_range=rt60_range,
        absorption_range=absorption_range,
        sr=sr,
        backend=backend,
        normalize=False,
        n_order_reflections=n_order_reflections,
        mic_pos=mic_pos,
        source_pos=source_pos,
    )

    os.makedirs(dataset_path, exist_ok=True) 

    for i, d in tqdm(enumerate(dataset), total=len(dataset)):
        audio = d["rir"]
        label = d["labels"]

        rir_path = os.path.join(dataset_path, f"rir_{i}.wav")
        label_path = os.path.join(dataset_path, f"label_{i}.pt")

        sf.write(rir_path, audio, sr)

        torch.save(label, label_path)
    
    print(f"Dataset saved to {dataset_path}")