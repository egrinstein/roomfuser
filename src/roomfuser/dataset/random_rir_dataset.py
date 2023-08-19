import numpy as np
import torch

from torch.utils.data import Dataset

BACKEND = ""

try:
    import pyroomacoustics as pra
    BACKEND = "pyroomacoustics"
except:
    pass

try:
    import gpuRIR
    BACKEND = "gpuRIR"
except:
    pass

if BACKEND == "":
    raise ImportError("No backend found. Install gpuRIR or pyroomacoustics.")


class RandomRirDataset(Dataset):
    """Generate a random room impulse response dataset."""

    def __init__(
        self,
        n_rir: int,
        n_samples_per_epoch: int,
        room_dims_range: tuple = ((3, 10), (3, 10), (2, 4)),
        rt60_range: tuple = (0.2, 0.8),
        absorption_range: tuple = (0.5, 1),
        sr: float = 16000,
        cat_labels: bool = True,
        backend: str = "auto",
    ):
        """
        n_rir: Size of each dataset sample
        n_samples_per_epoch: Number of samples to generate per epoch
        room_dims_range: 3d tuple of tuples with the range of the room dimensions in meters
        rt60_range: Reverberation time range of the room in seconds
        absorption_range: wall absorption coefficients
        sr: Sample rate of the generated RIRs
        cat_labels: whether to concatenate labels into a torch tensor or return them as a dict
        """

        self.n_rir = n_rir
        self.room_dims_range = np.array(room_dims_range)
        self.rt60_range = np.array(rt60_range)
        self.n_samples_per_epoch = n_samples_per_epoch
        self.absorption_range = np.array(absorption_range)
        self.sr = sr
        self.cat_labels = cat_labels

        if backend == "auto":
            self.backend = BACKEND
        else:
            self.backend = backend
        
        super().__init__()

    def __len__(self):
        return self.n_samples_per_epoch

    def __getitem__(self, idx):
        room_dims, rt60, source_pos, mic_pos = get_random_room_config(
            self.room_dims_range, self.rt60_range, self.absorption_range
        )

        if self.backend == "gpuRIR":
            rir = self.simulate_rir_gpurir(room_dims, rt60[0], self.absorption_range, source_pos, mic_pos)
        elif self.backend == "pyroomacoustics":
            rir = self.simulate_rir_pyroomacoustics(room_dims, rt60[0], source_pos, mic_pos)
        else:
            raise NotImplementedError("Unknown backend: {}".format(self.backend))

        if self.cat_labels:
            conditioner = np.concatenate(
                (room_dims, source_pos, mic_pos, rt60),
                axis=0
            )
            conditioner = torch.from_numpy(conditioner).float()
        else:
            conditioner = {
                "room_dims": torch.from_numpy(room_dims),
                "rt60": torch.from_numpy(rt60),
                # "absorption": torch.from_numpy(absorption),
                "source_pos": torch.from_numpy(source_pos),
                "mic_pos": torch.from_numpy(mic_pos),
            }
        
        max_len = min(rir.shape[-1], self.n_rir)
        # Pad with zeros in the end if the RIR is too short, or truncate if it's too long
        rir = torch.nn.functional.pad(rir, (0, self.n_rir - max_len))[:self.n_rir]

        return {"audio": rir, "conditioner": conditioner}
   
    def simulate_rir_gpurir(self, room_dims, rt60, absorption, source_pos, mic_pos):
        if rt60 == 0:
            Tdiff = 0.1
            Tmax = 0.1
            nb_img = [1, 1, 1]
        else:
            Tdiff = gpuRIR.att2t_SabineEstimator(
                12, rt60
            )  # Use ISM until the RIRs decay 12dB
            Tmax = gpuRIR.att2t_SabineEstimator(
                40, rt60
            )  # Use diffuse model until the RIRs decay 40dB
            if rt60 < 0.15:
                Tdiff = Tmax  # Avoid issues with too short RIRs
            nb_img = gpuRIR.t2n(Tdiff, room_dims)

        # TODO: This will generate random absorption coefficients for each surface
        #       We should probably use this as a conditioning input to the model
        absorption = np.array([
            np.random.uniform(*absorption)
            for i in range(6) # 6: 6 surfaces
        ])

        beta = gpuRIR.beta_SabineEstimation(room_dims, rt60, absorption)

        rir = gpuRIR.simulateRIR(
            room_dims,
            beta,
            source_pos[np.newaxis, :],
            mic_pos[np.newaxis, :],
            nb_img,
            Tmax,
            self.sr,
            Tdiff=Tdiff,
        )

        return torch.from_numpy(rir[0, 0]).float()

    def simulate_rir_pyroomacoustics(self, room_dims, rt60, source_pos, mic_pos):
        e_absorption, max_order = pra.inverse_sabine(rt60, room_dims)

        room = pra.ShoeBox(
            room_dims,
            fs=self.sr,
            materials=pra.Material(e_absorption),
            max_order=max_order,
        )

        room.add_source(source_pos)
        room.add_microphone(mic_pos)

        room.compute_rir()

        rir = torch.from_numpy(room.rir[0][0]).float()

        return rir


def get_random_room_config(
    room_dims_range: tuple = ((3, 10), (3, 10), (2, 4)),
    rt60_range: tuple = (0.2, 0.8),
    absorption_range: tuple = (0.5, 1), cat=False
    ):    
    # 1. Generate random room
    room_dims = np.array([
        np.random.uniform(*room_dims_range[i])
        for i in range(len(room_dims_range))
    ])

    rt60 = np.array([np.random.uniform(*rt60_range)])
    
    # 2. Generate random source and mic positions
    source_pos = np.random.uniform(0, 1, 3) * room_dims
    mic_pos = np.random.uniform(0, 1, 3) * room_dims
    # 2.1. Make sure the mic is not too close to the source
    while np.linalg.norm(mic_pos - source_pos) < 0.3 * np.linalg.norm(room_dims):
        mic_pos = np.random.uniform(0, 1, 3) * room_dims

    out = [room_dims, rt60, source_pos, mic_pos]

    if cat:
        out = np.concatenate(out, axis=0)
        out = torch.from_numpy(out).float()

    return out