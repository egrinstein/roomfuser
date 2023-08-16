import gpuRIR
import numpy as np

from torch.utils.data import Dataset


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
        cat_labels: bool = True
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
        
        super().__init__()

    def __len__(self):
        return self.n_samples_per_epoch

    def __getitem__(self, idx):
        audio, conditioner = self.get_random_rir()
        
        return {"audio": audio[:self.n_rir], "conditioner": conditioner}

    def get_random_rir(self):
        # 1. Generate random room
        room_dims = [
            np.random.uniform(*self.room_dims_range[i])
            for i in range(len(self.room_dims_range))
        ]

        rt60 = np.random.uniform(*self.rt60_range)
        absorption = [
            np.random.uniform(*self.absorption_range)
            for i in range(6) # 6: 6 surfaces
        ]
        beta = gpuRIR.beta_SabineEstimation(room_dims, rt60, absorption)

        # 2. Generate random source and mic positions
        source_pos = np.random.uniform(0, 1, 3) * room_dims
        mic_pos = np.random.uniform(0, 1, 3) * room_dims
        # 2.1. Make sure the mic is not too close to the source
        while np.linalg.norm(mic_pos - source_pos) < 0.3 * np.linalg.norm(room_dims):
            mic_pos = np.random.uniform(0, 1, 3) * room_dims

        rir = self.simulate_rir(room_dims, rt60, absorption, beta, source_pos, mic_pos)
        
        if self.cat_labels:
            conditioner = np.concatenate(
                (room_dims, absorption, source_pos, mic_pos, [rt60]),
                axis=0
            )
            conditioner = torch.from_numpy(conditioner)
        else:
            conditioner = {
                "room_dims": room_dims,
                "rt60": rt60,
                "absorption": absorption,
                "beta": beta,
                "source_pos": source_pos,
                "mic_pos": mic_pos,
            }

        return rir, conditioner
        
    def simulate_rir(self, room_dims, rt60, absorption, beta, source_pos, mic_pos):
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

        return rir[0, 0]
