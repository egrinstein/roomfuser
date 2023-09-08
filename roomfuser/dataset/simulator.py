import numpy as np
import torch

from roomfuser.utils import format_rir

BACKEND = ""

try:
    import pyroomacoustics as pra
    BACKEND = "pyroomacoustics"
except:
    pass

if torch.cuda.is_available():
    try:
        import gpuRIR
        BACKEND = "gpuRIR"
    except:
        pass

if BACKEND == "":
    raise ImportError("No backend found. Install gpuRIR or pyroomacoustics.")


class RirSimulator:
    def __init__(self, sr=16000, backend="auto", n_order_reflections=None, trim_direct_path=False, n_rir=None,
                 normalize=True):
        self.sr = sr
        self.n_order_reflections = n_order_reflections        
        self.trim_direct_path = trim_direct_path
        self.n_rir = n_rir
        self.normalize = normalize

        if backend == "auto":
            self.backend = BACKEND
        else:
            self.backend = backend
        
        if self.backend not in ["gpuRIR", "pyroomacoustics"]:
            raise ValueError("Unknown backend: {}".format(self.backend))

    def __call__(self, config_dict):
        keys = list(config_dict.keys())
        device = config_dict[keys[0]].device
        dtype = config_dict[keys[0]].dtype

        room_dims = config_dict["room_dims"].cpu().numpy().astype(np.double)
        source_pos = config_dict["source_pos"].cpu().numpy().astype(np.double)
        mic_pos = config_dict["mic_pos"].cpu().numpy().astype(np.double)

        if isinstance(config_dict["rt60"], torch.Tensor):
            rt60 = config_dict["rt60"].cpu().numpy().astype(np.double)[0]
        else:
            rt60 = config_dict["rt60"]

        if self.backend == "gpuRIR":
            rir = simulate_rir_gpurir(
                room_dims, rt60,
                source_pos, mic_pos,
                n_order_reflections=self.n_order_reflections,
                sr=self.sr)
        elif self.backend == "pyroomacoustics":
            rir = simulate_rir_pyroomacoustics(
                room_dims, rt60, source_pos, mic_pos, n_order_reflections=self.n_order_reflections,
                sr=self.sr)

        rir = torch.from_numpy(rir).to(device=device, dtype=dtype)

        rir = format_rir(rir, config_dict, self.n_rir, self.trim_direct_path)

        if self.normalize:
            rir = rir / torch.max(torch.abs(rir))

        return rir


def simulate_rir_gpurir(room_dims, rt60, source_pos, mic_pos, n_order_reflections=None, sr=16000):
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
        0.5 #np.random.uniform(*absorption)
        for i in range(6) # 6: 6 surfaces
    ])

    beta = gpuRIR.beta_SabineEstimation(room_dims, rt60, absorption)
    
    if n_order_reflections is not None:
        nb_img = [n_order_reflections]*3

    rir = gpuRIR.simulateRIR(
        room_dims,
        beta,
        source_pos[np.newaxis, :],
        mic_pos[np.newaxis, :],
        nb_img,
        Tmax,
        sr,
        Tdiff=Tdiff,
    )

    return rir[0, 0]

def simulate_rir_pyroomacoustics(room_dims, rt60, source_pos, mic_pos, n_order_reflections=None, sr=16000):
    e_absorption, max_order = pra.inverse_sabine(rt60, room_dims)

    if n_order_reflections is not None:
        max_order = n_order_reflections

    room = pra.ShoeBox(
        room_dims,
        fs=sr,
        materials=pra.Material(e_absorption),
        max_order=max_order,
    )

    room.add_source(source_pos)
    room.add_microphone(mic_pos)

    room.compute_rir()

    rir = room.rir[0][0]

    return rir
