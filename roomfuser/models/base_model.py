import torch
import torch.nn as nn

from roomfuser.noise_scheduler import NoiseScheduler
from roomfuser.dataset.simulator import RirSimulator


class BaseModel(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params

        self._init_noise_manager(params)

    def _init_noise_manager(self, params):
        "Initialize the noise manager, which handles the noise schedule and noise prior."

        simulator = None
        if params.prior_mean_mode == "low_ord_rir":
            if params.dataset_name == "roomfuser":
                scaler_path = params.roomfuser_scaler_path
            elif params.dataset_name == "fast_rir":
                scaler_path = params.fast_rir_scaler_path
                
            simulator = RirSimulator(params.sample_rate, params.rir_backend, params.n_rir_order_reflection,
                            params.trim_direct_path, n_rir=params.rir_len, scaler_path=scaler_path)

        inference_noise_schedule = params.inference_noise_schedule if params.fast_sampling else None
        self.noise_scheduler = NoiseScheduler(
            params.training_noise_schedule,
            not params.trim_direct_path,
            params.prior_variance_mode,
            params.prior_mean_mode,
            rir_simulator=simulator,
            inference_noise_schedule=inference_noise_schedule,
            frequency_response=params.frequency_response
        )

    def load_state_dict(self, path, strict=True):
        if isinstance(path, str):
            model_params = torch.load(path, map_location="cpu")["model"]
        else:
            model_params = path
        super().load_state_dict(model_params, strict=strict)
        
