# Compute the MSE between the model and the data

# Visualizations of the diffusion process

import os
import numpy as np
import torch

from tqdm import tqdm

from roomfuser.params import params
from roomfuser.dataset import RirDataset, FastRirDataset
from roomfuser.models.diffwave import DiffWave
from roomfuser.inference import predict_batch


def analyze_mse():

    # Load dataset
    if params.dataset_name == "fast_rir":
        rir_dataset = FastRirDataset(
            params.fast_rir_dataset_path,
            n_rir=params.rir_len,
            trim_direct_path=params.trim_direct_path,
            scaler_path=params.fast_rir_scaler_path,
            frequency_response=params.frequency_response,
        )
    elif params.dataset_name == "roomfuser":
        rir_dataset = RirDataset(
            params.roomfuser_dataset_path,
            n_rir=params.rir_len,
            trim_direct_path=params.trim_direct_path,
            scaler_path=params.roomfuser_scaler_path,
            frequency_response=params.frequency_response,
        )
    scaler = rir_dataset.scaler

    # Load model
    model = DiffWave(params)
    model.load_state_dict(params.model_path)
    model.device = torch.device("cpu")
    model.eval()

    mse = 0

    bar = tqdm(rir_dataset)
    for i, d in enumerate(bar):
        #i = np.random.randint(len(rir_dataset))

        target_audio = d["rir"]
        conditioner = d["conditioner"]
        target_labels = d["labels"]
        
        # Generate audio
        audio, sr = predict_batch(model, conditioner=conditioner.unsqueeze(0),
                              batch_size=1, return_steps=False, labels=[target_labels],
                              scaler=scaler, frequency_response=params.frequency_response)
        audio = audio[0].numpy()
        if scaler is not None:
            target_audio = scaler.descale(target_audio)
        if params.frequency_response:
            target_audio = torch.complex(target_audio[0], target_audio[1])
            target_audio = torch.fft.irfft(target_audio)

        target_audio /= torch.max(torch.abs(target_audio))
        target_audio = target_audio.numpy()
        # Plot diffusion process

        mse += np.mean((audio - target_audio) ** 2)

        bar.set_description(f"MSE: {mse / (i + 1)}")

    mse /= len(rir_dataset)

if __name__ == "__main__":
    analyze_mse()
