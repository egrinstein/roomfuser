# Visualizations of the diffusion process

import os
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch

from matplotlib.animation import FuncAnimation, PillowWriter
from tqdm import trange

from roomfuser.params import params
from roomfuser.dataset import RirDataset, FastRirDataset
from roomfuser.models import load_model
from roomfuser.inference import predict_batch


def plot_diffusion(steps: np.array, target: np.array = None, envelopes=None, sr: int = 16000, rt60=None,
                   low_ord_input: np.array = None):
    """Plot the diffusion process.

    Args:
        steps (np.array): A numpy array of shape (n_steps, n_rir).

    """

    # Repeat the last step the same number of times as the first steps
    # This is to make the animation stop at the last step

    n_steps = steps.shape[0]

    last_step = steps[-1:]
    last_step = np.repeat(last_step, steps.shape[0], axis=0)
    steps = np.concatenate([steps, last_step], axis=0)

    fig, ax = plt.subplots()
    ax.set_xlim(0, steps.shape[1])
    ax.set_ylim(-1, 1)
    ax.set_xlabel("Tap number")
    ax.set_ylabel("Value")
    ax.set_title("Diffusion Process")

    line, = ax.plot([], [], lw=2)
    label = None
    if isinstance(rt60, torch.Tensor):
        rt60 = rt60.item()
    if rt60:
        label = "RT60={:.2f}".format(rt60)

    if target is not None:
        mse = np.mean((target - steps[-1])**2)
        ax.plot(target, label="Target (MSE=%.0E" %mse, alpha=0.2, color="r")

    # Plot the envelope of the RIR based on the RT60
    if envelopes is not None:
        ax.plot(envelopes.numpy())

    if low_ord_input is not None:
        ax.plot(low_ord_input, label="Low order input", alpha=0.2, linestyle="--", color="g")

    if target is not None or label is not None:
        ax.legend(loc='upper right')

    def init():
        line.set_data([], [])
        return (line,)

    def animate(i):
        x = np.arange(steps.shape[1])
        y = steps[i]
        line.set_data(x, y)
        n_step = min(i, n_steps)
        ax.set_title(f"Diffusion Process (Step {n_step})")
        return (line,)

    anim = FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=steps.shape[0],
        interval=10,
        blit=True,
    )

    plt.close()
    return anim


def generate_random_rir():
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

    model = load_model(params)
    
    model.load_state_dict(params.model_path)
    model.device = torch.device("cpu")
    model.eval()

    animations_dir = "logs/animations"
    os.makedirs(animations_dir, exist_ok=True)

    noise_prior = model.noise_scheduler.noise_prior
    for i in trange(params.n_viz_samples):
        # i = np.random.randint(len(rir_dataset))
        d = rir_dataset[i]
        target_audio = d["rir"]
        conditioner = d["conditioner"]
        target_labels = d["labels"]
        
        prior_mean = noise_prior.get_mean([target_labels], target_audio.unsqueeze(0))[0]
        # Generate audio
        audio, sr = predict_batch(model, conditioner=conditioner.unsqueeze(0),
                              batch_size=1, return_steps=True, labels=[target_labels],
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

        anim = plot_diffusion(audio, target_audio,
                              rt60=target_labels["rt60"],
                              low_ord_input=None)#prior_mean)
        anim.save(f"{animations_dir}/diffusion_{i}.gif", writer=PillowWriter(fps=10))

        # Save audio
        # sf.write(f"{animations_dir}/audio_{i}.wav", audio[-1], sr)


if __name__ == "__main__":
    generate_random_rir()
