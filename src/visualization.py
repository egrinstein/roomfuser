# Visualizations of the diffusion process

import os
import matplotlib.pyplot as plt
import numpy as np
import torch

from matplotlib.animation import FuncAnimation, PillowWriter
from tqdm import trange

from roomfuser.params import params
from roomfuser.dataset import RirDataset, FastRirDataset
from roomfuser.model import DiffWave
from roomfuser.inference import predict_batch


def plot_diffusion(steps: np.array, target: np.array = None):
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

    if target is not None:
        ax.plot(target, label="Target", alpha=0.5)
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
        )
    else:
        rir_dataset = RirDataset(
            params.gpu_rir_dataset_path,
            n_rir=params.rir_len,
        )

    model = DiffWave(params)
    
    model.load_state_dict(params.model_path)
    model.device = torch.device("cpu")
    model.eval()

    animations_dir = "logs/animations"
    os.makedirs(animations_dir, exist_ok=True)

    for i in trange(params.n_viz_samples):
        d = rir_dataset[i]
        target_audio = d["audio"].numpy()
        target_label = d["conditioner"]
        
        # Generate audio
        audio, sr = predict_batch(model, conditioner=target_label.unsqueeze(0),
                              n_samples=1, fast_sampling=False, return_steps=True)
        audio = audio[0].numpy()

        # Plot diffusion process
        anim = plot_diffusion(audio, target_audio)
        anim.save(f"{animations_dir}/diffusion_{i}.gif", writer=PillowWriter(fps=10))


if __name__ == "__main__":
    generate_random_rir()