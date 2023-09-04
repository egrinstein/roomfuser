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
from roomfuser.noise_scheduler import NoiseScheduler
from roomfuser.dataset.simulator import RirSimulator


def plot_diffusion(steps: np.array, target, labels, priors: NoiseScheduler = None):
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

    rt60 = labels[0]["rt60"]
    if isinstance(rt60, torch.Tensor):
        rt60 = rt60.item()
    if rt60:
        label = "RT60={:.2f}".format(rt60)

    ax.plot(target, label="Target" + label, alpha=0.5)
        
    # Plot the envelope of the RIR based on the RT60
    
    if priors is not None:
        dummy = torch.zeros(1, steps.shape[1])
        mean = priors.get_mean(dummy, labels)[0]
        var = priors.get_variance(dummy, labels)[0]

        ax.plot(mean, label="Prior Mean", alpha=0.5)
        ax.plot(var, label="Prior Var.", alpha=0.5)
       

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
        )
    else:
        rir_dataset = RirDataset(
            params.roomfuser_dataset_path,
            n_rir=params.rir_len,
            trim_direct_path=params.trim_direct_path,
            scaler_path=params.roomfuser_scaler_path,
            frequency_response=params.frequency_response,
        )
    
    animations_dir = "logs/animations"
    os.makedirs(animations_dir, exist_ok=True)

    noise_schedule = params.training_noise_schedule

    simulator = RirSimulator(params.sample_rate, params.rir_backend, params.n_rir_order_reflection,
                             params.trim_direct_path, n_rir=params.rir_len)

    inference_noise_schedule = params.inference_noise_schedule if params.fast_sampling else None
    noise_scheduler = NoiseScheduler(
        noise_schedule, not params.trim_direct_path, params.prior_variance_mode,
        mean_mode=params.prior_mean_mode,
        rir_simulator=simulator, inference_noise_schedule=inference_noise_schedule,
        frequency_response=params.frequency_response)

    for i in trange(params.n_viz_samples):
        #i = np.random.randint(len(rir_dataset))
        d = rir_dataset[i]
        target_audio = d["rir"]
        target_labels = d["labels"]
        
        # Generate audio
        noisy_audio = noise_scheduler.get_all_noisy_steps(target_audio.unsqueeze(0),
                                                          [target_labels])[0]
        # Plot diffusion process
        anim = plot_diffusion(noisy_audio, target_audio, labels=[target_labels], priors=noise_scheduler)
        anim.save(f"{animations_dir}/diffusion_{i}.gif", writer=PillowWriter(fps=10))

        # Save audio
        # sf.write(f"{animations_dir}/audio_{i}.wav", audio[-1], sr)


if __name__ == "__main__":
    generate_random_rir()
