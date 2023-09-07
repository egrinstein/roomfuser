import matplotlib.pyplot as plt
import numpy as np

from roomfuser.params import params


def main():
    training_schedule = params["training_noise_schedule"]
    inference_schedule = params["inference_noise_schedule"]

    # Plot the training schedule
    fig, axs = plt.subplots(2, 1, sharex=False)
    axs[0].plot(training_schedule)
    axs[0].set_title("Training noise schedule")
    axs[0].set_ylabel("Noise level")
    axs[0].set_xlabel("Training step")

    # Plot the inference schedule
    axs[1].plot(inference_schedule)
    axs[1].set_title("Inference noise schedule")
    axs[1].set_ylabel("Noise level")
    axs[1].set_xlabel("Inference step")

    plt.tight_layout()
    plt.savefig("noise_schedule.png")


if __name__ == "__main__":
    main()