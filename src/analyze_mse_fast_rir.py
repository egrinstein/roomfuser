# This script compares the MSE of the RIRs generated using the FAST-RIR method,
# which are stored in the `fast_rir_gan_dataset_path` directory, with the RIRs
# generated using DAS simulator, which are stored in the `fast_rir_dataset_path`

import os
import numpy as np
import torch
import soundfile as sf

from tqdm import tqdm

from roomfuser.params import params


def compare_mse(rir_path_1, rir_path_2):
    rir_1, sr_1 = sf.read(rir_path_1)
    rir_2, sr_2 = sf.read(rir_path_2)
    assert sr_1 == sr_2

    n = min(len(rir_1), len(rir_2))
    rir_1 = rir_1[:n]
    rir_2 = rir_2[:n]

    mse = np.mean((rir_1 - rir_2)**2)
    return mse

def analyze_mse():
    dir_1 = params.fast_rir_gan_dataset_path
    dir_2 = os.path.join(params.fast_rir_dataset_path, "RIR")

    mse = 0

    files = [f for f in os.listdir(dir_1) if f.endswith(".wav")]
    bar = tqdm(files)
    for i, f in enumerate(bar):
        mse += compare_mse(os.path.join(dir_1, f), os.path.join(dir_2, f))

        bar.set_description(f"MSE: {mse / (i + 1)}")

    mse /= len(files)
    print(f"MSE: {mse}")

if __name__ == "__main__":
    analyze_mse()
