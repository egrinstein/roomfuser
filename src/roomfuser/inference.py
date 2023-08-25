# Copyright 2020 LMNT, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import os
import torch
import torchaudio

from argparse import ArgumentParser

from roomfuser.params import AttrDict, params as base_params
from roomfuser.model import DiffWave


models = {}


def predict_batch(model, conditioner=None, n_samples=1, fast_sampling=False,
                  return_steps=False, envelopes=None):
    with torch.no_grad():
        # Change in notation from the DiffWave paper for fast sampling.
        # DiffWave paper -> Implementation below
        # --------------------------------------
        # alpha -> talpha
        # beta -> training_noise_schedule
        # gamma -> alpha
        # eta -> beta
        training_noise_schedule = np.array(model.params.noise_schedule)
        inference_noise_schedule = (
            np.array(model.params.inference_noise_schedule)
            if fast_sampling
            else training_noise_schedule
        )

        talpha = 1 - training_noise_schedule
        talpha_cum = np.cumprod(talpha)

        beta = inference_noise_schedule
        alpha = 1 - beta
        alpha_cum = np.cumprod(alpha)

        T = []
        for s in range(len(inference_noise_schedule)):
            for t in range(len(training_noise_schedule) - 1):
                if talpha_cum[t + 1] <= alpha_cum[s] <= talpha_cum[t]:
                    twiddle = (talpha_cum[t] ** 0.5 - alpha_cum[s] ** 0.5) / (
                        talpha_cum[t] ** 0.5 - talpha_cum[t + 1] ** 0.5
                    )
                    T.append(t + twiddle)
                    break
        T = np.array(T, dtype=np.float32)

        audio = torch.randn(n_samples, model.params.rir_len, device=model.device)
        if envelopes is not None: # Weight the noise by the RIR envelopes
            audio *= envelopes

        if conditioner is not None:
            conditioner = conditioner.to(model.device)
 
        if return_steps:
            steps = [audio]

        for n in range(len(alpha) - 1, -1, -1):
            c1 = 1 / alpha[n] ** 0.5
            c2 = beta[n] / (1 - alpha_cum[n]) ** 0.5
            audio = c1 * (
                audio
                - c2
                * model(
                    audio, torch.tensor([T[n]], device=model.device), conditioner
                ).squeeze(1)
            )
            if n > 0:
                noise = torch.randn_like(audio)
                sigma = (
                    (1.0 - alpha_cum[n - 1]) / (1.0 - alpha_cum[n]) * beta[n]
                ) ** 0.5
                audio += sigma * noise
            audio = torch.clamp(audio, -1.0, 1.0)
            if return_steps:
                steps.append(audio)

    if return_steps:
        audio = torch.stack(steps, dim=1)

    return audio, model.params.sample_rate


def predict(
    conditioner=None,
    model_dir=None,
    params=None,
    device=torch.device("cuda"),
    fast_sampling=False,
    envelopes=None,
):
    # Lazy load model.
    if not model_dir in models:
        if os.path.exists(f"{model_dir}/weights.pt"):
            checkpoint = torch.load(f"{model_dir}/weights.pt")
        else:
            checkpoint = torch.load(model_dir)
        model = DiffWave(AttrDict(base_params)).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        models[model_dir] = model

    model = models[model_dir]
    model.params.override(params)

    return predict_batch(model, fast_sampling, conditioner, envelopes=envelopes)


def main(args):
    audio, sr = predict(
        model_dir=args.model_dir,
        fast_sampling=args.fast,
        params=base_params,
    )
    torchaudio.save(args.output, audio.cpu(), sample_rate=sr)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="runs inference on a trained DiffWave model"
    )
    parser.add_argument(
        "model_dir",
        help="directory containing a trained model (or full path to weights.pt file)",
    )
    parser.add_argument(
        "--conditioner_path",
        "-s",
        help="path to a conditioner file",
    )
    parser.add_argument("--output", "-o", default="output.wav", help="output file name")
    parser.add_argument(
        "--fast", "-f", action="store_true", help="fast sampling procedure"
    )
    main(parser.parse_args())
