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

import torch

from roomfuser.utils import MinMaxScaler, dict_to_device


def predict_batch(model, conditioner=None, batch_size=1,
                  return_steps=False, labels=None, scaler: MinMaxScaler=None,
                  frequency_response=False):
    
    inference_config = model.noise_scheduler.get_inference_config()
    alpha = inference_config["alpha"]
    alpha_cum = inference_config["alpha_cum"]
    beta = inference_config["beta"]
    T = inference_config["T"]

    if not frequency_response:
        audio_shape = (batch_size, model.params.rir_len)
    else:
        audio_shape = (batch_size, 2, model.params.rir_len // 2 + 1)

    with torch.no_grad():
        if labels is not None:
            for label in labels:
                dict_to_device(label, model.device)
        if conditioner is not None:
            conditioner = conditioner.to(model.device)

        audio = torch.randn(*audio_shape, device=model.device)
        audio = model.noise_scheduler.get_base_noise(audio, labels)

        if return_steps:
            steps = [] #[audio]

        for n in range(len(alpha) - 1, -1, -1):
            c1 = 1 / alpha[n] ** 0.5
            c2 = beta[n] / (1 - alpha_cum[n]) ** 0.5
            model_output = model(
                audio, torch.tensor([T[n]], device=model.device), conditioner
            ).squeeze(1)
            audio = c1 * (audio - c2 * model_output)

            if n > 0:
                sigma = (
                    (1.0 - alpha_cum[n - 1]) / (1.0 - alpha_cum[n]) * beta[n]
                ) ** 0.5
                # TODO: The variance and mean are constant, so they could be computed outside the loop
                noise = model.noise_scheduler.get_base_noise(audio, labels)
                audio += sigma * noise
            # audio = torch.clamp(audio, -1.0, 1.0)

            # This last part is only used to produce the audio output, but the loop should use the unprocessed signal
            # TODO: This should be done outside the loop
            if not frequency_response:
                out = audio
            else:
                # In this case, this model is using the frequency response: apply the IDFT to get the audio back
                out = torch.complex(audio[:, 0], audio[:, 1])
                out = torch.fft.irfft(out)
            
            if scaler is not None:
                out = scaler.descale(out)
            
            out = out / torch.max(torch.abs(out), dim=1, keepdim=True)[0]
            # out = torch.clamp(out, -1.0, 1.0)

            if return_steps:
                steps.append(out)

    if return_steps:
        out = torch.stack(steps, dim=1)

    return out, model.params.sample_rate
