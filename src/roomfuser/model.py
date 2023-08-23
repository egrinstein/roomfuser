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
import torch
import torch.nn as nn
import torch.nn.functional as F

from math import sqrt


Linear = nn.Linear
ConvTranspose2d = nn.ConvTranspose2d


def Conv1d(*args, **kwargs):
    layer = nn.Conv1d(*args, **kwargs)
    nn.init.kaiming_normal_(layer.weight)
    return layer


@torch.jit.script
def silu(x):
    return x * torch.sigmoid(x)


class DiffusionEmbedding(nn.Module):
    def __init__(self, max_steps):
        super().__init__()
        self.register_buffer(
            "embedding", self._build_embedding(max_steps), persistent=False
        )
        self.projection1 = Linear(128, 512)
        self.projection2 = Linear(512, 512)

    def forward(self, diffusion_step):
        if diffusion_step.dtype in [torch.int32, torch.int64]:
            x = self.embedding[diffusion_step]
        else:
            x = self._lerp_embedding(diffusion_step)
        x = self.projection1(x)
        x = silu(x)
        x = self.projection2(x)
        x = silu(x)
        return x

    def _lerp_embedding(self, t):
        low_idx = torch.floor(t).long()
        high_idx = torch.ceil(t).long()
        low = self.embedding[low_idx]
        high = self.embedding[high_idx]
        return low + (high - low) * (t - low_idx)

    def _build_embedding(self, max_steps):
        steps = torch.arange(max_steps).unsqueeze(1)  # [T,1]
        dims = torch.arange(64).unsqueeze(0)  # [1,64]
        table = steps * 10.0 ** (dims * 4.0 / 63.0)  # [T,64]
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table


class ResidualBlock(nn.Module):
    def __init__(
        self,
        n_conditioner,
        residual_channels,
        dilation,
        condition_time_idx=True,
    ):
        """
        :param n_conditioner: size of conditioner
        :param residual_channels: audio conv
        :param dilation: audio conv dilation
        """
        super().__init__()
        self.dilated_conv = Conv1d(
            residual_channels,
            2 * residual_channels,
            3,
            padding=dilation,
            dilation=dilation,
        )
        self.diffusion_projection = Linear(512, residual_channels)

        self.condition_time_idx = condition_time_idx
        if condition_time_idx:
            n_conditioner += 1
       
        self.conditioner_projection_fc = Linear(
            n_conditioner, 2 * residual_channels
        )

        self.output_projection = Conv1d(residual_channels, 2 * residual_channels, 1)

    def forward(self, x, diffusion_step, conditioner=None):
        assert (conditioner is None and self.conditioner_projection_fc is None) or (
            conditioner is not None and self.conditioner_projection_fc is not None
        )

        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        y = x + diffusion_step
        y = self.dilated_conv(y)

        if self.conditioner_projection_fc is not None:  # using a conditional model
            if self.condition_time_idx:
                # add time index to conditioner
                t = (
                    torch.linspace(0.0, 1.0, x.shape[-1], device=x.device)
                    .unsqueeze(0)
                    .repeat(x.shape[0], 1)
                )
                t = t.unsqueeze(1)
                conditioner = conditioner.unsqueeze(-1).repeat(1, 1, x.shape[-1])
                conditioner = torch.cat([conditioner, t], dim=1).transpose(1, 2)

                conditioner = self.conditioner_projection_fc(conditioner).transpose(1, 2)
            else:
                conditioner = self.conditioner_projection_fc(conditioner).unsqueeze(-1)

            y += conditioner

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / sqrt(2.0), skip


class DiffWave(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.input_projection = Conv1d(1, params.residual_channels, 1)
        self.diffusion_embedding = DiffusionEmbedding(len(params.noise_schedule))

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    params.n_conditioner,
                    params.residual_channels,
                    2 ** (i % params.dilation_cycle_length),
                )
                for i in range(params.residual_layers)
            ]
        )
        self.skip_projection = Conv1d(
            params.residual_channels, params.residual_channels, 1
        )
        self.output_projection = Conv1d(params.residual_channels, 1, 1)
        nn.init.zeros_(self.output_projection.weight)


        #self.cond_norm = nn.BatchNorm1d(params.n_conditioner)

    def forward(self, audio, diffusion_step, conditioner=None):
        x = audio.unsqueeze(1)
        x = self.input_projection(x)
        x = F.relu(x)

        diffusion_step = self.diffusion_embedding(diffusion_step)

        # if conditioner is not None:
        #     conditioner = self.cond_norm(conditioner)

        skip = None
        for layer in self.residual_layers:
            x, skip_connection = layer(x, diffusion_step, conditioner)
            skip = skip_connection if skip is None else skip_connection + skip

        x = skip / sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.relu(x)
        x = self.output_projection(x)
        return x

    def load_state_dict(self, path, strict=False):
        if isinstance(path, str):
            model_params = torch.load(path, map_location="cpu")["model"]
        else:
            model_params = path
        super().load_state_dict(model_params, strict=strict)
        
