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
import random
import torch

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from .random_sinusoid_dataset import RandomSinusoidDataset
from .random_rir_dataset import RandomRirDataset, RirDataset


class Collator:
    def __init__(self, params):
        self.params = params

    def collate(self, minibatch):
        for record in minibatch:
            if self.params.unconditional:
                # Filter out records that aren't long enough.
                if len(record["audio"]) < self.params.rir_len:
                    del record["conditioner"]
                    del record["audio"]
                    continue

                # Crop bigger records to rir_len.
                start = random.randint(
                    0, record["audio"].shape[-1] - self.params.rir_len
                )
                end = start + self.params.rir_len
                record["audio"] = record["audio"][start:end]
                record["audio"] = np.pad(
                    record["audio"],
                    (0, (end - start) - len(record["audio"])),
                    mode="constant",
                )

        audio = np.stack([record["audio"] for record in minibatch if "audio" in record])
        if self.params.unconditional:
            return {
                "audio": torch.from_numpy(audio),
                "conditioner": None,
            }
        spectrogram = np.stack(
            [record["conditioner"] for record in minibatch if "conditioner" in record]
        )
        return {
            "audio": torch.from_numpy(audio),
            "conditioner": torch.from_numpy(spectrogram),
        }


def from_path(data_dirs, params, is_distributed=False):
    if params.dataset_name == "sinusoid":
        dataset = RandomSinusoidDataset(
            n_sample=params.rir_len, n_samples_per_epoch=params.n_samples_per_epoch
        )
    elif params.dataset_name == "rir":
        if os.path.exists(params.dataset_path):
            dataset = RirDataset(
                params.dataset_path,
                n_rir=params.rir_len
            )
        else:
            dataset = RandomRirDataset(
                n_rir=params.rir_len, n_samples_per_epoch=params.n_samples_per_epoch,
                backend=params.rir_backend
            )
    else:
        raise NotImplementedError(f"Unknown dataset: {params.dataset_name}")

    return DataLoader(
        dataset,
        batch_size=params.batch_size,
        collate_fn=Collator(params).collate,
        shuffle=not is_distributed,
        num_workers=os.cpu_count() if params.n_workers is None else params.n_workers,
        sampler=DistributedSampler(dataset) if is_distributed else None,
        pin_memory=True,
        drop_last=True,
    )
