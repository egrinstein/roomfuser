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
from .roomfuser_dataset import RandomRirDataset, RirDataset
from .fast_rir_dataset import FastRirDataset


class Collator:
    def __init__(self, params):
        self.params = params

    def collate(self, minibatch):

        output = {}
 
        output["audio"] = torch.stack([
            record["audio"] for record in minibatch])
        
        if "conditioner" in minibatch[0]:
            output["conditioner"] = torch.stack(
                [record["conditioner"] for record in minibatch]
            )
        
        if "labels" in minibatch[0]:
            output["labels"] = [
                record["labels"] for record in minibatch
            ]
        return output


def from_path(data_dirs, params, is_distributed=False):
    if params.dataset_name == "sinusoid":
        dataset = RandomSinusoidDataset(
            n_sample=params.rir_len, n_samples_per_epoch=params.n_samples_per_epoch
        )
    elif params.dataset_name == "roomfuser":
        if os.path.exists(params.roomfuser_dataset_path):
            dataset = RirDataset(
                params.roomfuser_dataset_path,
                n_rir=params.rir_len
            )
        else:
            dataset = RandomRirDataset(
                n_rir=params.rir_len, n_samples_per_epoch=params.n_samples_per_epoch,
                backend=params.rir_backend
            )
    elif params.dataset_name == "fast_rir":
        dataset = FastRirDataset(
            params.fast_rir_dataset_path,
            n_rir=params.rir_len
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
