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

import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn

from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from roomfuser.dataset import from_path
from roomfuser.dataset import RandomRirDataset, RandomSinusoidDataset
from roomfuser.model import DiffWave
from roomfuser.inference import predict_batch


def _nested_map(struct, map_fn):
    if isinstance(struct, tuple):
        return tuple(_nested_map(x, map_fn) for x in struct)
    if isinstance(struct, list):
        return [_nested_map(x, map_fn) for x in struct]
    if isinstance(struct, dict):
        return {k: _nested_map(v, map_fn) for k, v in struct.items()}
    return map_fn(struct)


class DiffWaveLearner:
    def __init__(self, model_dir, model, dataset, optimizer, params, *args, **kwargs):
        os.makedirs(model_dir, exist_ok=True)
        self.model_dir = model_dir
        self.model = model
        self.dataset = dataset
        self.optimizer = optimizer
        self.params = params
        self.autocast = torch.cuda.amp.autocast(enabled=kwargs.get("fp16", False))
        self.scaler = torch.cuda.amp.GradScaler(enabled=kwargs.get("fp16", False))
        self.step = 0
        self.is_master = True

        beta = np.array(self.params.noise_schedule)
        noise_level = np.cumprod(1 - beta)
        self.noise_level = torch.tensor(noise_level.astype(np.float32))
        self.loss_fn = nn.L1Loss()
        self.summary_writer = None

    def state_dict(self):
        if hasattr(self.model, "module") and isinstance(self.model.module, nn.Module):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
        return {
            "step": self.step,
            "model": {
                k: v.cpu() if isinstance(v, torch.Tensor) else v
                for k, v in model_state.items()
            },
            "optimizer": {
                k: v.cpu() if isinstance(v, torch.Tensor) else v
                for k, v in self.optimizer.state_dict().items()
            },
            "params": dict(self.params),
            "scaler": self.scaler.state_dict(),
        }

    def load_state_dict(self, state_dict):
        if hasattr(self.model, "module") and isinstance(self.model.module, nn.Module):
            self.model.module.load_state_dict(state_dict["model"])
        else:
            self.model.load_state_dict(state_dict["model"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.scaler.load_state_dict(state_dict["scaler"])
        self.step = state_dict["step"]

    def save_to_checkpoint(self, n_epoch, filename="weights"):
        save_basename = f"{filename}-{self.step}.pt"
        save_name = f"{self.model_dir}/{save_basename}"
        torch.save(self.state_dict(), save_name)

    def restore_from_checkpoint(self, filename="weights"):
        try:
            checkpoint = torch.load(f"{self.model_dir}/{filename}.pt", map_location="cpu")
            self.load_state_dict(checkpoint)
            print(f"Restored checkpoint from {self.model_dir}/{filename}.pt")
            return True
        except FileNotFoundError:
            return False

    def train(self):
        device = next(self.model.parameters()).device
        best_loss = float("inf")
        for n_epoch in range(self.params.n_epochs):
            progress_bar = tqdm(total=len(self.dataset))
            progress_bar.set_description(f"Epoch: {n_epoch}")
            epoch_loss = 0.0
            for batch in self.dataset:
                batch = _nested_map(
                    batch,
                    lambda x: x.to(device) if isinstance(x, torch.Tensor) else x,
                )
                loss = self.train_step(batch)
                if torch.isnan(loss).any():
                    raise RuntimeError(f"Detected NaN loss at step {self.step}.")
                self.step += 1
                progress_bar.update(1)
                progress_bar.set_postfix(loss=loss.item())
                epoch_loss += loss.item()/len(self.dataset)
            progress_bar.close()

            if self.is_master:
                if n_epoch % self.params.n_log_epochs == 0:
                    self._write_summary(n_epoch, batch, loss)
                    self._log_output_viz(
                        self.model,
                        self.params.n_viz_samples,
                        self.params.audio_len,
                        n_epoch,
                        self.model_dir,
                    )
            # Save the model if it's the best one so far
            if self.is_master and epoch_loss < best_loss:
                best_loss = epoch_loss
                print(f"Epoch[{n_epoch}]: Saving best model with loss {best_loss}")
                self.save_to_checkpoint(n_epoch)
            
    def train_step(self, batch):
        for param in self.model.parameters():
            param.grad = None

        audio = batch["audio"]
        conditioner = batch["conditioner"]

        batch_size, T = audio.shape
        device = audio.device
        self.noise_level = self.noise_level.to(device)

        with self.autocast:
            
            # 1. Assign a timestep to each sample in the batch.
            t = torch.randint(
                0, len(self.params.noise_schedule), [batch_size], device=audio.device
            )
            # 2. Get the corresponding noise for each sample, and add it to the audio.
            noise_scale = self.noise_level[t].unsqueeze(1)
            noise_scale_sqrt = noise_scale**0.5
            noise = torch.randn_like(audio)
            noisy_audio = noise_scale_sqrt * audio + (1.0 - noise_scale) ** 0.5 * noise

            # 3. Compute the score (gradient of the likelihood) for the noisy audio.
            predicted = self.model(noisy_audio, t, conditioner)

            # 4. Compute the loss.
            loss = self.loss_fn(noise, predicted.squeeze(1))

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        self.grad_norm = nn.utils.clip_grad_norm_(
            self.model.parameters(), self.params.max_grad_norm or 1e9
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return loss

    def _write_summary(self, step, batch, loss):
        writer = self.summary_writer or SummaryWriter(self.model_dir, purge_step=step)
        writer.add_audio(
            "feature/audio",
            batch["audio"][0],
            step,
            sample_rate=self.params.sample_rate,
        )
        writer.add_scalar("train/loss", loss, step)
        writer.add_scalar("train/grad_norm", self.grad_norm, step)
        writer.flush()
        self.summary_writer = writer

    def _log_output_viz(self, model, n_viz_samples, n_sample, epoch, outputs_dir):
        fig, axs = plt.subplots(nrows=n_viz_samples, ncols=1, figsize=(5, 15))

        # TODO: Fetch room config from params
        if self.params.dataset_name == "sinusoid": 
            dataset = RandomSinusoidDataset(n_sample, n_viz_samples)
        elif self.params.dataset_name == "rir":
            dataset = RandomRirDataset(n_sample, n_viz_samples, cat_labels=True)

            target_samples = [
                dataset[i] for i in range(n_viz_samples)
            ]
            conditioner = torch.stack(
                [target_sample["conditioner"] for target_sample in target_samples]
            ).to(model.device)
            audio = torch.stack(
                [target_sample["audio"] for target_sample in target_samples]
            ).to(model.device)

        outputs = predict_batch(
            model, conditioner, model.device, n_sample, n_viz_samples,
            fast_sampling=self.params.fast_sampling
        )[0]

        for i in range(n_viz_samples):
            axs[i].plot(audio[i].cpu().detach().numpy(), label="Target")
            axs[i].plot(outputs[i].cpu().detach().numpy(), label="Predicted")
            axs[i].legend()

        # Save the images
        os.makedirs(outputs_dir, exist_ok=True)
        plt.tight_layout()
        plt.savefig(f"{outputs_dir}/{epoch:04d}.png")


def _train_impl(replica_id, model, dataset, args, params):
    torch.backends.cudnn.benchmark = True
    opt = torch.optim.Adam(model.parameters(), lr=params.learning_rate)

    learner = DiffWaveLearner(
        args.model_dir, model, dataset, opt, params, fp16=args.fp16
    )
    learner.is_master = replica_id == 0
    learner.restore_from_checkpoint()
    learner.train()


def train(args, params):
    dataset = from_path(args.data_dirs, params)
    model = DiffWave(params)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    model = model.to(device)
    model.device = device

    _train_impl(0, model, dataset, args, params)


def train_distributed(replica_id, replica_count, port, args, params):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    torch.distributed.init_process_group(
        "nccl", rank=replica_id, world_size=replica_count
    )

    dataset = from_path(args.data_dirs, params, is_distributed=True)
    device = torch.device("cuda", replica_id)
    torch.cuda.set_device(device)
    model = DiffWave(params).to(device)
    model = DistributedDataParallel(model, device_ids=[replica_id])
    model.params = params
    _train_impl(replica_id, model, dataset, args, params)
