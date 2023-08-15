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
import torch.nn.functional as F
import torchaudio

from glob import glob
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader


class RandomSinusoidDataset(Dataset):
    """
    Random Sinusoid Dataset
    """
    def __init__(self, n_sample, n_samples_per_epoch, n_min_cycle=1, n_max_cycle=4):
        self.n_sample = n_sample
        self.n_samples_per_epoch = n_samples_per_epoch
        self.n_min_cycle = n_min_cycle
        self.n_max_cycle = n_max_cycle
    
    def __len__(self):
        return self.n_samples_per_epoch
    
    def __getitem__(self, idx):
        """
        Returns a random sinusoid function
        """
        # Generate random amplitude, phase, and number of cycles
        
        amplitude, phase, n_cycles = get_random_sinusoid_config(
                                      self.n_max_cycle, self.n_min_cycle)

        # Generate x values
        x = torch.arange(0, self.n_sample)

        # Generate y values
        y = amplitude * torch.sin(2 * torch.tensor(np.pi) * n_cycles * x / self.n_sample + phase)
        
        # Concatenate amplitude, phase, and number of cycles
        labels = torch.cat((amplitude, phase, n_cycles))

        return {
            'audio': y,
            'conditioner': labels,
        }

def get_random_sinusoid_config(n_max_cycle=4, n_min_cycle=1, cat=False):
    # Amplitude is uniform in [0.1, 1]
    #amplitude = torch.rand(1) * 0.9 + 0.1
    amplitude = torch.Tensor([0.5])

    # Phase is uniform in [0, 2*pi]
    phase = torch.rand(1) * 2 * torch.tensor(np.pi)
    
    # Number of cycles is uniform in [n_min_cycle, n_max_cycle]
    n_cycles = torch.rand(1)*(n_max_cycle - 1) + n_min_cycle

    if cat:
      return torch.cat((amplitude, phase, n_cycles))
    else:
      return amplitude, phase, n_cycles


class ConditionalDataset(Dataset):
  def __init__(self, paths, load_spectrograms=False):
    super().__init__()
    self.filenames = []
    for path in paths:
      self.filenames += glob(f'{path}/**/*.wav', recursive=True)

    self.load_spectrograms = load_spectrograms

  def __len__(self):
    return len(self.filenames)

  def __getitem__(self, idx):
    audio_filename = self.filenames[idx]
    signal, _ = torchaudio.load(audio_filename)

    spec_filename = f'{audio_filename}.spec.npy'
    spectrogram = None
    if os.path.exists(spec_filename) and self.load_spectrograms:
      spectrogram = np.load(spec_filename).T
    
    breakpoint()
    return {
        'audio': signal[0],
        'conditioner': spectrogram
    }


class UnconditionalDataset(ConditionalDataset):
  def __init__(self, paths):
    super().__init__(paths, load_spectrograms=False)


class Collator:
  def __init__(self, params):
    self.params = params

  def collate(self, minibatch):
    samples_per_frame = self.params.hop_samples
    for record in minibatch:
      if self.params.unconditional:
          # Filter out records that aren't long enough.
          if len(record['audio']) < self.params.audio_len:
            del record['conditioner']
            del record['audio']
            continue

          # Crop bigger records to audio_len.
          start = random.randint(0, record['audio'].shape[-1] - self.params.audio_len)
          end = start + self.params.audio_len
          record['audio'] = record['audio'][start:end]
          record['audio'] = np.pad(record['audio'], (0, (end - start) - len(record['audio'])), mode='constant')
      else:
          # # Filter out records that aren't long enough.
          # if len(record['conditioner']) < self.params.crop_mel_frames:
          #   del record['conditioner']
          #   del record['audio']
          #   continue

          # start = random.randint(0, record['conditioner'].shape[0] - self.params.crop_mel_frames)
          # end = start + self.params.crop_mel_frames
          # record['conditioner'] = record['conditioner'][start:end].T

          # start *= samples_per_frame
          # end *= samples_per_frame
          # record['audio'] = record['audio'][start:end]
          # record['audio'] = np.pad(record['audio'], (0, (end-start) - len(record['audio'])), mode='constant')
          pass

    audio = np.stack([record['audio'] for record in minibatch if 'audio' in record])
    if self.params.unconditional:
        return {
            'audio': torch.from_numpy(audio),
            'conditioner': None,
        }
    spectrogram = np.stack([record['conditioner'] for record in minibatch if 'conditioner' in record])
    return {
        'audio': torch.from_numpy(audio),
        'conditioner': torch.from_numpy(spectrogram),
    }

  # for gtzan
  def collate_gtzan(self, minibatch):
    ldata = []
    mean_audio_len = self.params.audio_len # change to fit in gpu memory
    # audio total generated time = audio_len * sample_rate
    # GTZAN statistics
    # max len audio 675808; min len audio sample 660000; mean len audio sample 662117
    # max audio sample 1; min audio sample -1; mean audio sample -0.0010 (normalized)
    # sample rate of all is 22050
    for data in minibatch:
      if data[0].shape[-1] < mean_audio_len:  # pad
        data_audio = F.pad(data[0], (0, mean_audio_len - data[0].shape[-1]), mode='constant', value=0)
      elif data[0].shape[-1] > mean_audio_len:  # crop
        start = random.randint(0, data[0].shape[-1] - mean_audio_len)
        end = start + mean_audio_len
        data_audio = data[0][:, start:end]
      else:
        data_audio = data[0]
      ldata.append(data_audio)
    audio = torch.cat(ldata, dim=0)
    return {
          'audio': audio,
          'conditioner': None,
    }


def from_path(data_dirs, params, is_distributed=False):
  if params.random_sinusoid_dataset:
    dataset = RandomSinusoidDataset(n_sample=params.audio_len,
                                    n_samples_per_epoch=params.n_samples_per_epoch)
  elif params.unconditional:
    dataset = UnconditionalDataset(data_dirs)
  else:#with condition
    dataset = ConditionalDataset(data_dirs)
  return DataLoader(
      dataset,
      batch_size=params.batch_size,
      collate_fn=Collator(params).collate,
      shuffle=not is_distributed,
      num_workers=os.cpu_count(),
      sampler=DistributedSampler(dataset) if is_distributed else None,
      pin_memory=True,
      drop_last=True)


def from_gtzan(params, is_distributed=False):
  dataset = torchaudio.datasets.GTZAN('./data', download=True)
  return DataLoader(
      dataset,
      batch_size=params.batch_size,
      collate_fn=Collator(params).collate_gtzan,
      shuffle=not is_distributed,
      num_workers=os.cpu_count(),
      sampler=DistributedSampler(dataset) if is_distributed else None,
      pin_memory=True,
      drop_last=True)
