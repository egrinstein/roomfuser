import numpy as np
import torch

from roomfuser.noise_prior import NoisePrior


class NoiseScheduler:
    def __init__(self, beta, start_at_direct_path, variance_mode="exponential",
                 mean_mode="constant", rir_simulator=None, n_rir=None, batch_size=None,
                inference_noise_schedule=None):
        
        self.noise_prior = NoisePrior(
            start_at_direct_path=start_at_direct_path,
            variance_mode=variance_mode,
            mean_mode=mean_mode,
            rir_simulator=rir_simulator,
            n_rir=n_rir,
            batch_size=batch_size,
        )

        self.beta = beta = np.array(beta)
        self.noise_level = np.cumprod(1 - beta)
        self.noise_level = torch.tensor(self.noise_level.astype(np.float32))

        self.n_rir = n_rir
        self.batch_size = batch_size

        self.inference_noise_schedule = inference_noise_schedule

    def __getitem__(self, t):
        return self.noise_level[t]

    def __len__(self):
        return len(self.noise_level)
    
    def get_base_noise(self, audio, labels):
        return self.noise_prior.get_base_noise(labels, audio)

    def add_noise_to_audio(self, audio, labels, timestamp):
        # 1. Get the base noise
        noise = self.get_base_noise(audio, labels)

        # 2. Get the corresponding noise for each sample, and add it to the audio.
        noise_scale = self.noise_level[timestamp].unsqueeze(1)
        noise_scale_sqrt = noise_scale**0.5
        noisy_audio = noise_scale_sqrt * audio + (1.0 - noise_scale) ** 0.5 * noise

        return noisy_audio, noise

    def get_all_noisy_steps(self, audio, labels):
        """Get all noisy steps of the diffusion process.
            Used mainly for visualization purposes.
        """

        result = []

        for t in range(len(self)):
            t = torch.tensor([t]*audio.shape[0])
            noisy_audio, noise = self.add_noise_to_audio(audio, labels, [t])
            result.append(noisy_audio)
        
        return torch.stack(result, dim=1)

    def get_variance(self, audio, labels):
        return self.noise_prior.get_variance(labels, audio)

    def get_mean(self, audio, labels):
        return self.noise_prior.get_mean(labels, audio)

    def get_inference_config(self):
        # Change in notation from the DiffWave paper for fast sampling.
        # DiffWave paper -> Implementation below
        # --------------------------------------
        # alpha -> talpha
        # beta -> training_noise_schedule
        # gamma -> alpha
        # eta -> beta
        training_noise_schedule = self.beta

        if self.inference_noise_schedule is None:
            inference_noise_schedule = training_noise_schedule
        else:
            inference_noise_schedule = np.array(self.inference_noise_schedule)

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

        return {
            "alpha": alpha,
            "alpha_cum": alpha_cum,
            "beta": beta,
            "T": T,
        }
