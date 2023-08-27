import numpy as np
import torch


def get_prior_variance(
        n_rir, source_pos, mic_pos, rt60, sr=16000, c=343.0,
        start_at_direct_path=True, mode="exponential"):
    """Get prior for the room impulse response.
    Args:
        n_envelope (int): Number of samples in the envelope.
        source_pos (np.array): Source position in meters.
        mic_pos (np.array): Microphone position in meters.
        sr (float): Sampling rate.
        rt60 (float): Reverberation time in seconds.
        c (float): Speed of sound in meters per second.
        start_at_direct_path (bool): If True, the envelope starts at the
            time of arrival of the direct path. If False, the envelope
            starts at t=0.
        mode: "exponential", "linear" or "constant"
    """

    if mode == "exponential":
        envelope = get_exponential_envelope(
            n_envelope=n_rir,
            rt60=rt60,
            sr=sr,
        )
    elif mode == "linear":
        envelope = torch.linspace(1, 0, n_rir)
    elif mode == "constant":
        envelope = torch.ones(n_rir)

    if start_at_direct_path:
        # Get distance between source and microphone
        distance = torch.linalg.norm(source_pos - mic_pos)

        # Get time of arrival of direct path, in samples
        t_direct = distance * sr / c
        # Shift envelope to the time of arrival of the direct path
        envelope = torch.roll(envelope, int(t_direct))
        # Zero pad the envelope
        envelope = torch.nn.functional.pad(envelope, (0, n_rir - len(envelope)))

    return envelope


def get_exponential_envelope(n_envelope, rt60, sr=16000):
    """Get Room Impulse Response envelope. The envelope is modeled as a
    decaying exponential which reaches -60 dB (10^(-6)) after rt60 seconds.
    The exponential starts at the time of arrival of the direct path,
    which is the distance between the source and the microphone divided
    by the speed of sound.
    
    Args:
        n_envelope (int): Number of samples in the envelope.
        source_pos (np.array): Source position in meters.
        mic_pos (np.array): Microphone position in meters.
        sr (float): Sampling rate.
        rt60 (float): Reverberation time in seconds.
        c (float): Speed of sound in meters per second.
        start_at_direct_path (bool): If True, the envelope starts at the
            time of arrival of the direct path. If False, the envelope
            starts at t=0.
    """

    # Get envelope centered at t=0
    t_envelope = torch.arange(n_envelope) / sr
    envelope = torch.exp(-6 * np.log(10) * t_envelope / rt60)
    
    return envelope


class NoiseScheduler:
    def __init__(self, beta, start_at_direct_path, variance_mode="exponential"):
        self.beta = beta = np.array(beta)
        self.start_at_direct_path = start_at_direct_path
        self.variance_mode = variance_mode

        self.noise_level = np.cumprod(1 - beta)
        self.noise_level = torch.tensor(self.noise_level.astype(np.float32))

    def __getitem__(self, t):
        return self.noise_level[t]

    def __len__(self):
        return len(self.noise_level)
    
    def add_noise_to_audio(self, audio, timestamp, envelope=None):
        noise = torch.randn_like(audio)
        if envelope is not None: # Weight the noise by the RIR envelopes
            noise *= envelope

        # 2. Get the corresponding noise for each sample, and add it to the audio.
        noise_scale = self.noise_level[timestamp].unsqueeze(1)
        noise_scale_sqrt = noise_scale**0.5
        noisy_audio = noise_scale_sqrt * audio + (1.0 - noise_scale) ** 0.5 * noise

        return noisy_audio, noise

    def get_all_noisy_steps(self, audio, envelope=None):
        """Get all noisy steps of the diffusion process.
            Used mainly for visualization purposes.
        """

        result = []

        for t in range(len(self)):
            t = torch.tensor([t]*audio.shape[0])
            noisy_audio, noise = self.add_noise_to_audio(audio, [t], envelope=envelope)
            result.append(noisy_audio)
        
        return torch.stack(result, dim=1)

    def get_envelope(self, audio, labels):

        envelopes = []
        for a, l in zip(audio, labels):
            envelopes.append(get_prior_variance(
                n_rir=len(a),
                source_pos=l["source_pos"],
                mic_pos=l["mic_pos"],
                rt60=l["rt60"],
                start_at_direct_path=self.start_at_direct_path,
                mode=self.variance_mode
            ))

        return torch.stack(envelopes).to(audio.device)
