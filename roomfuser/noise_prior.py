import torch
from roomfuser.utils import get_exponential_envelope


class NoisePrior:
    def __init__(self, start_at_direct_path, variance_mode="exponential",
                 mean_mode="constant", rir_simulator=None, n_rir=None, batch_size=None,
                 frequency_response=False):
        self.start_at_direct_path = start_at_direct_path
        self.variance_mode = variance_mode
        self.mean_mode = mean_mode
        self.batch_size = batch_size
        self.frequency_response = frequency_response

        self.n_rir = n_rir
        if n_rir and frequency_response:
            self.n_rir = n_rir//2 + 1

        if mean_mode == "low_ord_rir":
            if rir_simulator is None:
                raise ValueError("rir_simulator must be provided when mean_mode is 'low_ord_rir'.")
            self.rir_simulator = rir_simulator

      
    def get_base_noise(self, labels, audio=None):
        """Get the base noise for the diffusion process.
        Args:
            labels (list): List of dictionaries containing the labels for each sample.
            audio (torch.Tensor): Audio tensor. If provided, the noise will have the same shape as the audio.
        """

        if audio is None:
            if self.n_rir is None or self.batch_size is None:
                raise ValueError("n_rir and batch_size must be provided at initialization when audio is None.")
            noise = torch.randn(self.batch_size, self.n_rir, device=labels[0]["source_pos"].device)
        else:
            noise = torch.randn_like(audio)
        
        if self.frequency_response: # Prior mean and variance are not computed in the frequency domain
            return noise

        # Compute prior mean and variance of the noise
        variance = self.get_variance(labels, audio)
        mean = self.get_mean(labels, audio)
        noise = noise * variance + mean

        return noise

    def get_variance(self, labels, audio=None):

        if audio is None:
            if self.n_rir is None or self.batch_size is None:
                raise ValueError("n_rir and batch_size must be provided at initialization when audio is None.")
        else:
            batch_size = audio.shape[0]
            n_rir = audio.shape[1]

        variance = torch.zeros(batch_size, n_rir, device=labels[0]["source_pos"].device)
        for i in range(len(audio)):
            a, l = audio[i], labels[i]
            variance[i] = get_prior_variance(
                n_rir=len(a),
                source_pos=l["source_pos"],
                mic_pos=l["mic_pos"],
                rt60=l["rt60"],
                start_at_direct_path=self.start_at_direct_path,
                mode=self.variance_mode
            ).to(audio.device)

        return variance

    def get_mean(self, labels, audio=None):
        if audio is None:
            if self.n_rir is None or self.batch_size is None:
                raise ValueError("n_rir and batch_size must be provided at initialization when audio is None.")
        else:
            batch_size = audio.shape[0]
            n_rir = audio.shape[1]

        mean = torch.zeros(batch_size, n_rir, device=labels[0]["source_pos"].device)
        
        if self.mean_mode == "constant":
            return mean
        
        # Make the mean of the noise for each sample the low order RIR
        for i in range(len(audio)):
            a, l = audio[i], labels[i]

            if self.mean_mode == "low_ord_rir":
                mean[i] = self.rir_simulator(l).to(a.device)


        return mean


def get_prior_variance(
        n_rir, source_pos, mic_pos, rt60, sr=16000, c=343.0,
        start_at_direct_path=True, mode="exponential", min_variance=0.1, scale=5):
    """Get prior for the room impulse response.
    Args:
        n_envelope (int): Number of samples in the envelope.
        source_pos (torch.Tensor): Source position in meters.
        mic_pos (torch.Tensor): Microphone position in meters.
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

    envelope += min_variance
    envelope *= scale

    return envelope
