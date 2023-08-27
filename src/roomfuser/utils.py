import numpy as np
import torch


def get_rir_envelope(n_envelope, source_pos, mic_pos, rt60, sr=16000, c=343.0):
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
    """

    # Get distance between source and microphone
    distance = torch.linalg.norm(source_pos - mic_pos)

    # Get time of arrival of direct path, in samples
    t_direct = distance * sr / c

    # Get envelope centered at t=0
    t_envelope = torch.arange(n_envelope) / sr
    # envelope = 0.1 + 0.9*torch.exp(-6 * np.log(10) * t_envelope / rt60)
    envelope = torch.exp(-6 * np.log(10) * t_envelope / rt60)
    # Shift envelope to the time of arrival of the direct path
    envelope = torch.roll(envelope, int(t_direct))

    # Zero pad the envelope
    # envelope = np.pad(envelope, (0, n_envelope - len(envelope)))
    return envelope
