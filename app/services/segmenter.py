import numpy as np
import librosa

def segment_audio(y: np.ndarray, sr: int, seg_s: float, hop_s: float) -> list[np.ndarray]:
    """
    Args:
        y: Audio array (float32, mono)
        sr: Sample rate
        seg_s: Segment duration in seconds (e.g., 3.0)
        hop_s: Hop size in seconds (e.g., 1.5 for 50% overlap)
    
    Returns:
        List of audio segment arrays
    """
    chunk_len = int(sr * seg_s)
    hop_len = int(sr * hop_s)
    if chunk_len <= 0 or hop_len <= 0:
        raise ValueError("Invalid segment/hop")

    if len(y) < chunk_len:
        # pad one to full segment
        return [y]

    chunks = []
    for start_idx in range(0, len(y) - chunk_len + 1, hop_len):
        end_idx = start_idx + chunk_len
        chunk = y[start_idx: end_idx]
        chunks.append(chunk)
    return chunks

def has_voice_activity(y: np.ndarray, sr: int, min_snr_db: float = 10.0, noise_frac: float = 0.1, signal_frac: float = 0.5) -> bool:
    if len(y) == 0:
        return False
    
    # calculate rms
    rms = librosa.feature.rms(y = y, frame_length = 1024, hop_length = 256)[0]
    if rms is None or len(rms) == 0:
        return False
    rms = np.asarray(rms, dtype = np.float32)
    rms_sorted = np.sort(rms)

    n_noise = max(1, int(noise_frac * len(rms_sorted)))
    start_signal = int((1.0 - signal_frac) * len(rms_sorted))
    noise_rms = np.mean(rms_sorted[:n_noise])
    signal_rms = np.mean(rms_sorted[start_signal:])
    snr_db = 20 * np.log10((signal_rms + 1e-12) / (noise_rms + 1e-12))
    return snr_db >= min_snr_db


