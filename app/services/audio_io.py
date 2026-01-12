import numpy as np
import soundfile as sf
from pathlib import Path
import librosa
import io
import tempfile
import os

def read_wav_bytes_to_mono_float32(data: bytes, target_sr: int) -> tuple[np.ndarray, int]:
    with tempfile.NamedTemporaryFile(delete = False, suffix = ".wav") as tmp_file:
        tmp_file.write(data)
        tmp_path = tmp_file.name   
    try:
        y, sr = librosa.load(tmp_path, sr = target_sr, mono = True, dtype = np.float32)
        y = np.clip(y, -1.0, 1.0).astype(np.float32)
        return y, sr
    except Exception as e:
        raise ValueError(f"Failed to read WAV file: {e}")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

def write_wav(path: Path, y: np.ndarray, sr: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path.as_posix(), y, sr, subtype="PCM_16")
