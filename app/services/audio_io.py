import numpy as np
import soundfile as sf
from pathlib import Path
import librosa
import io
import tempfile
import os

def read_audio_bytes_to_mono_float_32(data: bytes, target_sr: int, file_ext: str = None) -> tuple[np.ndarray, int]:
    if file_ext:
        suffix = file_ext if file_ext.startswith(".") else f".{file_ext}"
    else:
        suffix = ".tmp"
    
    with tempfile.NamedTemporaryFile(delete = False, suffix = suffix) as temp_file:
        temp_file.write(data)
        temp_path = temp_file.name
    try:
        y, sr = librosa.load(temp_path, sr = target_sr, mono = True, dtype = np.float32)
        y = np.clip(y, -1.0, 1.0).astype(np.float32)
        return y, sr
    except Exception as e:
        raise ValueError(f"Failed to read audio file: {e}")
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)

def read_wav_bytes_to_mono_float32(data: bytes, target_sr: int) -> tuple[np.ndarray, int]:
    return read_audio_bytes_to_mono_float_32(data, target_sr, "wav")

def write_wav(path: Path, y: np.ndarray, sr: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path.as_posix(), y, sr, subtype="PCM_16")
