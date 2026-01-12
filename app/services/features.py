import numpy as np
import librosa
import parselmouth
from parselmouth import praat
from scipy import signal
from typing import List, Dict, Optional


class FeatureExtraction:
    def __init__(self,
                 fmin: int = 75,
                 fmax: int = 500,
                 min_duration: float = 0.6,
                 min_voiced_ratio: float = 0.5,
                 max_f0_sd: float = 60.0,
                 top_db: int = 25,
                 use_sustained_segment: bool = True):
        self.fmin = fmin
        self.fmax = fmax
        self.min_duration = min_duration
        self.min_voiced_ratio = min_voiced_ratio
        self.max_f0_sd = max_f0_sd
        self.top_db = top_db
        self.use_sustained_segment = use_sustained_segment


    def find_sustained_segment(self, y: np.ndarray, sr: int) -> np.ndarray:
        y = np.asarray(y, dtype= np.float32)
        intervals = librosa.effects.split(y, top_db=self.top_db)
        best_seg = None
        best_score = -np.inf

        for start, end in intervals:
            seg = y[start:end]
            dur = (end - start) / sr
            if dur < self.min_duration:
                continue
            try:
                f0, voiced_flag, _ = librosa.pyin(
                    seg,
                    fmin = self.fmin,
                    fmax = self.fmax,
                    sr = sr,
                    frame_length = 2048,
                    hop_length = 512,
                    fill_na = np.nan,
                )
            except Exception:
                continue
            
            if f0 is None:
                continue
            
            voiced_mask = ~np.isnan(f0)
            voiced_ratio = voiced_mask.mean()
            if voiced_ratio < self.min_voiced_ratio:
                continue
            
            voiced_f0 = f0[voiced_mask]
            f0_sd = np.std(voiced_f0)
            if f0_sd > self.max_f0_sd:
                continue
            
            score = voiced_ratio * dur / (1.0 + f0_sd / 50.0)
            if score > best_score:
                best_score = score
                best_seg = seg
                
        if best_seg is None:
            total = len(y) / sr
            if total <= self.min_duration:
                return y
            start_t = (total - self.min_duration) / 2
            end_t = start_t + self.min_duration
            start = int(start_t * sr)
            end = int(end_t * sr)
            return y[start:end]
        return best_seg


    def compute_f0_features(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin = self.fmin,
            fmax = self.fmax,
            sr = sr,
            frame_length = 2048,
            hop_length = 512,
            fill_na = np.nan
        )

        if f0 is None or np.all(np.isnan(f0)):
            return {k: np.nan for k in [
                'f0_mean', 'f0_median', 'f0_min', 'f0_max', 'f0_sd', 'f0_range'
            ]}
        
        voiced_f0 = f0[~np.isnan(f0)]
        if voiced_f0.size == 0:
            return {k: np.nan for k in [
                'f0_mean', 'f0_median', 'f0_min', 'f0_max', 'f0_sd', 'f0_range'
            ]}
        return {
            'f0_mean': float(np.mean(voiced_f0)),
            'f0_median': float(np.median(voiced_f0)),
            'f0_min': float(np.min(voiced_f0)),
            'f0_max': float(np.max(voiced_f0)),
            'f0_range': float(np.max(voiced_f0) - np.min(voiced_f0)),
            'f0_sd': float(np.std(voiced_f0)),
        }


    def compute_praat_measures(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        sound = parselmouth.Sound(y, sampling_frequency = sr)
        point_process = praat.call(sound, "To PointProcess (periodic, cc)", self.fmin, self.fmax)
        
        # jitter
        jitter_local = praat.call(point_process, 'Get jitter (local)', 0, 0, 0.0001, 0.02, 1.3)
        jitter_rap = praat.call(point_process, 'Get jitter (rap)', 0, 0, 0.0001, 0.02, 1.3)
        jitter_ppq5 = praat.call(point_process, 'Get jitter (ppq5)', 0, 0, 0.001, 0.02, 1.3)

        # shimmer
        shimmer_local = praat.call([sound, point_process], 'Get shimmer (local)', 0, 0, 0.0001, 0.02, 1.3, 1.6)
        shimmer_apq3 = praat.call([sound, point_process], 'Get shimmer (apq3)', 0, 0, 0.0001, 0.02, 1.3, 1.6)
        shimmer_apq5 = praat.call([sound, point_process], 'Get shimmer (apq5)', 0, 0, 0.0001, 0.02, 1.3, 1.6)
        shimmer_apq11 = praat.call([sound, point_process], 'Get shimmer (apq11)', 0, 0, 0.0001, 0.02, 1.3, 1.6)

        # hnr
        harmonicity = praat.call(sound, "To Harmonicity (cc)", 0.01, self.fmin, 0.1, 1.0)
        hnr_mean = praat.call(harmonicity, 'Get mean', 0, 0)

        # cpp
        cp = praat.call(sound, "To PowerCepstrogram", 60, 0.002, 5000, 50)
        cpps = praat.call(
            cp, "Get CPPS", "yes", 0.02, 0.0005, 60, 330, 0.05,
            "parabolic", 0.001, 0.05, "Straight", "Robust"
        )

        return {
            'jitter_local': float(jitter_local),
            'jitter_rap': float(jitter_rap),
            'jitter_ppq5': float(jitter_ppq5),
            'shimmer_local': float(shimmer_local),
            'shimmer_apq3': float(shimmer_apq3),
            'shimmer_apq5': float(shimmer_apq5),
            'shimmer_apq11': float(shimmer_apq11),
            'hnr_mean': float(hnr_mean),
            'cpp_mean': float(cpps),
        }


    def spectral_features(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        S = np.abs(librosa.stft(y, n_fft = 2048, hop_length = 512))
        centroid = librosa.feature.spectral_centroid(S=S, sr=sr).mean()
        flatness = librosa.feature.spectral_flatness(S=S).mean()
        zcr = librosa.feature.zero_crossing_rate(y).mean()
        rolloff = librosa.feature.spectral_rolloff(S=S, sr = sr, roll_percent = 0.85).mean()
        return {
            'spectral_centroid': float(centroid),
            'spectral_flatness': float(flatness),
            'zcr': float(zcr),
            'spectral_rolloff_85': float(rolloff),
        }


    def ltas_bands(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        freqs, Pxx = signal.welch(y, fs=sr, nperseg=4096)
        total = np.trapz(Pxx, freqs) + 1e-12
        def band_energy(f_low, f_high):
            mask = (freqs >= f_low) & (freqs < f_high)
            if not np.any(mask):
                return np.nan
            band_power = np.trapz(Pxx[mask], freqs[mask])
            return float(band_power / total)
        return {
            'ltas_0_1k': band_energy(0, 1000),
            'ltas_1_2k': band_energy(1000, 2000),
            'ltas_2_4k': band_energy(2000, 4000),
            'ltas_4_8k': band_energy(4000, 8000),
        }

    def extract_features_one_segment(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        if self.use_sustained_segment:
            y = self.find_sustained_segment(y, sr)
        # extract features
        features = {}
        features.update(self.compute_f0_features(y, sr))
        features.update(self.compute_praat_measures(y, sr))
        features.update(self.spectral_features(y, sr))
        features.update(self.ltas_bands(y, sr))
        return self._clean_features(features)

    def aggregate_segment_features(self, segment_features: List[Dict[str, float]]) -> Dict[str, float]:
        if not segment_features:
            raise ValueError("No segment features provided")

        # Collect keys from first segment (all segments must match)
        keys = segment_features[0].keys()

        aggregated = {}

        for k in keys:
            values = []
            for seg in segment_features:
                v = seg.get(k, 0.0)

                # Safety: handle NaN / inf from Praat
                if v is None or not np.isfinite(v):
                    v = 0.0
                values.append(v)

            aggregated[k] = float(np.mean(values))
        return aggregated

    def _clean_features(self, feats: Dict[str, float]) -> Dict[str, float]:
        clean = {}
        for k, v in feats.items():
            if v is None or not np.isfinite(v):
                clean[k] = 0.0
            else:
                clean[k] = float(v)
        return clean


# global extractor
_default_extrator = FeatureExtraction()

def extract_features_one_segment(y: np.ndarray, sr: int) -> Dict[str, float]:
    return _default_extrator.extract_features_one_segment(y, sr)

def aggregate_segment_features(segment_features: List[Dict[str, float]]) -> Dict[str, float]:
    return _default_extrator.aggregate_segment_features(segment_features)

