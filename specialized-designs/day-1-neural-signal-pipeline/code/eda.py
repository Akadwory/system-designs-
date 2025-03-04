import numpy as np
from scipy.signal import coherence
import logging
from typing import Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SignalQualityAnalyzer:
    """Production-ready EDA for neural signal validation."""
    def __init__(self, fs: int = 10000, snr_threshold: float = 10.0):
        self.fs = fs
        self.snr_threshold = snr_threshold

    def compute_snr(self, signal: np.ndarray, noise_estimate: np.ndarray) -> float:
        """Compute Signal-to-Noise Ratio in dB."""
        try:
            signal_power = np.var(signal, axis=1)
            noise_power = np.var(noise_estimate, axis=1)
            snr = 10 * np.log10(signal_power / noise_power)
            avg_snr = np.mean(snr)
            logger.info(f"SNR computed: {avg_snr:.2f} dB")
            if avg_snr < self.snr_threshold:
                logger.warning(f"SNR {avg_snr:.2f} dB below threshold {self.snr_threshold}")
            return avg_snr
        except Exception as e:
            logger.error(f"SNR computation failed: {e}")
            raise

    def channel_coherence(self, data: np.ndarray) -> float:
        """Compute average coherence across channel pairs below 50 Hz."""
        try:
            n_channels = data.shape[0]
            coh_sum = 0
            count = 0
            for i in range(n_channels):
                for j in range(i + 1, n_channels):
                    f, Cxy = coherence(data[i], data[j], fs=self.fs)
                    coh_sum += np.mean(Cxy[f < 50])
                    count += 1
            avg_coh = coh_sum / count if count > 0 else 0
            logger.info(f"Average coherence: {avg_coh:.3f}")
            return avg_coh
        except Exception as e:
            logger.error(f"Coherence computation failed: {e}")
            raise