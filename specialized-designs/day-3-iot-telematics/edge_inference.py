import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EdgePreprocessor:
    def __init__(self, use_fpga=True):
        self.use_fpga = use_fpga  # FPGA or RPi fallback
        self.cnn_filters = 128  # 8-layer CNN params
        self.fs = 100  # Hz

    def preprocess(self, data: np.ndarray) -> np.ndarray:
        """Process telematics data on edge—FPGA or RPi."""
        try:
            # Data: [channels=3 (GPS, accel, CAN), samples]
            if data.shape[0] != 3:
                raise ValueError("Expected 3 channels")
            
            # FFT for spectral features (8-13 Hz)
            fft = np.fft.fft(data, axis=1)
            fft_power = np.abs(fft[:, 8:14])  # mu rhythm band
            
            # Velocity diffs (GPS-derived)
            velocity = data[0]  # GPS channel
            diffs = np.diff(velocity) / (1 / self.fs)
            
            # CNN feature extraction (simplified—FPGA runs HLS C++)
            features = np.concatenate([fft_power, diffs[np.newaxis, :-1]], axis=0)
            if self.use_fpga:
                # FPGA: 16-bit fixed-point, <1ms
                logger.info("FPGA preprocessing: <1ms latency")
                return features.astype(np.int16)  # Simulated output
            else:
                # RPi: ARM NEON, 5ms
                logger.info("RPi preprocessing: ~5ms latency")
                return features.astype(np.float32)
        except Exception as e:
            logger.error(f"Edge preprocessing failed: {e}")
            raise

# Simulate edge device
preprocessor = EdgePreprocessor(use_fpga=True)
data = np.random.randn(3, 100)  # 1s sample
features = preprocessor.preprocess(data)