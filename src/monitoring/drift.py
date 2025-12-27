import numpy as np
from src.utils.logger import get_logger

logger = get_logger("DRIFT")

class DataDriftMonitor:
    def __init__(self, baseline_mean: float, threshold: float = 30.0):
        self.baseline_mean = baseline_mean
        self.threshold = threshold

    def check_text_length_drift(self, texts: list[str]) -> bool:
        current_mean = np.mean([len(t) for t in texts])
        drift = abs(self.baseline_mean - current_mean)

        logger.info(f"Baseline mean={self.baseline_mean}")
        logger.info(f"Current mean={current_mean}")

        if drift > self.threshold:
            logger.warning("⚠️ Text length drift detected")
            return True

        return False
