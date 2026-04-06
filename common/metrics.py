"""評価指標計算。"""

import numpy as np

from audio_processing import istft, stft
from common.constants import EPSILON, LOG_EPSILON
from config_models import GlaVisualizationConfig


def calculate_spectral_convergence(
    mag_ref: np.ndarray, mag_est: np.ndarray, eps: float = EPSILON
) -> float:
    """Spectral Convergence を計算する。"""
    num = np.linalg.norm(mag_ref - mag_est, ord="fro")
    den = np.linalg.norm(mag_ref, ord="fro") + eps
    return float(num / den)


def calculate_log_magnitude_l1(
    mag_ref: np.ndarray, mag_est: np.ndarray, eps: float = LOG_EPSILON
) -> float:
    """対数振幅の L1 誤差を計算する。"""
    return float(np.mean(np.abs(np.log(mag_ref + eps) - np.log(mag_est + eps))))


def calculate_stft_consistency(y: np.ndarray, config: GlaVisualizationConfig) -> float:
    """STFT の整合性誤差を計算する。"""
    d1 = stft(y, config)
    y_rt = istft(d1, config)
    d2 = stft(y_rt, config)
    num = np.linalg.norm(d1 - d2, ord="fro")
    den = np.linalg.norm(d1, ord="fro") + EPSILON
    return float(num / den)
