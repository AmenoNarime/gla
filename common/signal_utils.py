"""信号処理ユーティリティ。"""

import numpy as np

from audio_processing import magnitude_spectrogram, stft
from config_models import GlaVisualizationConfig


def align_signals_to_min_length(*signals: np.ndarray) -> tuple[np.ndarray, ...]:
    """複数波形を最短長に揃える。"""
    min_length = min(len(sig) for sig in signals)
    return tuple(sig[:min_length] for sig in signals)


def compute_magnitude(y: np.ndarray, config: GlaVisualizationConfig) -> np.ndarray:
    """波形から振幅スペクトルを計算する。"""
    return magnitude_spectrogram(stft(y, config))
