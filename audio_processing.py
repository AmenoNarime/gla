"""音声処理ユーティリティ。"""

import os

import librosa
import numpy as np
import soundfile as sf

from config_models import GlaVisualizationConfig

AppConfig = GlaVisualizationConfig


def _ensure_parent_dir(file_path: str) -> None:
    """保存先ファイルの親ディレクトリを作成する。"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)


def _get_stft_params(config: AppConfig) -> tuple[int, int, str]:
    """設定から STFT 共通パラメータを取得する。"""
    return (config.stft.n_fft, config.stft.hop_length, config.stft.window)


def _get_gla_params(config: AppConfig) -> tuple[int, int, str, int]:
    """設定から Griffin-Lim パラメータを取得する。"""
    return (
        config.gla.n_fft,
        config.gla.hop_length,
        config.gla.window,
        config.gla.n_iter,
    )


def load_audio(audio_path: str, config: AppConfig) -> tuple[np.ndarray, int]:
    """音声ファイルを読み込む。

    Args:
        audio_path (str): 音声ファイルのパス
        config (dict): 設定ファイル

    Returns:
        tuple[np.ndarray, int]: 音声データとサンプリングレート
    """
    y, sr = librosa.load(audio_path, sr=config.audio.sr)
    return y, int(sr)


def save_audio(y: np.ndarray, sr: int, output_path: str) -> None:
    """音声データをファイルとして保存する。

    Args:
        y (np.ndarray): 音声データ
        sr (int): サンプリングレート
        output_path (str): 保存先のパス
    """
    _ensure_parent_dir(output_path)
    sf.write(output_path, y, sr)


def stft(y: np.ndarray, config: AppConfig) -> np.ndarray:
    """STFT（短時間フーリエ変換）を計算する。

    Args:
        y (np.ndarray): 音声データ
        config (dict): 設定ファイル

    Returns:
        np.ndarray: STFTの結果（複素数）
    """
    n_fft, hop_length, window = _get_stft_params(config)
    return librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window=window)


def istft(D: np.ndarray, config: AppConfig) -> np.ndarray:
    """ISTFT（逆短時間フーリエ変換）を計算する。

    Args:
        D (np.ndarray): STFTの結果（複素数）
        config (dict): 設定ファイル

    Returns:
        np.ndarray: 逆STFTの結果（音声データ）
    """
    n_fft, hop_length, window = _get_stft_params(config)
    return librosa.istft(D, n_fft=n_fft, hop_length=hop_length, window=window)


def magnitude_spectrogram(D: np.ndarray) -> np.ndarray:
    """STFT結果から振幅スペクトル（Magnitude Spectrogram）を取得する。

    Args:
        D (np.ndarray): STFTの結果（複素数）

    Returns:
        np.ndarray: 振幅スペクトル
    """
    S, _ = librosa.magphase(D)  # 複素数を強度（振幅スペクトル）と位相へ変換
    return S


def griffin_lim_algorithm(S: np.ndarray, config: AppConfig, **kwargs) -> np.ndarray:
    """Griffin-Lim法で振幅スペクトルから波形（位相）を推定する。

    Args:
        S (np.ndarray): 振幅スペクトル
        config (dict): 設定ファイル
        **kwargs: librosa.griffinlim に渡す追加引数 (例: random_state, init)

    Returns:
        np.ndarray: 推定された音声波形
    """
    n_fft, hop_length, window, n_iter = _get_gla_params(config)

    return librosa.griffinlim(
        S, n_iter=n_iter, n_fft=n_fft, hop_length=hop_length, window=window, **kwargs
    )


def compute_lsd(S_true: np.ndarray, S_recon: np.ndarray) -> float:
    """Log-Spectral Distance (LSD) を計算する。

    Args:
        S_true (np.ndarray): 正解の振幅スペクトログラム (Linear scale)
        S_recon (np.ndarray): 復元の振幅スペクトログラム (Linear scale)

    Returns:
        float: LSD
    """
    eps = 1e-10
    # パワースペクトルへ変換し、対数をとる
    L_true = 10 * np.log10(S_true**2 + eps)
    L_recon = 10 * np.log10(S_recon**2 + eps)

    # 二乗誤差の平均平方根を計算
    diff_squared = (L_true - L_recon) ** 2
    # 周波数方向の平均 -> ルート -> 時間方向の平均
    lsd = np.mean(np.sqrt(np.mean(diff_squared, axis=0)))
    return lsd


def normalize_audio(y: np.ndarray, peak: float = 0.99) -> np.ndarray:
    """ピーク正規化（クリップ回避）

    Args:
        y (np.ndarray): 音声データ
        peak (float): ピーク値

    Returns:
        np.ndarray: 正規化された音声データ
    """
    m = np.max(np.abs(y)) + 1e-12
    return (y / m) * peak
