"""GLA 可視化実験のコア処理。"""

import numpy as np
from tqdm import tqdm

from audio_processing import istft, stft
from common.constants import EPSILON
from common.signal_utils import align_signals_to_min_length
from config_models import GlaVisualizationConfig


def generate_signal(config: GlaVisualizationConfig) -> tuple[np.ndarray, np.ndarray]:
    """減衰正弦波信号を生成する。

    Args:
        config (dict): 音声生成パラメータを含む設定。

    Returns:
        tuple[np.ndarray, np.ndarray]: 生成波形と時間軸。
    """
    sr = config.audio.sr
    duration = config.audio.duration
    f0 = config.audio.f0
    tau = config.audio.tau

    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    x = np.exp(-t / tau) * np.sin(2 * np.pi * f0 * t)
    return x, t


def amplitude_projection(Y: np.ndarray, A: np.ndarray) -> np.ndarray:
    """振幅集合 M への射影を行う。

    Args:
        Y (np.ndarray): 現在の複素スペクトル推定値。
        A (np.ndarray): 目標振幅スペクトル。

    Returns:
        np.ndarray: 位相を維持して振幅だけを合わせたスペクトル。
    """
    phase = np.angle(Y)
    return A * np.exp(1j * phase)


def consistency_projection(
    Y: np.ndarray, config: GlaVisualizationConfig
) -> tuple[np.ndarray, np.ndarray]:
    """整合性集合 C への射影を行う。

    Args:
        Y (np.ndarray): 複素スペクトル。
        config (dict): STFT/ISTFT 設定。

    Returns:
        tuple[np.ndarray, np.ndarray]: 射影後スペクトルと対応する波形。
    """
    y_rec = istft(Y, config)
    return stft(y_rec, config), y_rec


def calculate_metrics(
    X_curr: np.ndarray,
    X_next_cons: np.ndarray | None,
    A: np.ndarray,
    x_curr: np.ndarray,
    x_true: np.ndarray,
) -> tuple[float, float]:
    """誤差指標を計算する。

    Args:
        X_curr (np.ndarray): 現在のスペクトル推定値。
        X_next_cons (np.ndarray | None): 互換性維持用の未使用引数。
        A (np.ndarray): 目標振幅スペクトル。
        x_curr (np.ndarray): 現在の波形推定値。
        x_true (np.ndarray): 参照波形。

    Returns:
        tuple[float, float]: 振幅誤差と波形誤差。
    """
    del X_next_cons  # for backward-compatible signature
    term1 = np.linalg.norm(np.abs(X_curr) - A, "fro")
    denom1 = np.linalg.norm(A, "fro") + EPSILON
    e_mag = term1 / denom1

    x_curr_aligned, x_true_aligned = align_signals_to_min_length(x_curr, x_true)
    term3 = np.linalg.norm(x_curr_aligned - x_true_aligned, 2)
    denom3 = np.linalg.norm(x_true_aligned, 2) + EPSILON
    e_x = term3 / denom3
    return e_mag, e_x


def calculate_phase_mae(
    X_est: np.ndarray, X_true: np.ndarray, magnitude_ratio_threshold: float = 1e-3
) -> float:
    """真値位相に対する重みなし MAE を計算する。

    Args:
        X_est (np.ndarray): 推定複素スペクトル。
        X_true (np.ndarray): 参照複素スペクトル。
        magnitude_ratio_threshold (float, optional): 参照振幅に対する閾値比率。
            小振幅ビンの位相は不安定なため評価から除外する。
            Defaults to 1e-3.

    Returns:
        float: 位相 MAE [rad]。有効ビンがない場合は NaN。
    """
    max_mag = float(np.max(np.abs(X_true)))
    threshold = max(max_mag * magnitude_ratio_threshold, EPSILON)
    mask = np.abs(X_true) > threshold
    if not np.any(mask):
        return float("nan")

    delta_phase = np.angle(X_est * np.conj(X_true))
    return float(np.mean(np.abs(delta_phase[mask])))


def select_target_bin(
    x_true: np.ndarray, t: np.ndarray, A: np.ndarray, hop_length: int
) -> tuple[int, float, int, int]:
    """波形ピークに対応する STFT ビンを選択する。

    Args:
        x_true (np.ndarray): 参照波形。
        t (np.ndarray): 時間軸。
        A (np.ndarray): 目標振幅スペクトル。
        hop_length (int): STFT のホップ長。

    Returns:
        tuple[int, float, int, int]: 波形ピークindex、ピーク時刻、時間ビン、周波数ビン。
    """
    idx_peak = np.argmax(np.abs(x_true))
    t_selected = t[idx_peak]
    print(f"Waveform Peak Time: {t_selected:.4f} s (index={idx_peak})")

    t0 = int(round(idx_peak / hop_length))
    t0 = min(t0, A.shape[1] - 1)
    k0 = np.argmax(A[:, t0])
    print(
        "Selected bin based on peak: "
        f"freq_idx={k0}, time_idx={t0}, Magnitude={A[k0, t0]:.2f}"
    )
    return idx_peak, t_selected, t0, k0


def initialize_gla_state(
    X_true: np.ndarray, A: np.ndarray, config: GlaVisualizationConfig
) -> tuple[np.ndarray, np.ndarray]:
    """初期位相を生成し、整合性集合上の初期状態を返す。

    Args:
        X_true (np.ndarray): 参照複素スペクトル。
        A (np.ndarray): 目標振幅スペクトル。
        config (dict): 初期化設定を含む実験設定。

    Returns:
        tuple[np.ndarray, np.ndarray]: 初期スペクトルと初期波形。
    """
    init_type = config.gla.init_type or "random"
    np.random.seed(42)

    if init_type == "perturbed":
        noise_scale = config.gla.noise_scale or 0.5
        print(f"Initialization: Perturbed True Phase (scale={noise_scale} rad)")
        true_phase = np.angle(X_true)
        noise = np.random.uniform(-noise_scale, noise_scale, A.shape)
        theta = np.angle(np.exp(1j * (true_phase + noise)))
    else:
        print("Initialization: Random Phase")
        theta = np.random.uniform(-np.pi, np.pi, A.shape)

    X_init_raw = A * np.exp(1j * theta)
    return consistency_projection(X_init_raw, config)


def run_gla_iterations(
    X_curr: np.ndarray,
    x_curr: np.ndarray,
    A: np.ndarray,
    X_true: np.ndarray,
    x_true: np.ndarray,
    config: GlaVisualizationConfig,
    t0: int,
    k0: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """GLA 反復を実行し、最終状態と履歴を返す。

    Args:
        X_curr (np.ndarray): 初期スペクトル。
        x_curr (np.ndarray): 初期波形。
        A (np.ndarray): 目標振幅スペクトル。
        X_true (np.ndarray): 参照複素スペクトル。
        x_true (np.ndarray): 参照波形。
        config (dict): 実験設定。
        t0 (int): 監視する時間ビン。
        k0 (int): 監視する周波数ビン。

    Returns:
        tuple[np.ndarray, np.ndarray, dict]: 最終スペクトル、最終波形、履歴。
    """
    history = {
        "e_mag": [],
        "d_c": [],
        "e_x": [],
        "phase_mae": [],
        "trajectory_bin": [],
        "waveforms": [x_curr],
    }

    n_iter = config.gla.n_iter
    for _ in tqdm(range(n_iter)):
        X_k_val = X_curr[k0, t0]
        X_hat = amplitude_projection(X_curr, A)
        X_hat_val = X_hat[k0, t0]
        X_next, x_next = consistency_projection(X_hat, config)
        X_next_val = X_next[k0, t0]

        e_mag, e_x = calculate_metrics(X_curr, None, A, x_curr, x_true)
        phase_mae = calculate_phase_mae(X_curr, X_true)
        term_dc = np.linalg.norm(X_hat - X_next, "fro")
        denom_dc = np.linalg.norm(X_hat, "fro") + EPSILON
        d_c = term_dc / denom_dc

        history["e_mag"].append(e_mag)
        history["d_c"].append(d_c)
        history["e_x"].append(e_x)
        history["phase_mae"].append(phase_mae)
        history["trajectory_bin"].append(
            {"X_k": X_k_val, "X_hat_k": X_hat_val, "X_k_next": X_next_val}
        )
        history["waveforms"].append(x_next)

        X_curr = X_next
        x_curr = x_next

    return X_curr, x_curr, history
