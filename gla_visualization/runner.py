"""GLA 可視化実験の実行エントリ。"""

import os

import numpy as np

from audio_processing import save_audio, stft
from gla_visualization.core import (
    generate_signal,
    initialize_gla_state,
    run_gla_iterations,
    select_target_bin,
)
from gla_visualization.plots import (
    create_combined_animation,
    plot_error_curve,
    plot_trajectory_static,
    plot_waveform_comparison,
)
from load_config import load_gla_config


def _visualize_gla_results(
    history: dict,
    output_dir: str,
    x_true: np.ndarray,
    x_curr: np.ndarray,
    t: np.ndarray,
    t_selected: float,
    idx_peak: int,
    A: np.ndarray,
    X_true: np.ndarray,
    t0: int,
    k0: int,
) -> None:
    """GLA 実験の可視化処理を実行する。

    Args:
        history (dict): 反復履歴。
        output_dir (str): 出力ディレクトリ。
        x_true (np.ndarray): 参照波形。
        x_curr (np.ndarray): 最終再構成波形。
        t (np.ndarray): 時間軸。
        t_selected (float): 強調時刻。
        idx_peak (int): 波形ピーク index。
        A (np.ndarray): 目標振幅スペクトル。
        X_true (np.ndarray): 参照複素スペクトル。
        t0 (int): 時間ビン index。
        k0 (int): 周波数ビン index。
    """
    print("Visualizing...")
    plot_error_curve(history, output_dir)

    peak_amp = np.abs(x_true[idx_peak])
    print(f"Target peak amplitude for visualization scaling: {peak_amp:.4f}")
    plot_trajectory_static(
        history["trajectory_bin"],
        A[k0, t0],
        X_true[k0, t0],
        output_dir,
        target_peak_amp=peak_amp,
    )
    plot_waveform_comparison(t, x_true, x_curr, output_dir, highlight_t=t_selected)

    print("Generating animation (this might take a while)...")
    create_combined_animation(
        history["trajectory_bin"],
        history["waveforms"],
        A[k0, t0],
        X_true[k0, t0],
        t,
        x_true,
        output_dir,
        highlight_t=t_selected,
        target_peak_amp=peak_amp,
    )


def run_gla_visualization(
    config_path: str = "config/default.yaml",
) -> None:
    """GLA プロセス可視化実験を実行する。

    Args:
        config_path (str, optional): 設定ファイルパス。
            Defaults to "config/default.yaml".
    """
    config = load_gla_config(config_path)
    output_dir = config.output.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print("Generating signal...")
    x_true, t = generate_signal(config)
    X_true = stft(x_true, config)
    A = np.abs(X_true)
    save_audio(x_true, config.audio.sr, os.path.join(output_dir, "1_original.wav"))

    hop_length = config.stft.hop_length
    idx_peak, t_selected, t0, k0 = select_target_bin(
        x_true=x_true, t=t, A=A, hop_length=hop_length
    )
    print("Starting GLA process...")
    X_curr, x_curr = initialize_gla_state(X_true=X_true, A=A, config=config)
    save_audio(x_curr, config.audio.sr, os.path.join(output_dir, "2_initial.wav"))

    X_curr, x_curr, history = run_gla_iterations(
        X_curr=X_curr,
        x_curr=x_curr,
        A=A,
        X_true=X_true,
        x_true=x_true,
        config=config,
        t0=t0,
        k0=k0,
    )
    save_audio(x_curr, config.audio.sr, os.path.join(output_dir, "3_reconstructed.wav"))

    _visualize_gla_results(
        history=history,
        output_dir=output_dir,
        x_true=x_true,
        x_curr=x_curr,
        t=t,
        t_selected=t_selected,
        idx_peak=idx_peak,
        A=A,
        X_true=X_true,
        t0=t0,
        k0=k0,
    )
    print("Done!")
