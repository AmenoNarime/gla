"""GLA 可視化実験の描画処理。"""

import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from common.constants import (
    PLOT_LIMIT_MARGIN,
    SCALE_THRESHOLD,
    ZOOM_WINDOW_SEC,
)
from common.plotting import save_figure

sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.size": 12})


def _save_current_figure(
    output_dir: str, filename: str, dpi: int = 300, bbox_inches: str = "tight"
) -> None:
    """現在の figure を保存して閉じる。

    Args:
        output_dir (str): 出力ディレクトリ。
        filename (str): 保存ファイル名。
        dpi (int, optional): 保存 DPI。Defaults to 300.
        bbox_inches (str, optional): 余白設定。Defaults to "tight".
    """
    save_path = os.path.join(output_dir, filename)
    save_figure(save_path, dpi=dpi, bbox_inches=bbox_inches)


def _compute_trajectory_scaling(
    target_peak_amp: float | None, A_val: float, label: str
) -> float:
    """軌跡可視化のスケーリング係数を計算する。

    Args:
        target_peak_amp (float | None): 目標ピーク振幅。
        A_val (float): 対象ビンの振幅。
        label (str): ログ表示用ラベル。

    Returns:
        float: スケーリング係数。
    """
    scale = 1.0
    if target_peak_amp is not None and A_val > SCALE_THRESHOLD:
        scale = target_peak_amp / A_val
        print(
            f"Scaling {label} trajectory by {scale:.4f} "
            f"(Target Amp: {target_peak_amp:.4f} / STFT Amp: {A_val:.4f})"
        )
    return scale


def plot_error_curve(history: dict, output_dir: str) -> None:
    """GLA 反復時の誤差推移を描画する。

    Args:
        history (dict): 反復履歴。
        output_dir (str): 出力ディレクトリ。
    """
    k_list = np.arange(len(history["e_mag"]))
    plt.figure(figsize=(10, 6))
    plt.plot(
        k_list, history["e_mag"], label=r"$E_{mag}$ (Amplitude Error)", linewidth=2
    )
    plt.plot(
        k_list,
        history["d_c"],
        label=r"$D_{\mathcal{C}}$ (Consistency Loss)",
        linewidth=2,
        linestyle="--",
    )
    if "phase_mae" in history:
        plt.plot(
            k_list,
            history["phase_mae"],
            label=r"Phase MAE to $X_{\star}$ [rad]",
            linewidth=2,
            linestyle="-.",
        )
    plt.xlabel("Iteration $k$")
    plt.ylabel("Error (log scale)")
    plt.yscale("log")
    plt.title("Convergence of GLA (Error Curves)")
    plt.legend()
    plt.grid(True, which="both", ls="-")
    _save_current_figure(output_dir, "error_curve.png")


def plot_trajectory_static(
    trajectory: list[dict],
    A_val: float,
    true_val: complex,
    output_dir: str,
    target_peak_amp: float | None = None,
) -> None:
    """複素平面で軌跡を静止画として描画する。

    Args:
        trajectory (list[dict]): 反復軌跡。
        A_val (float): 目標振幅。
        true_val (complex): 参照複素値。
        output_dir (str): 出力ディレクトリ。
        target_peak_amp (float | None, optional): スケーリング基準。Defaults to None.
    """
    scale = _compute_trajectory_scaling(target_peak_amp, A_val, "static")
    eff_A_val = A_val * scale
    eff_true_val = true_val * scale
    plt.figure(figsize=(8, 8))

    circle = plt.Circle(
        (0, 0),
        eff_A_val,
        color="lightgray",
        fill=False,
        linestyle="--",
        label=r"Amplitude Set $\mathcal{M}$",
    )
    plt.gca().add_artist(circle)
    plt.plot(
        eff_true_val.real,
        eff_true_val.imag,
        "r*",
        markersize=15,
        label=r"True Value $X_{\star}$",
    )

    cmap = plt.get_cmap("viridis")
    n_steps = len(trajectory)
    for k in range(n_steps):
        color = cmap(k / n_steps)
        xk = trajectory[k]["X_k"] * scale
        x_hat_k = trajectory[k]["X_hat_k"] * scale
        x_k_next = trajectory[k]["X_k_next"] * scale
        plt.plot(
            [xk.real, x_hat_k.real],
            [xk.imag, x_hat_k.imag],
            color=color,
            linestyle="-",
            linewidth=1.5,
            alpha=0.8,
        )
        plt.plot(
            [x_hat_k.real, x_k_next.real],
            [x_hat_k.imag, x_k_next.imag],
            color=color,
            linestyle=":",
            linewidth=1.5,
            alpha=0.8,
        )
        if k == 0:
            plt.plot(
                xk.real,
                xk.imag,
                "kx",
                markersize=10,
                markeredgewidth=2,
                label="Start ($k=0$)",
            )
        if k == n_steps - 1:
            plt.plot(
                x_k_next.real,
                x_k_next.imag,
                "ko",
                markersize=8,
                label="End ($k=K$)",
            )

    plt.xlabel("Real")
    plt.ylabel("Imaginary")
    plt.title("Trajectory in Complex Plane (bin value)")
    plt.axis("equal")
    plt.legend(loc="upper right")
    limit = max(eff_A_val * PLOT_LIMIT_MARGIN, abs(eff_true_val) * PLOT_LIMIT_MARGIN)
    plt.xlim(-limit, limit)
    plt.ylim(-limit, limit)
    _save_current_figure(output_dir, "trajectory_static.png")


def plot_waveform_comparison(
    t: np.ndarray,
    x_true: np.ndarray,
    x_recon: np.ndarray,
    output_dir: str,
    highlight_t: float | None = None,
) -> None:
    """元波形と再構成波形を全体/拡大で比較描画する。

    Args:
        t (np.ndarray): 時間軸。
        x_true (np.ndarray): 参照波形。
        x_recon (np.ndarray): 再構成波形。
        output_dir (str): 出力ディレクトリ。
        highlight_t (float | None, optional): 強調時刻。Defaults to None.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    min_len = min(len(x_true), len(x_recon))
    t = t[:min_len]
    x_true = x_true[:min_len]
    x_recon = x_recon[:min_len]

    if highlight_t is not None:
        idx_hl = np.argmin(np.abs(t - highlight_t))
        val_hl = x_true[idx_hl]
        center_idx = idx_hl
    else:
        center_idx = len(t) // 2
        idx_hl = None

    ax1.plot(t, x_true, "k-", alpha=0.5, label=r"Original $x_{\star}$")
    ax1.plot(t, x_recon, "r--", alpha=0.8, label="Reconstructed $x^{(K)}$")
    if idx_hl is not None:
        ax1.plot(t[idx_hl], val_hl, "r*", markersize=15, label="Selected Frame")
    ax1.set_title("Waveform Comparison (Full)")
    ax1.set_ylabel("Amplitude")
    ax1.legend(loc="upper right")

    sr = len(t) / t[-1] if t[-1] > 0 else 22050
    window_samples = int(ZOOM_WINDOW_SEC * sr)
    start = max(0, center_idx - window_samples // 2)
    end = min(len(t), center_idx + window_samples // 2)
    ax2.plot(t[start:end], x_true[start:end], "k-", alpha=0.5, label="Original")
    ax2.plot(t[start:end], x_recon[start:end], "r--", alpha=0.8, label="Reconstructed")
    if idx_hl is not None and start <= idx_hl < end:
        ax2.plot(t[idx_hl], val_hl, "r*", markersize=15)
    ax2.set_title("Waveform Comparison (Zoom 0.1s)")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Amplitude")
    plt.tight_layout()
    _save_current_figure(output_dir, "waveform_comparison.png")


def create_combined_animation(
    trajectory: list[dict],
    waveforms: list[np.ndarray],
    A_val: float,
    true_val: complex,
    t: np.ndarray,
    x_true: np.ndarray,
    output_dir: str,
    highlight_t: float | None = None,
    target_peak_amp: float | None = None,
) -> None:
    """軌跡と波形進化を同時に示すアニメーションを作成する。

    Args:
        trajectory (list[dict]): 反復軌跡。
        waveforms (list[np.ndarray]): 各反復の波形。
        A_val (float): 目標振幅。
        true_val (complex): 参照複素値。
        t (np.ndarray): 時間軸。
        x_true (np.ndarray): 参照波形。
        output_dir (str): 出力ディレクトリ。
        highlight_t (float | None, optional): 強調時刻。Defaults to None.
        target_peak_amp (float | None, optional): スケーリング基準。Defaults to None.
    """
    scale = _compute_trajectory_scaling(target_peak_amp, A_val, "animation")
    eff_A_val = A_val * scale
    eff_true_val = true_val * scale

    fig = plt.figure(figsize=(18, 8))
    gs = fig.add_gridspec(2, 2)
    ax_traj = fig.add_subplot(gs[:, 0])
    ax_wave_full = fig.add_subplot(gs[0, 1])
    ax_wave_zoom = fig.add_subplot(gs[1, 1])

    limit = max(eff_A_val * PLOT_LIMIT_MARGIN, abs(eff_true_val) * PLOT_LIMIT_MARGIN)
    ax_traj.set_xlim(-limit, limit)
    ax_traj.set_ylim(-limit, limit)
    ax_traj.set_aspect("equal")
    circle = plt.Circle(
        (0, 0),
        eff_A_val,
        color="lightgray",
        fill=False,
        linestyle="--",
        label=r"Amplitude Set $\mathcal{M}$",
    )
    ax_traj.add_artist(circle)
    ax_traj.plot(
        eff_true_val.real,
        eff_true_val.imag,
        "r*",
        markersize=15,
        label=r"True Value $X_{\star}$",
    )
    ax_traj.set_title("Trajectory in Complex Plane")
    ax_traj.set_xlabel("Real")
    ax_traj.set_ylabel("Imaginary")
    ax_traj.legend(loc="upper right")
    (line_proj,) = ax_traj.plot([], [], "b-", alpha=0.8)
    (line_cons,) = ax_traj.plot([], [], "b:", alpha=0.8)
    (marker_curr,) = ax_traj.plot([], [], "kx", markersize=8)

    min_len = len(t)
    if len(x_true) != min_len:
        min_len = min(len(x_true), min_len)
        t = t[:min_len]
        x_true = x_true[:min_len]

    all_max = max(np.max(np.abs(x_true)), 1e-6)
    for w in waveforms:
        w_curr = w[:min_len] if len(w) >= min_len else w
        current_max = np.max(np.abs(w_curr)) if len(w_curr) > 0 else 0
        if current_max > all_max:
            all_max = current_max

    y_limit = all_max * 1.1
    ax_wave_full.plot(t, x_true, "k-", alpha=0.3, label="Original")
    ax_wave_full.set_xlim(t[0], t[-1])
    ax_wave_full.set_ylim(-y_limit, y_limit)
    ax_wave_full.set_title("Waveform Evolution (Full)")
    ax_wave_full.set_ylabel("Amplitude")
    (line_wave_full,) = ax_wave_full.plot(
        [], [], "r--", alpha=0.8, label="Reconstructed"
    )

    if highlight_t is not None:
        idx_hl = np.argmin(np.abs(t - highlight_t))
        val_hl = x_true[idx_hl]
        ax_wave_full.plot(
            t[idx_hl], val_hl, "r*", markersize=15, label="Selected Frame"
        )
        ax_wave_full.legend(loc="upper right")
        center_idx = idx_hl
    else:
        center_idx = len(t) // 2
        idx_hl = None

    sr = len(t) / t[-1] if t[-1] > 0 else 22050
    window_samples = int(ZOOM_WINDOW_SEC * sr)
    start = max(0, center_idx - window_samples // 2)
    end = min(len(t), center_idx + window_samples // 2)
    t_zoom = t[start:end]
    x_true_zoom = x_true[start:end]
    y_limit_zoom = max(np.max(np.abs(x_true_zoom)), 1e-6) * 1.5

    ax_wave_zoom.plot(t_zoom, x_true_zoom, "k-", alpha=0.3, label="Original")
    ax_wave_zoom.set_xlim(t_zoom[0], t_zoom[-1])
    ax_wave_zoom.set_ylim(-y_limit_zoom, y_limit_zoom)
    ax_wave_zoom.set_title("Waveform Evolution (Zoom 0.1s)")
    ax_wave_zoom.set_xlabel("Time [s]")
    ax_wave_zoom.set_ylabel("Amplitude")
    (line_wave_zoom,) = ax_wave_zoom.plot(
        [], [], "r--", alpha=0.8, label="Reconstructed"
    )
    if idx_hl is not None and start <= idx_hl < end:
        ax_wave_zoom.plot(t[idx_hl], x_true[idx_hl], "r*", markersize=15)

    text_iter = ax_traj.text(
        0.05, 0.95, "", transform=ax_traj.transAxes, fontsize=12, fontweight="bold"
    )

    def init():
        line_proj.set_data([], [])
        line_cons.set_data([], [])
        marker_curr.set_data([], [])
        line_wave_full.set_data([], [])
        line_wave_zoom.set_data([], [])
        text_iter.set_text("")
        return (
            line_proj,
            line_cons,
            marker_curr,
            line_wave_full,
            line_wave_zoom,
            text_iter,
        )

    def update(k):
        xs_proj, ys_proj, xs_cons, ys_cons = [], [], [], []
        for i in range(k):
            if i >= len(trajectory):
                break
            xi = trajectory[i]["X_k"] * scale
            xhi = trajectory[i]["X_hat_k"] * scale
            xni = trajectory[i]["X_k_next"] * scale
            xs_proj.extend([xi.real, xhi.real, np.nan])
            ys_proj.extend([xi.imag, xhi.imag, np.nan])
            xs_cons.extend([xhi.real, xni.real, np.nan])
            ys_cons.extend([xhi.imag, xni.imag, np.nan])
        line_proj.set_data(xs_proj, ys_proj)
        line_cons.set_data(xs_cons, ys_cons)

        if k == 0:
            curr = trajectory[0]["X_k"] if len(trajectory) > 0 else 0
        elif k <= len(trajectory):
            curr = trajectory[k - 1]["X_k_next"]
        else:
            curr = trajectory[-1]["X_k_next"] if len(trajectory) > 0 else 0
        if isinstance(curr, (complex, np.complex64, np.complex128)):
            curr = curr * scale
            marker_curr.set_data([curr.real], [curr.imag])
        else:
            marker_curr.set_data([], [])

        w = waveforms[k]
        if len(w) >= min_len:
            current_wave = w[:min_len]
        else:
            current_wave = np.zeros(min_len)
            current_wave[: len(w)] = w
        line_wave_full.set_data(t, current_wave)
        line_wave_zoom.set_data(t_zoom, current_wave[start:end])
        text_iter.set_text(f"Iteration: {k}")
        return (
            line_proj,
            line_cons,
            marker_curr,
            line_wave_full,
            line_wave_zoom,
            text_iter,
        )

    ani = animation.FuncAnimation(
        fig, update, frames=len(waveforms), init_func=init, blit=True
    )
    save_path = os.path.join(output_dir, "combined_evolution.gif")
    ani.save(save_path, writer="pillow", fps=10)
    print(f"Saved: {save_path}")
    plt.close()
