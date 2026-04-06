"""音声可視化ユーティリティ。"""

import os

import IPython.display
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def _save_figure(save_path: str | None) -> None:
    """現在の figure を保存する。"""
    if not save_path:
        return
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Saved figure to {save_path}")


def set_plot_style() -> None:
    """プロットのスタイルを設定する。

    Qiita記事: https://qiita.com/lilacs/items/a331a8933ec135f63ab1
    """
    plt.rcParams.update(
        {
            "font.size": 10,
            "figure.figsize": [10.0, 5.0],
            "figure.dpi": 100,  # 300は少し重いかもしれないので調整
            "savefig.dpi": 300,
            "figure.titlesize": "large",
            "legend.fontsize": "small",
            "axes.labelsize": "medium",
            "xtick.labelsize": "small",
            "ytick.labelsize": "small",
        }
    )
    # 日本語フォントの設定（環境に合わせて調整が必要）
    if os.name == "nt":
        plt.rcParams["font.family"] = "Meiryo"


def plot_waveform(
    y: np.ndarray,
    sr: int,
    title: str = "Audio Waveform",
    save_path: str | None = None,
) -> None:
    """音声波形を表示する。

    Args:
        y (np.ndarray): 音声データ
        sr (int): サンプリングレート
        title (str): タイトル
        save_path (str, optional): 保存先のパス. Defaults to None.
    """
    plt.figure()
    # librosaのバージョンによって使い分ける
    waveshow = getattr(librosa.display, "waveshow", None)
    if callable(waveshow):
        waveshow(y, sr=sr)
    else:
        waveplot = getattr(librosa.display, "waveplot", None)
        if callable(waveplot):
            waveplot(y, sr=sr)
        else:
            raise AttributeError(
                "librosa.display に waveshow / waveplot がありません。"
            )

    plt.title(title)
    plt.ylabel("Amplitude")
    plt.margins(x=0)
    plt.tight_layout()
    _save_figure(save_path)

    plt.show()


def plot_spectrogram(
    S: np.ndarray,
    sr: int,
    hop_length: int,
    title: str = "Spectrogram",
    save_path: str | None = None,
) -> None:
    """スペクトログラムを表示する。

    Args:
        S (np.ndarray): 振幅スペクトル (Magnitude Spectrogram)
        sr (int): サンプリングレート
        hop_length (int): ホップ長
        title (str): タイトル
        save_path (str, optional): 保存先のパス. Defaults to None.
    """
    # 強度をdB単位へ変換
    S_db = librosa.amplitude_to_db(S, ref=np.max)

    plt.figure()
    librosa.display.specshow(
        S_db, sr=sr, hop_length=hop_length, x_axis="time", y_axis="log"
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title(title)
    plt.tight_layout()
    _save_figure(save_path)

    plt.show()


def play_audio(y: np.ndarray, sr: int) -> IPython.display.Audio:
    """音声を再生する（Jupyter Notebook用）。

    Args:
        y (np.ndarray): 音声データ
        sr (int): サンプリングレート
    Returns:
        IPython.display.Audio: オーディオオブジェクト
    """
    return IPython.display.Audio(y, rate=sr)
