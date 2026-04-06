"""プロット保存ユーティリティ。"""

import os

import matplotlib.pyplot as plt


def save_figure(
    save_path: str,
    dpi: int = 300,
    bbox_inches: str = "tight",
    close_fig: bool = True,
) -> None:
    """現在の figure を保存する。"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches=bbox_inches, dpi=dpi)
    if close_fig:
        plt.close()
    print(f"Saved: {save_path}")
