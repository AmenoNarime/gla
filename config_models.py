"""設定モデル定義。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class ConfigModel(BaseModel):
    """設定モデルの共通基底クラス。"""

    model_config = ConfigDict(extra="forbid")


class AudioConfig(ConfigModel):
    """音声処理設定。"""

    sr: int
    duration: float | None = None
    f0: float | None = None
    tau: float | None = None


class StftConfig(ConfigModel):
    """STFT 設定。"""

    n_fft: int
    hop_length: int
    window: str
    win_length: int | None = None
    center: bool | None = None
    pad_mode: str | None = None


class GlaConfig(ConfigModel):
    """Griffin-Lim 設定。"""

    n_fft: int
    hop_length: int
    window: str
    win_length: int | None = None
    n_iter: int
    momentum: float | None = None
    init_type: str | None = None
    noise_scale: float | None = None


class OutputConfig(ConfigModel):
    """出力設定。"""

    output_dir: str


class BaseProcessingConfig(ConfigModel):
    """音声/STFT/GLA を共通で持つ設定。"""

    audio: AudioConfig
    stft: StftConfig
    gla: GlaConfig


class BaseExperimentConfig(BaseProcessingConfig):
    """実験設定の共通ベース。"""


class GlaVisualizationConfig(BaseExperimentConfig):
    """GLA 可視化実験用設定。"""

    experiment_type: Literal["gla_visualization"] = "gla_visualization"
    output: OutputConfig
