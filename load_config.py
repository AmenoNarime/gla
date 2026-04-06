"""設定ファイル読み込み。"""

import yaml

from config_models import GlaVisualizationConfig


def _load_yaml(config_path: str) -> dict:
    """YAML ファイルを辞書として読み込む。"""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_gla_config(
    config_path: str = "config/default.yaml",
) -> GlaVisualizationConfig:
    """GLA 可視化用設定を読み込む。"""
    raw_config = _load_yaml(config_path)
    return GlaVisualizationConfig.model_validate(raw_config)
