# GLA 実験コード

このリポジトリは Griffin-Lim Algorithm (GLA) の検証用コードです。

## 実行

```bash
python main.py
```

- `main.py` で GLA 可視化実験を実行します。
- 個別実行したい場合は `gla_visualization.runner` を直接呼び出してください。

### 例

```bash
# デフォルト設定で実行
python main.py

# 任意設定で個別に実行（runnerを直接呼び出す例）
python -c "from gla_visualization.runner import run_gla_visualization; run_gla_visualization('config/perturbed.yaml')"
```
