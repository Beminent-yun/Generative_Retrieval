# Generated Retrieval

## 环境搭建

建议使用 Python 3.8+，推荐使用虚拟环境：

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 训练与评估

1) 训练 RQVAE（如有需要）：

```bash
python train_rqvae.py
```

2) 训练生成式推荐模型：

```bash
python train.py
```

3) 评估模型（需指定 checkpoint 路径）：

```bash
python evaluate.py --checkpoint checkpoints/rec/best_model.pt --split test
```

如需自定义数据路径、模型参数等，请在 `train.py` 和 `evaluate.py` 的 CONFIG 字典中修改。
