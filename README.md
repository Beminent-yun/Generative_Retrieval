# Generated Retrieval

## 项目结构

```
├── Amazon_Data.py           # 数据处理脚本
├── Amazon_Dataset.py        # 数据集定义
├── models/
│   ├── RQVAE.py             # RQVAE 模型
│   └── Transformer.py       # 生成式推荐主模型
├── utils.py                 # 工具函数
├── metrics.py               # 评估指标
├── train_rqvae.py           # 训练 RQVAE
├── train.py                 # 训练生成式推荐模型
├── evaluate.py              # 评估脚本
├── requirements.txt         # 依赖包
├── README.md                # 项目说明
```

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

## 一键运行脚本

你可以使用 `run_all.sh` 一键完成依赖安装、训练与评估：

```sh
#!/bin/bash
set -e

# 创建虚拟环境并安装依赖
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 训练 RQVAE
python train_rqvae.py

# 训练生成式推荐模型
python train.py

# 评估（可根据实际 checkpoint 路径调整）
python evaluate.py --checkpoint checkpoints/rec/best_model.pt --split test
```

赋予执行权限后运行：

```bash
chmod +x run_all.sh
./run_all.sh
```

---
如需自定义数据路径、模型参数等，请在 `train.py` 和 `evaluate.py` 的 CONFIG 字典中修改。
