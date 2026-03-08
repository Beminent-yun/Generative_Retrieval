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
