#!/bin/bash
# 优化：添加venv激活，支持模式选择（train/run）。安装依赖。

source venv/bin/activate || echo "No venv found, proceeding without."
pip install -r requirements.txt
if [ "$1" = "train" ]; then
    python train_models.py
else
    python main.py
fi