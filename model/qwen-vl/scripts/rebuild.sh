#!/bin/bash
# 出错立即停止，防止前面命令失败还继续跑（毕设必备！）
set -e

echo "======== 1. 删除旧向量库 ========"
bash delete_chroma_db.sh

echo "======== 2. 删除旧LoRA权重 ========"
bash delete_LoRA_weight.sh

echo "======== 3. 训练LoRA模型 ========"
python model_train_LoRA.py

echo "======== 4. 构建RAG知识库 ========"
python RAG_build.py

echo "🎉 rebuild 全流程执行完成，可以开始推理测试！"