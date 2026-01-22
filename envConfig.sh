#!/bin/bash
# 1. 设置HF镜像（无后台执行，直接生效）
export HF_ENDPOINT=https://hf-mirror.com

# 2. 激活conda环境
conda activate qwen_as

# 3. 加载Git提示符（用绝对路径，任意目录执行都能找到）
source ~/data/dhc/script/.git_prompt.sh

# 提示所有配置生效
echo "✅ 环境配置完成！当前conda环境：$(conda info --envs | grep '*' | awk '{print $1}')，Git提示符已加载～"
