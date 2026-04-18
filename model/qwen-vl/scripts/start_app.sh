#!/bin/bash
# 强直性脊柱炎智能诊疗助手启动脚本

echo "🚀 启动强直性脊柱炎智能诊疗助手..."
echo "================================"

# 激活 conda 环境
source /apps/users/icps_intelligence/.conda/etc/profile.d/conda.sh
conda activate qwen_vl

# 检查端口占用
if netstat -tlnp 2>/dev/null | grep -q ":7860 "; then
    echo "⚠️  端口 7860 已被占用，正在终止旧进程..."
    pkill -f "python.*app_ui.py"
    sleep 2
fi

# 进入脚本目录
cd "$(dirname "$0")"

# 启动应用
echo "📡 启动 Gradio 服务..."
python3 app_ui.py

# 如果异常退出，显示错误信息
if [ $? -ne 0 ]; then
    echo "❌ 应用启动失败！"
    echo "请检查上面的错误信息。"
    exit 1
fi