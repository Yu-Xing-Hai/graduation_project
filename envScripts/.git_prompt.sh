#!/bin/bash
# ========== 你本地验证过的 Git 状态函数 ==========
git_branch_status() {
  # 获取当前分支名（屏蔽错误输出，非 Git 仓库不报错）
  branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null)
  if [ -z "$branch" ]; then
    return  # 非 Git 仓库不显示任何 Git 信息
  fi

  # 检查工作区状态（有未提交/未跟踪文件则标红 + 号）
  if [ -n "$(git status --porcelain 2>/dev/null)" ]; then
    echo -e "\033[0;31m($branch +)\033[0m"  # 红色 (分支 +)
  else
    echo -e "\033[0;32m($branch)\033[0m"    # 绿色 (分支)
  fi
}

# ========== 配置提示符（完全用你的 PS1 格式） ==========
export PS1='\[\033[01;35m\]($CONDA_DEFAULT_ENV)\[\033[00m\] \[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[0m\]$(git_branch_status) \$ '

# ========== 适配 Linux 远程的环境变量（已改 UV_CACHE_DIR） ==========
# 1. 远程 Linux 用 UTF-8 编码（和你本地一致，避免乱码）
export LC_ALL=en_US.UTF-8
# 2. UV 缓存目录：改成你指定的 ~/data/dhc/.uv_cache
export UV_CACHE_DIR=~/data/dhc/.uv_cache

# 自动创建 UV 缓存目录（避免路径不存在报错）
mkdir -p ~/data/dhc/.uv_cache

# 提示配置生效
echo "✅ Git 状态提示符已生效！进入 Git 仓库即可看到分支状态～"
