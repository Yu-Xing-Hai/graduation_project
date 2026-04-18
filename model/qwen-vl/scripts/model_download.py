from modelscope import snapshot_download

# 下载 Qwen2-VL-Instruct-7B 官方最新版
model_dir = snapshot_download(
    model_id="qwen/Qwen2-VL-7B-Instruct",
    local_dir="../qwen2-vl-7b-instruct",  # 下载到当前文件夹
    revision="master"
)

print(f"模型下载完成！路径：{model_dir}")