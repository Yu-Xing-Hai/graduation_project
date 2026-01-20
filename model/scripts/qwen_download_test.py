import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ******** 双重镜像配置：确保国内镜像生效，延长超时时间 ********
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 国内镜像地址
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"  # 延长下载超时时间至5分钟，应对网络波动

# 2. 模型核心配置（贴合你的目录结构，方便后续迁移权重）
MODEL_NAME = "../qwen-7b-local"
LOCAL_MODEL_PATH = "../qwen-7b-local"  # 后续迁移缓存权重的目标目录（对应model/qwen-7b-local）

# 3. 加载分词器（先下载小体积分词器，快速验证环境是否正常）
print("="*50)
print("开始下载分词器文件...（体积较小，约几百KB，很快完成）")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,  # Qwen系列模型必需：加载阿里自定义分词逻辑和模型结构
    cache_dir="./hf_cache"  # 临时缓存目录（自动创建，无需手动新建，方便后续迁移权重）
)
print("分词器下载完成！✅")
print("="*50)

# 4. 加载Qwen-7B模型权重（核心步骤，适配transformers 4.35.2，消除废弃警告）
print("开始下载Qwen-7B模型权重...（约13GB，分2个文件，支持断点续传，耐心等待）")
print("温馨提示：你的L40 GPU有45G显存，完全足够，无需担心显存溢出～")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,  # Qwen系列模型必需参数
    torch_dtype=torch.float16,  # 改回torch_dtype，适配Qwen自定义模型类
    device_map="auto",  # 自动分配模型到L40 GPU运行
    cache_dir="./hf_cache",  # 已缓存完整权重，无需重新下载
    low_cpu_mem_usage=True  # 优化CPU内存占用
)
print("模型权重下载并加载完成！✅")
print("="*50)

# 5. 模型简单测试（强直诊疗场景，优化后降低医疗幻觉，更精准）
print("开始运行强直诊疗场景测试...")
test_query = "强直性脊柱炎合并炎症性肠病（IBD），TNFi治疗无效，推荐什么药物？依据是什么？"

# 编码输入文本，自动迁移到GPU（L40）进行推理
inputs = tokenizer(test_query, return_tensors="pt").to("cuda")

# 生成回答（优化医疗场景精准性，降低幻觉）
outputs = model.generate(
    **inputs,
    max_new_tokens=128,  # 回答最大长度，足够容纳药物推荐+依据
    temperature=0.1,  # 低温度值，保证回答的事实性和稳定性，避免胡编乱造
    do_sample=False  # 禁用随机采样，优先输出基于训练语料的事实性内容，医疗场景更精准
)

# 解码输出，去除特殊字符，得到清晰回答
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

# 打印测试结果
print("模型测试输出：")
print("-"*50)
print(response)
print("-"*50)
print("测试完成！✅")
