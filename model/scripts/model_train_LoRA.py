import json
import os
# 基础计算库
import torch
# PEFT（LoRA核心，Parameter-Efficient Fine-Tuning，参数高效微调）
from peft import LoraConfig, get_peft_model
# Transformers（加载Qwen模型/tokenizer）
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
# 数据集处理
from datasets import Dataset
from config import *

# 2. 加载Qwen的tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_PATH,
    trust_remote_code=True,
    padding_side="left"
)
print(f"✅ Qwen Tokenizer加载完成，词汇表大小：{len(tokenizer)}")

# 3. 加载Qwen预训练模型（应用8bit量化）
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.float16
)

LORA_CONFIG = LoraConfig(
    r=32,  # LoRA低秩矩阵的秩，秩越大，可训练参数越多，微调能力越强（常规8-64）
    lora_alpha=64,  # LoRA缩放因子，控制权重更新幅度，通常设置为r的2倍
    target_modules=["c_attn", "c_proj", "w1", "w2"],  # 给Qwen模型的注意力层、输出投影层、前馈层插入LoRA
    lora_dropout=0.05,  # LoRA层的dropout率，防止训练过拟合
    bias="none",  # 不训练偏置项，最大化减少参数量
    task_type="CAUSAL_LM"  # 任务类型：因果语言模型（大模型文本生成专用）
)

# 将LoRA配置绑定到Qwen基础模型
model = get_peft_model(model, LORA_CONFIG)

# 打印可训练参数比例（验证LoRA配置生效）
model.print_trainable_parameters()

try:
    with open(JSON_FILE_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)  # 加载JSON数组
    print(f"✅ 成功读取JSON数据，共加载 {len(raw_data)} 条AS医疗数据")
    
    # 校验：必须是数组，且每条数据包含所有必填字段
    required_fields = ["symptom_combination", "guideline_rule", "risk_level", 
                       "risk_grading_standard", "risk_reason", "suggestion"]
    as_qa_data = []  # 存储拼接后的text数据
    for idx, item in enumerate(raw_data):
        # 检查必填字段是否缺失
        missing_fields = [f for f in required_fields if f not in item]
        if missing_fields:
            raise ValueError(f"❌ 第{idx+1}条数据（data_id={item.get('data_id','未知')}）缺失字段：{missing_fields}")
        
        # 核心：拼接成LoRA微调的「问题+回答」text格式
        text = f"""### 问题：{item['symptom_combination']}
### 回答：
风险等级：{item['risk_level']}
判定依据：{item['guideline_rule']}
诊疗建议：{item['suggestion']}
"""
        
        as_qa_data.append({"text": text})  # 转为之前的text字段格式
    
    print(f"✅ 数据拼接完成，生成 {len(as_qa_data)} 条微调文本数据")
            
except FileNotFoundError:
    print(f"❌ 找不到JSON文件，请检查路径：{JSON_FILE_PATH}")
    exit(1)  # 终止程序，避免后续报错
except json.JSONDecodeError:
    print(f"❌ JSON文件格式错误，请检查是否符合JSON数组规范")
    exit(1)
except ValueError as e:
    print(e)
    exit(1)

dataset = Dataset.from_list(as_qa_data)

# 4. 定义数据预处理函数
def preprocess_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=1024,
        return_overflowing_tokens=False
    )

# 5. 批量Tokenize数据集
tokenized_dataset = dataset.map(preprocess_function, batched=True)
print(f"✅ 数据集Tokenize完成，处理后数据量：{len(tokenized_dataset)}")

# 6. 自定义DataCollator（绕过tokenizer的pad token，手动padding）
# 6. 自定义DataCollator（数据整理器）
# 作用：针对Qwen模型无默认pad_token的问题，手动实现文本序列填充，适配大模型训练
def custom_data_collator(features):
    """
    手动处理数据批次的填充(Padding)，替代原生DataCollator，解决Qwen模型pad_token缺失问题：
    1. 从batch数据中提取input_ids和labels
    2. 将所有序列填充到当前批次的最大长度
    3. labels的填充位设置为-100（PyTorch交叉熵损失会自动忽略该值）
    """
    # 从数据特征中提取tokenized后的input_ids，转为PyTorch张量
    input_ids = [torch.tensor(f["input_ids"]) for f in features]
    # 因果语言模型(CAUSAL LM)：标签 = 输入序列，模型需要预测下一个token
    labels = [torch.tensor(f["input_ids"]) for f in features]  
    
    # --------------------- 1. 配置填充参数 ---------------------
    # 获取当前batch中最长的序列长度，所有句子统一填充到该长度
    max_len = max([len(ids) for ids in input_ids])
    # Qwen模型无专用pad_token，使用eos_token作为填充符
    if tokenizer.eos_token_id is not None:
        pad_id = tokenizer.eos_token_id
    else:
        # 兜底方案：使用词汇表最后一个有效ID作为填充符（适配Qwen-7B词汇表）
        pad_id = 151850

    # --------------------- 2. 填充输入序列(input_ids) ---------------------
    input_ids_padded = []
    for ids in input_ids:
        # 计算需要填充的token数量
        pad_len = max_len - len(ids)
        # 左侧填充（与模型输入格式对齐），指定张量类型为long（必须为整型）
        padded = torch.cat([torch.tensor([pad_id]*pad_len, dtype=torch.long), ids])
        input_ids_padded.append(padded)
    # 将列表堆叠为批次张量
    input_ids_padded = torch.stack(input_ids_padded)
    
    # --------------------- 3. 填充标签(labels) ---------------------
    # 标签填充位设为-100，训练时损失函数会忽略这些位置，不计算梯度
    labels_padded = []
    for lbl in labels:
        pad_len = max_len - len(lbl)
        padded = torch.cat([torch.tensor([-100]*pad_len, dtype=torch.long), lbl])
        labels_padded.append(padded)
    labels_padded = torch.stack(labels_padded)
    
    # 返回处理好的批次数据，传入模型训练
    return {
        "input_ids": input_ids_padded,
        "labels": labels_padded
    }

# 6. 配置数据整理器
data_collator = custom_data_collator

# ========== 模块5：配置训练参数（适配46GB显存） ==========
# LoRA模型训练参数配置（适配Qwen-7B单卡GPU训练）
training_args = TrainingArguments(
    # 模型训练权重、日志的输出保存路径
    output_dir="../lora_checkpoints",
    # 模型保存策略：按训练步数保存（而非按轮次）
    save_strategy="steps",
    # 每训练100步，自动保存一次模型权重
    save_steps=100,
    # 最多保留2个最新的模型文件，避免占用过多磁盘空间
    save_total_limit=2,
    # 单张GPU每批次训练的样本数量（小批次适配显存不足的问题）
    per_device_train_batch_size=2,
    # 梯度累积步数：累积3步后更新一次参数，模拟更大的批次训练效果
    gradient_accumulation_steps=3,

    # ------------------- 核心修复参数 -------------------
    # 关闭FP16混合精度训练（避免梯度缩放器报错）
    fp16=False,
    # 关闭BF16混合精度训练（适配消费级显卡，提升兼容性）
    bf16=False,
    # 关闭梯度检查点（关闭后训练速度更快，显存充足时使用）
    gradient_checkpointing=False,
    # ----------------------------------------------------

    # 训练结束后，不自动加载效果最好的模型
    load_best_model_at_end=False,
    # 评估策略：不进行验证（本项目无独立验证集）
    evaluation_strategy="no",
    # 学习率：LoRA微调推荐值，控制模型参数更新幅度
    learning_rate=3e-5,
    # 总训练轮次：全程训练6轮，保证模型充分学习专科知识
    num_train_epochs=2,
    # 8位量化优化器：大幅降低显存占用，单卡可运行7B大模型微调
    optim="paged_adamw_8bit",
    # 关闭分布式训练参数检查（单卡训练专用，避免报错）
    ddp_find_unused_parameters=False,
    # 每训练20步，打印一次训练日志（损失值、进度等）
    logging_steps=20,
    # 不向云端上报训练日志（本地训练专用）
    report_to="none",
    # 自动移除数据集中无用的列，提升训练效率
    remove_unused_columns=True,
    # 固定随机种子：保证训练结果可复现
    seed=42,
)

# ========== 模块6：启动训练 ==========
# 初始化Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

# 3. 启动训练
print("🚀 开始训练AS LoRA模型...")
trainer.train()

# 4. 保存最终LoRA权重
model.save_pretrained(LORA_WEIGHTS_PATH)
print(f"✅ LoRA权重已保存至：{LORA_WEIGHTS_PATH}")

