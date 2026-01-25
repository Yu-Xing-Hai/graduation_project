import json
import os
# 基础计算库
import torch
# PEFT（LoRA核心）
from peft import LoraConfig, get_peft_model
# Transformers（加载Qwen模型/tokenizer）
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
# 数据集处理
from datasets import Dataset

# 2. 加载Qwen的tokenizer
model_path = "../qwen-7b-local"
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True,
    padding_side="left"
)
print(f"✅ Qwen Tokenizer加载完成，词汇表大小：{len(tokenizer)}")

# 3. 加载Qwen预训练模型（应用8bit量化）
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.float16
)

lora_config = LoraConfig(
    r=32,  
    lora_alpha=64,  
    target_modules=["c_attn", "c_proj", "w1", "w2"],
    lora_dropout=0.05,  
    bias="none",  
    task_type="CAUSAL_LM"  
)

# 将LoRA配置绑定到Qwen基础模型
model = get_peft_model(model, lora_config)

# 打印可训练参数比例（验证LoRA配置生效）
model.print_trainable_parameters()

json_file_path = "~/data/dhc/data/LoRA/LoRA-textData.json"
abs_json_path = os.path.expanduser(json_file_path)

try:
    with open(abs_json_path, "r", encoding="utf-8") as f:
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
        text = f"""### 指令：回答需采用临床诊疗分析的段落/分点形式，禁止使用A/B/C/D、①/②/③等选择题格式，所有内容需逻辑连贯、贴合强直性脊柱炎临床诊疗规范。
### 问题：根据以下强直性脊柱炎患者的症状组合，判定其风险等级、说明风险判定依据（含指南规则+分级标准+核心原因），并给出针对性诊疗建议：{item['symptom_combination']}
### 回答：
该患者的风险等级为{item['risk_level']}。
**判定依据**：
- 指南规则依据：{item['guideline_rule'].replace('1. ', '').replace('2. ', '').replace('3. ', '')}
- 风险分级标准：{item['risk_grading_standard'].replace('1. ', '').replace('2. ', '').replace('3. ', '').replace('4. ', '').replace('5. ', '')}
- 核心风险原因：{item['risk_reason']}
**针对性诊疗建议**：{item['suggestion'].replace('1. 西医干预：', '').replace('2. 中医干预：', '').replace('3. 随访与生活方式：', '').replace('1.', '').replace('2.', '').replace('3.', '')}
"""
        
        as_qa_data.append({"text": text})  # 转为之前的text字段格式
    
    print(f"✅ 数据拼接完成，生成 {len(as_qa_data)} 条微调文本数据")
            
except FileNotFoundError:
    print(f"❌ 找不到JSON文件，请检查路径：{abs_json_path}")
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
def custom_data_collator(features):
    """
    手动处理padding，无需依赖tokenizer的pad_token：
    1. 提取input_ids和labels
    2. 手动padding到batch内最长长度
    3. labels的padding部分设为-100（CrossEntropy忽略）
    """
    # 提取所有input_ids和labels（tokenize后的结果）
    input_ids = [torch.tensor(f["input_ids"]) for f in features]
    labels = [torch.tensor(f["input_ids"]) for f in features]  # 因果LM的labels=input_ids
    
    # 手动padding到batch内最长长度
    max_len = max([len(ids) for ids in input_ids])
    if tokenizer.eos_token_id is not None:
        pad_id = tokenizer.eos_token_id
    else:
        # 方案1：用词汇表最后一个有效ID（你的词汇表大小是151851，ID范围0-151850）
        pad_id = 151850

    input_ids_padded = []
    for ids in input_ids:
        pad_len = max_len - len(ids)
        padded = torch.cat([ids, torch.tensor([pad_id]*pad_len, dtype=torch.long)])
        input_ids_padded.append(padded)
    input_ids_padded = torch.stack(input_ids_padded)
    
    # 对labels padding（padding部分设为-100，避免计算loss）
    labels_padded = []
    for lbl in labels:
        pad_len = max_len - len(lbl)
        padded = torch.cat([lbl, torch.tensor([-100]*pad_len, dtype=torch.long)])
        labels_padded.append(padded)
    labels_padded = torch.stack(labels_padded)
    
    return {
        "input_ids": input_ids_padded,
        "labels": labels_padded
    }

# 6. 配置数据整理器
data_collator = custom_data_collator

# ========== 模块5：配置训练参数（适配46GB显存） ==========
training_args = TrainingArguments(
    output_dir="../lora_checkpoints",
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=3,
    # 核心修复1：关闭bf16，启用fp16（解决GradScaler兼容问题）
    fp16=False,
    bf16=False,
    # 核心修复2：梯度检查点保留（省显存）
    gradient_checkpointing=False,
    # 修正参数名：gradient_clip_val（替代错误的gradient_clipping）
    load_best_model_at_end=False,
    evaluation_strategy="no",
    learning_rate=1e-4,
    num_train_epochs=6,
    optim="paged_adamw_8bit",  # 8bit优化器保留，省显存
    ddp_find_unused_parameters=False,
    logging_steps=20,
    report_to="none",
    remove_unused_columns=True,
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
final_lora_path = "../lora_checkpoints/qwen-lora-as-final"
model.save_pretrained(final_lora_path)
print(f"✅ LoRA权重已保存至：{final_lora_path}")

