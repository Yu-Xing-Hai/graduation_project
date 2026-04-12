import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from config import *

# ===================== 模型加载（封装成函数，复用性强） =====================
def load_as_model(base_model_path: str) -> tuple:
    """
    加载基础模型 + AS（强直性脊柱炎）LoRA权重
    :param base_model_path: 本地Qwen-7B模型路径
    :return: (加载好的模型, tokenizer)
    """
    try:
        # 1. 加载Tokenizer（和训练时配置一致）
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            padding_side="left"
        )
        # 设置pad_token（避免推理时警告）
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

        # 2. 加载基础模型（统一float16精度，和训练一致）
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            device_map={"": DEVICE},
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )

        # 推理模式（禁用梯度，提升速度、节省显存）
        base_model.eval()

        print(f"✅ 模型加载完成，使用设备：{DEVICE}")
        return base_model, tokenizer

    except FileNotFoundError as e:
        raise FileNotFoundError(f"模型/权重路径错误：{e}")
    except Exception as e:
        raise RuntimeError(f"模型加载失败：{e}")

# ===================== 推理函数（封装，可批量调用） =====================
def as_inference(model, tokenizer, prompt: str) -> str:
    """
    AS（强直性脊柱炎）问题推理
    :param model: 加载好的LoRA模型
    :param tokenizer: 对应tokenizer
    :param prompt: 输入的问题（需符合训练时的格式：### 问题：xxx ### 回答：）
    :return: 模型生成的完整回答
    """
    try:
        # ========== 核心修改：手动padding（复用训练时的逻辑） ==========
        # 1. 编码时关闭自动padding，只做truncation
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=False,  # 关键：关掉自动padding
            truncation=True,
            max_length=1024  # 和训练时的max_length一致
        )
        
        # 2. 提取input_ids，手动padding（和训练时用同一个pad_id）
        input_ids = inputs["input_ids"]
        max_len = 1024  # 训练时的最大长度，保持一致
        pad_len = max_len - input_ids.shape[1]
        
        # 确定pad_id（和训练时完全一样：优先eos_token_id，兜底151850）
        if tokenizer.eos_token_id is not None:
            pad_id = tokenizer.eos_token_id
        else:
            pad_id = 151850  # 你的词汇表最后一个有效ID
        
        # 手动padding input_ids（左边padding，和训练时padding_side="left"一致）
        if pad_len > 0:
            pad_tensor = torch.tensor([[pad_id]*pad_len], dtype=torch.long)
            input_ids = torch.cat([pad_tensor, input_ids], dim=1)
        
        # 3. 手动构造attention_mask（padding部分设为0，有效部分设为1）
        attention_mask = torch.ones_like(input_ids)
        if pad_len > 0:
            attention_mask[:, :pad_len] = 0
        
        # 4. 重构inputs并转到指定设备
        inputs = {
            "input_ids": input_ids.to(DEVICE),
            "attention_mask": attention_mask.to(DEVICE)
        }
        # ========== 手动padding逻辑结束 ==========

        # 生成回答（使用配置参数）
        with torch.no_grad():  # 禁用梯度计算，节省显存
            outputs = model.generate(
                **inputs,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=pad_id,  # 用手动指定的pad_id
                **INFERENCE_CONFIG  # 传入推理参数
            )

        # 解码并提取回答（跳过特殊token）
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 只提取“### 回答：”后的内容（更整洁）
        answer = answer.split("### 回答：")[-1].strip()
        return answer

    except Exception as e:
        raise RuntimeError(f"推理失败：{e}")

# ===================== 测试入口（单独调用，方便调试） =====================
if __name__ == "__main__":
    # 1. 加载模型
    try:
        model, tokenizer = load_as_model(BASE_MODEL_PATH)

        # 2. 测试AS相关问题
        test_prompt = """### 问题：强直性脊柱炎的典型症状有哪些？
### 回答："""
        print(f"📝 输入问题：{test_prompt.split('### 回答：')[0]}")
        
        # 3. 执行推理
        result = as_inference(model, tokenizer, test_prompt)
        print(f"💡 模型回答：\n{result}")

    except Exception as e:
        print(f"❌ 运行出错：{e}")