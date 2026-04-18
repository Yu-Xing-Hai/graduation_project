import warnings
import logging
import os

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=FutureWarning, message=r".*torch.utils._pytree._register_pytree_node.*")
os.environ["PYTHONWARNINGS"] = "ignore"
logging.basicConfig(level=logging.CRITICAL)
logging.getLogger().disabled = True
logging.getLogger("transformers").disabled = True
logging.getLogger("torch").disabled = True
logging.getLogger("huggingface_hub").disabled = True
logging.getLogger("peft").disabled = True
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TRANSFORMERS_SUPPRESS_LOGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["FLASH_ATTENTION_FORCE_DISABLE"] = "1"

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import *
import sys

# ===================== 模型加载 =====================
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

        # 3. 融合LoRA权重到基础模型
        lora_model = PeftModel.from_pretrained(base_model, LORA_WEIGHTS_PATH)
        lora_model = lora_model.to(DEVICE)
        # 推理模式（禁用梯度，提升速度、节省显存）
        lora_model.eval()

        print(f"✅ 模型加载完成，使用设备：{DEVICE}")
        return lora_model, tokenizer

    except FileNotFoundError as e:
        raise FileNotFoundError(f"模型/权重路径错误：{e}")
    except Exception as e:
        raise RuntimeError(f"模型加载失败：{e}")

# ===================== 推理函数 =====================
def as_inference(model, tokenizer, prompt: str) -> str:
    """
    AS（强直性脊柱炎）问题推理
    :param model: 加载好的LoRA模型
    :param tokenizer: 对应tokenizer
    :param prompt: 输入的问题（需符合训练时的格式：### 问题：xxx ### 回答：）
    :return: 模型生成的完整回答
    """
    try:
        # ========== 手动padding逻辑 ==========
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=1024
        )
        
        input_ids = inputs["input_ids"]
        max_len = 1024
        pad_len = max_len - input_ids.shape[1]
        
        if tokenizer.eos_token_id is not None:
            pad_id = tokenizer.eos_token_id
        else:
            pad_id = 151850
        
        if pad_len > 0:
            pad_tensor = torch.tensor([[pad_id]*pad_len], dtype=torch.long)
            input_ids = torch.cat([pad_tensor, input_ids], dim=1)
        
        attention_mask = torch.ones_like(input_ids)
        if pad_len > 0:
            attention_mask[:, :pad_len] = 0
        
        inputs = {
            "input_ids": input_ids.to(DEVICE),
            "attention_mask": attention_mask.to(DEVICE)
        }
        # ========== 手动padding逻辑结束 ==========

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=pad_id,
                **INFERENCE_CONFIG
            )

        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = answer.split("### 回答：")[-1].strip()
        return answer

    except Exception as e:
        raise RuntimeError(f"推理失败：{e}")

if __name__ == "__main__":
    test_questions = [
        "我早上起来背僵，活动一会就好，是强直性脊柱炎吗？",
        "强直性脊柱炎的典型症状有哪些？",
        "40岁男性，骶髂关节双侧Ⅱ级损伤，风险等级是多少？",
        "强直性脊柱炎的中医证型和干预方案是什么？",
        "我腰疼好几个月了，怎么办？",
        "感冒了应该吃什么药？",
        "膝盖疼是不是AS的症状？"
    ]

    try:
        # 加载纯LoRA模型
        model, tokenizer = load_as_model(BASE_MODEL_PATH)

        print("\n=====================================")
        print("强直性脊柱炎智能诊疗系统 - LoRA版（无RAG）")
        print("=====================================\n")

        # 遍历所有问题，批量测试
        for idx, question in enumerate(test_questions, 1):
            print(f"【测试问题 {idx}】{question}")
            print("-" * 50)

            # 构造LoRA训练时的标准Prompt
            prompt = f"""### 问题：{question}
### 回答："""

            # 纯LoRA推理
            result = as_inference(model, tokenizer, prompt)
            
            # 输出格式和 RAG 版本完全一致
            print(f"【LoRA model回答】\n{result}\n")
            print("-" * 50 + "\n")

        print("\n✅ LoRA model 批量推理测试完成！")

    except Exception as e:
        print(f"❌ 运行出错：{e}")