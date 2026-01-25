import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# ===================== 核心配置（抽离出来，方便修改） =====================
# 基础模型路径（和训练时一致）
BASE_MODEL_PATH = "../qwen-7b-local"
# LoRA权重路径（训练后保存的最终权重）
LORA_WEIGHTS_PATH = "../lora_checkpoints/qwen-lora-as-final"
# 推理设备（固定GPU 0，和训练一致）
DEVICE = "cuda:0"
# 推理参数（可根据需求调整）
INFERENCE_CONFIG = {
    "max_new_tokens": 200,    # 适配训练数据的回答长度
    "do_sample": False,       # 纯贪心解码（无随机性，最精准）
    "repetition_penalty": 1.5, # 加大重复惩罚，避免循环
    "num_beams": 1,           # 贪心解码（单beam）
}

# ===================== 模型加载（封装成函数，复用性强） =====================
def load_as_model(base_model_path: str, lora_path: str) -> tuple:
    """
    加载基础模型 + AS（强直性脊柱炎）LoRA权重
    :param base_model_path: 本地Qwen-7B模型路径
    :param lora_path: 训练好的LoRA权重路径
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
        lora_model = PeftModel.from_pretrained(base_model, lora_path)
        lora_model = lora_model.to(DEVICE)
        # 推理模式（禁用梯度，提升速度、节省显存）
        lora_model.eval()

        print(f"✅ 模型加载完成，使用设备：{DEVICE}")
        return lora_model, tokenizer

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
        model, tokenizer = load_as_model(BASE_MODEL_PATH, LORA_WEIGHTS_PATH)

        # 2. 测试AS相关问题
        # 先测试训练数据里有的问题（比如症状→风险等级），能验证模型是否正常
        test_prompt = """### 任务背景：基于《非放射学中轴型脊柱关节炎诊疗指南（2024版）》《强直性脊柱炎诊疗规范（2024版）》《强直性脊柱炎病证结合诊疗指南（2024）》，完成强直性脊柱炎患者的风险评估与诊疗建议输出。
### 患者症状：发病年龄40岁，中年男性，2种非甾体抗炎药（NSAIDs）治疗无效，骶髂关节CT显示双侧Ⅱ级损伤，存在骨侵蚀表现。
### 输出要求：
1. 风险等级：仅按“Ⅹ级（××风险）”格式明确判定（如“Ⅱ级（中等风险）”“Ⅲ级（中-高风险）”）；
2. 判定依据：分三部分清晰阐述——
   （1）指南规则依据：对应上述2024版指南的具体推荐意见；
   （2）风险分级标准依据：匹配Ⅰ/Ⅱ/Ⅲ/Ⅳ级风险分级的核心判定条件（需结合症状、炎症指标、结构损伤、治疗状态等维度）；
   （3）核心风险原因：结合患者症状，说明判定该风险等级的核心逻辑（如结构损伤、治疗应答、年龄等因素）；
3. 诊疗建议：分“西医干预”“中医干预”“随访与生活方式”三部分给出针对性建议，需贴合临床实操且匹配指南推荐；
4. 格式约束：禁止使用选择题形式（如A/B/C/D、①/②/③等选项式表述），所有内容采用“1./2./3.”分点或段落式呈现，语言专业、逻辑连贯，贴合临床诊疗语境。
### 回答："""
        print(f"📝 输入问题：{test_prompt.split('### 回答：')[0]}")
        
        # 3. 执行推理
        result = as_inference(model, tokenizer, test_prompt)
        print(f"💡 模型回答：\n{result}")

    except Exception as e:
        print(f"❌ 运行出错：{e}")