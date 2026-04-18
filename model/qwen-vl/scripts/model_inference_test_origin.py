from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
import torch

# ===================== 全局配置（直接用你的路径） =====================
MODEL_PATH = "/apps/users/icps_intelligence/data/dhc/model/qwen-vl/qwen2-vl-7b-instruct"
IMAGE_PATH = "/apps/users/icps_intelligence/data/dhc/data/AS_photos/image.png"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.bfloat16

# ===================== 模型加载（一次性加载，复用） =====================
def load_vl_model():
    """加载Qwen2-VL模型和处理器，全局复用"""
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=TORCH_DTYPE,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    print("✅ Qwen2-VL 模型加载完成！")
    return model, processor

# ===================== 核心统一处理函数（核心！） =====================
def predict(model, processor, text_query: str = None, image_path: str = None):
    """
    统一入口：自动区分 纯文本 / 纯图片 / 图片+文本
    :param text_query: 用户文本问题（可选）
    :param image_path: 图片路径（可选）
    """
    # 1. 加载图片（如果有）
    image = None
    if image_path:
        image = Image.open(image_path).convert("RGB")

    # 2. 构建提示词（自动适配输入类型）
    if image is not None:
        # ============== 场景：有图片（图片/图片+文本）===============
        if text_query and text_query.strip():
            prompt = f"用户描述：{text_query}。请结合影像和描述，完成：1.症状分析评估；2.专业诊疗建议"
        else:
            prompt = "请分析这张强直性脊柱炎影像，完成：1.症状分析评估；2.专业诊疗建议"
        
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}
        ]
    else:
        # ============== 场景：纯文本 ==============
        if not text_query:
            return "❌ 请输入文本问题或上传图片"
        messages = [
            {"role": "user", "content": [{"type": "text", "text": text_query}]}
        ]

    # 3. 模型推理
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=prompt_text,
        images=image,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=800, repetition_penalty=1.2)
    
    # 4. 解析结果
    result = processor.decode(outputs[0], skip_special_tokens=True)
    return result

# ===================== 测试主函数（一键测试3种场景） =====================
if __name__ == "__main__":
    # 加载模型
    model, processor = load_vl_model()

    print("\n" + "="*50)
    print("测试1：纯文本输入")
    print("="*50)
    res1 = predict(model, processor, text_query="强直性脊柱炎的诊断标准是什么")
    print(res1)

    print("\n" + "="*50)
    print("测试2：仅图片输入")
    print("="*50)
    res2 = predict(model, processor, image_path=IMAGE_PATH)
    print(res2)

    print("\n" + "="*50)
    print("测试3：图片+文本输入（最终演示场景）")
    print("="*50)
    res3 = predict(model, processor,
                   text_query="我骶髂关节疼了3个月，早晨僵硬",
                   image_path=IMAGE_PATH)
    print(res3)