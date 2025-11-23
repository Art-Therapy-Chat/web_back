from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch


def generate_with_qwen(prompt: str):
    base = "Qwen/Qwen2.5-1.5B-Instruct"

    # 토크나이저
    tokenizer = AutoTokenizer.from_pretrained(base)

    # base model
    model = AutoModelForCausalLM.from_pretrained(
        base,
        device_map="auto",
        torch_dtype="auto"
    )

    # LoRA 적용
    peft_model = PeftModel.from_pretrained(
        model,
        "./data/adapter_model",
        adapter_name="qwen_lora"
    )

    # 입력 텐서 준비
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # 생성
    outputs = peft_model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7
    )

    # -------------------------------
    # 🔥 프롬프트 제거: 입력 토큰 이후만 추출
    # -------------------------------
    input_len = inputs["input_ids"].shape[1]
    generated_ids = outputs[0][input_len:]

    result = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # 모델 메모리 해제
    del model
    torch.cuda.empty_cache()

    return result
