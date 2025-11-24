from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def generate_with_qwen(prompt: str):
    # Hugging Face에서 파인튜닝된 모델 직접 로드
    model_name = "helena29/Qwen2.5_LoRA_for_HTP"
    
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 모델 로드 (LoRA 어댑터가 이미 병합된 상태)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto"
    )
    
    # 입력 텐서 준비
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # 생성
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7
    )
    
    # 프롬프트 제거: 입력 토큰 이후만 추출
    input_len = inputs["input_ids"].shape[1]
    generated_ids = outputs[0][input_len:]
    
    result = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    
    # 모델 메모리 해제
    del model
    torch.cuda.empty_cache()
    
    return result
