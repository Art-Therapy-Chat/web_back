from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 전역 변수로 모델과 토크나이저를 한 번만 로드
_model = None
_tokenizer = None
_model_name = "helena29/Qwen2.5_LoRA_for_HTP"

def _load_model():
    """모델을 한 번만 로드 (싱글톤 패턴)"""
    global _model, _tokenizer
    
    if _model is None:
        print(f"🔥 Loading Qwen HTP Model: {_model_name}")
        
        # 토크나이저 로드
        _tokenizer = AutoTokenizer.from_pretrained(_model_name)
        
        # 모델 로드 (LoRA 어댑터가 이미 병합된 상태)
        _model = AutoModelForCausalLM.from_pretrained(
            _model_name,
            device_map="auto",
            torch_dtype="auto"
        )
        
        print("✅ Qwen HTP Model loaded successfully!")
    
    return _model, _tokenizer


def generate_with_qwen(prompt: str):
    """
    Qwen 모델을 사용해 텍스트 생성
    모델은 최초 1회만 로드되고 재사용됨
    """
    # 모델 로드 (이미 로드되어 있으면 재사용)
    model, tokenizer = _load_model()
    
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
    
    return result
