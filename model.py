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
        print(f"🔍 CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"🔍 CUDA Device: {torch.cuda.get_device_name(0)}")
            print(f"🔍 CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        # 토크나이저 로드
        _tokenizer = AutoTokenizer.from_pretrained(_model_name)
        
        # 모델 로드 (LoRA 어댑터가 이미 병합된 상태)
        _model = AutoModelForCausalLM.from_pretrained(
            _model_name,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        print(f"✅ Qwen HTP Model loaded successfully!")
        print(f"✅ Model Device: {_model.device}")
    
    return _model, _tokenizer


def generate_with_qwen(prompt: str):
    """
    Qwen 모델을 사용해 텍스트 생성
    모델은 최초 1회만 로드되고 재사용됨
    """
    # 모델 로드 (이미 로드되어 있으면 재사용)
    model, tokenizer = _load_model()
    
    print("=" * 80)
    print("📝 [PROMPT] 해석 생성 프롬프트:")
    print("-" * 80)
    print(prompt)
    print("=" * 80)
    
    print(f"🔍 [generate_with_qwen] Model device: {model.device}")
    
    # 입력 텐서 준비
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # 모든 입력을 모델과 같은 디바이스로 이동
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    print(f"🔍 [generate_with_qwen] Input device: {inputs['input_ids'].device}")
    
    # 생성
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # 프롬프트 제거: 입력 토큰 이후만 추출
    input_len = inputs["input_ids"].shape[1]
    generated_ids = outputs[0][input_len:]
    
    result = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    
    print(f"✅ [generate_with_qwen] Generated {len(result)} characters")
    
    return result
