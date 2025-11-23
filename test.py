from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

def load_model(model_name=MODEL_NAME):
    print("📥 Loading model... (will use HuggingFace cache)")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"   # GPU 있으면 GPU 자동 사용
    )
    print(f"[Qwen] model device: {model.device}")
    print(f"[Qwen] torch.cuda.is_available: {torch.cuda.is_available()}")
    print("✅ Model loaded and cached!")
    return tokenizer, model


def ask(model, tokenizer, prompt: str):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    tokenizer, model = load_model()
    print("🔥 Ready to infer.")

    # 간단한 테스트 프롬프트
    prompt = "안녕! 너는 누구야?"
    result = ask(model, tokenizer, prompt)

    print("\n=== 모델 출력 ===")
    print(result)
