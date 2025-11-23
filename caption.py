import base64
import io
import os
import json
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_caption(image_base64: str) -> str:
    """
    Base64 이미지 → GPT-4o-mini Vision으로 캡션 생성
    반환: JSON 문자열 (예: '{"ko": "...", "en": "..."}')
    """

    # 1) Base64 유효성 검증
    try:
        image_bytes = base64.b64decode(image_base64)
        Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        print("❌ 이미지 디코딩 오류:", e)
        return json.dumps({"ko": "", "en": ""}, ensure_ascii=False)

    try:
        content = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_base64}"
                },
            },
            {
                "type": "text",
                "text": (
                    "이미지를 보고 그림의 내용을 매우 구체적으로 설명한 캡션 두 개와, "
                    "HTP(집-나무-사람) 심리검사 맥락에서 사용할 수 있는 질문 5개를 생성하세요.\n\n"
                    "질문은 반드시 HTP 심리검사에서 다루는 영역만 포함해야 합니다:\n"
                    "- 정서 상태, 안정감, 불안, 스트레스 요인\n"
                    "- 대인 관계 및 가족 관계\n"
                    "- 자아 개념, 자존감, 자기상\n"
                    "- 내적 갈등 또는 무의식적 욕구\n"
                    "- 통제감, 자율성, 대처 방식\n"
                    "- 환경에 대한 태도 및 미래 전망\n"
                    "이 범위를 벗어난 단순 설명형 질문은 절대 금지합니다.\n\n"
                    "출력은 반드시 다음 JSON 형식으로만 작성하세요:\n"
                    "{\n"
                    "  \"ko\": \"자연스러운 한국어 캡션 한 문장\",\n"
                    "  \"en\": \"Natural English caption in one sentence\",\n"
                    "  \"q\": [\n"
                    "      \"HTP 관련 질문1\",\n"
                    "      \"HTP 관련 질문2\",\n"
                    "      \"HTP 관련 질문3\",\n"
                    "      \"HTP 관련 질문4\",\n"
                    "      \"HTP 관련 질문5\"\n"
                    "  ]\n"
                    "}\n\n"
                    "규칙:\n"
                    "- 출력은 반드시 위 JSON 형식만 사용하세요.\n"
                    "- JSON 외의 다른 텍스트, 설명, 줄바꿈 금지.\n"
                    "- 질문은 반드시 5개이며 모두 문자열이어야 합니다.\n"
                    "- 질문은 반드시 HTP 심리검사 문맥을 반영한 심리적 탐색 질문이어야 합니다.\n"

                ),
            },
        ]

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": content}],
            max_tokens=500,
        )

        message = response.choices[0].message
        content_field = message.content

        # 🔹 content가 문자열인 경우
        if isinstance(content_field, str):
            raw_text = content_field.strip()
        # 🔹 content가 파트 리스트인 경우
        else:
            text_parts = []
            for part in content_field:
                if getattr(part, "type", None) == "text":
                    text_parts.append(part.text)
            raw_text = "".join(text_parts).strip()

        # JSON 파싱
        try:
            obj = json.loads(raw_text)
        except Exception:
            print("⚠️ GPT JSON 파싱 실패, 원본:", raw_text)
            obj = {"ko": "", "en": ""}

        return json.dumps(obj, ensure_ascii=False)

    except Exception as e:
        print("❌ GPT 요청 오류:", e)
        return json.dumps({"ko": "", "en": ""}, ensure_ascii=False)
