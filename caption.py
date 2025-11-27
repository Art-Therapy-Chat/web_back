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
                    "이미지를 보고 HTP(집-나무-사람) 심리검사 해석에 필요한 그림의 요소들을 구체적으로 설명하는 캡션을 생성하세요.\n\n"
                    "HTP 해석에서 중요한 요소들:\n"
                    "- 크기와 비율 (전체 크기, 부분별 크기, 비율)\n"
                    "- 위치와 배치 (종이 내 위치, 중심, 가장자리)\n"
                    "- 선의 특징 (선의 강도, 굵기, 연속성, 떨림 여부)\n"
                    "- 세부 묘사 정도 (디테일, 생략된 부분, 강조된 부분)\n"
                    "- 구조적 특징 (대칭성, 안정성, 왜곡)\n"
                    "- 추가 요소 (배경, 장식, 부가 물체)\n\n"
                    "집 그림의 경우: 지붕, 벽, 문, 창문, 굴뚝, 울타리 등의 특징\n"
                    "나무 그림의 경우: 뿌리, 줄기, 가지, 나뭇잎, 열매, 크기 등의 특징\n"
                    "사람 그림의 경우: 신체 비율, 얼굴 표정, 자세, 옷차림, 손발 등의 특징\n\n"
                    "출력은 반드시 다음 JSON 형식으로만 작성하세요:\n"
                    "{\n"
                    "  \"ko\": \"HTP 해석에 필요한 그림 요소를 구체적으로 설명한 한국어 캡션\",\n"
                    "  \"en\": \"Detailed English caption describing drawing elements needed for HTP interpretation\"\n"
                    "}\n\n"
                    "캡션 작성 예시:\n"
                    "{\n"
                    "  \"ko\": \"나무는 크고 가지가 많으며 뿌리가 깊게 그려졌다\",\n"
                    "  \"en\": \"The tree is large with many branches and deeply drawn roots\"\n"
                    "}\n\n"
                    "규칙:\n"
                    "- 출력은 반드시 위 JSON 형식만 사용하세요.\n"
                    "- JSON 외의 다른 텍스트, 설명, 줄바꿈 금지.\n"
                    "- 캡션은 HTP 심리검사 해석에 필요한 객관적이고 구체적인 그림 요소를 설명해야 합니다.\n"
                    "- 심리적 해석이나 추론은 포함하지 마세요. 오직 관찰 가능한 그림의 특징만 설명하세요.\n"

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
