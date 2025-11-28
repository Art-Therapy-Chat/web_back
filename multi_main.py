from dotenv import load_dotenv
load_dotenv()
from embeddings import vectorstore           # 벡터 DB
from rag_engine import AdvancedConversationalRAG  # 멀티쿼리 RAG 엔진

from fastapi import FastAPI
from pydantic import BaseModel
from caption import generate_caption
from model import generate_with_qwen
from fastapi.middleware.cors import CORSMiddleware
import logging
import json

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()
# CORS 설정 - 더 명시적으로 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 출처 허용
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

# -------------------------------------
# RAG 엔진 초기화
# -------------------------------------
rag = AdvancedConversationalRAG(vectorstore)

# ----------------------------- #
# 1) 이미지 캡션 생성
# ----------------------------- #
class CaptionRequest(BaseModel):
    image_base64: str

@app.post("/caption")
def caption(req: CaptionRequest):
    logger.info("=" * 80)
    logger.info("📸 [CAPTION] 이미지 캡션 생성 시작")
    logger.info("🤖 사용 모델: Florence-2-large")
    logger.info(f"입력 이미지 크기: {len(req.image_base64)} bytes")
    
    caption = generate_caption(req.image_base64)
    
    logger.info(f"✅ [CAPTION] 생성된 캡션: {caption}")
    logger.info("=" * 80)
    return {"caption": caption}

# ----------------------------- #
# 2) 멀티쿼리 기반 RAG 검색
# ----------------------------- #
class RagRequest(BaseModel):
    caption: str
    image_type: str    # "집" | "나무" | "사람"

@app.post("/rag")
def rag_search_api(req: RagRequest):
    logger.info("=" * 80)
    logger.info("🔍 [RAG] RAG 검색 시작 (검색 전용 모드)")
    logger.info("🤖 쿼리 재작성 모델: GPT-4o (OpenAI)")
    logger.info(f"입력 캡션: {req.caption}")
    logger.info(f"이미지 타입: {req.image_type}")
    
    try:
        # search_only 메서드 사용 (해석 생성 제거)
        result = rag.search_only(req.caption, req.image_type)
        
        logger.info(f"✅ [RAG] 검색 완료")
        logger.info(f"재작성된 쿼리: {result.get('rewritten_queries', [])}")
        logger.info(f"검색된 문서 수: {len(result.get('rag_docs', []))}")
        
        # 각 문서의 내용 출력
        for idx, doc in enumerate(result.get('rag_docs', []), 1):
            logger.info(f"\n📄 문서 {idx}:")
            logger.info(f"  내용: {doc[:200]}..." if len(doc) > 200 else f"  내용: {doc}")
        
        logger.info("=" * 80)
        return result
        
    except Exception as e:
        logger.error(f"❌ [RAG] 검색 실패: {str(e)}")
        logger.error(f"에러 타입: {type(e).__name__}")
        import traceback
        logger.error(f"스택 트레이스:\n{traceback.format_exc()}")
        logger.info("=" * 80)
        
        # 빈 결과 반환 (에러 발생 시)
        return {
            "rewritten_queries": [req.caption],
            "rag_docs": [],
            "error": str(e)
        }

# ----------------------------- #
# 3) Qwen 로라 모델 개별 해석
# ----------------------------- #
class InterpretSingle(BaseModel):
    caption: str
    rag_docs: list
    image_type: str

@app.post("/interpret_single")
def interpret_single(req: InterpretSingle):
    logger.info("=" * 80)
    logger.info("🧠 [INTERPRET_SINGLE] 개별 해석 시작")
    logger.info("🤖 사용 모델: Qwen (helena29/Qwen2.5_LoRA_for_HTP)")
    logger.info(f"이미지 타입: {req.image_type}")
    logger.info(f"입력 캡션: {req.caption}")
    logger.info(f"RAG 문서 수: {len(req.rag_docs)}")
    
    # RAG 문서가 있으면 참고문헌으로 활용, 없으면 캡션만으로 해석
    if req.rag_docs and len(req.rag_docs) > 0:
        literature_section = f"""
        HTP Research References (Korean):
        {req.rag_docs}
        
        Please refer to the above literature for your interpretation.
        """
        logger.info("✅ RAG 문서를 참고하여 해석")
    else:
        literature_section = """
        No specific references available. Please base your interpretation on general HTP psychology principles and the observed drawing features from the caption.
        """
        logger.info("⚠️  RAG 문서 없음 - 일반적인 HTP 원리로 해석")
    
    prompt = f"""
        You are an expert in HTP (House-Tree-Person) psychological test interpretation.

**Input Data:**
* **Drawing Type:** {req.image_type}
* **Drawing Caption:** {req.caption}
* **Reference Literature:** {literature_section}

**Task:**
Analyze the provided Drawing Caption and generate a psychological interpretation. Instead of writing a general essay, you must break down the caption into specific visual features and interpret each one individually based on HTP research patterns and psychological theory.

**Output Structure:**

**Part 1: Feature-by-Feature Analysis**
Extract key visual elements from the caption and provide a specific interpretation for each. Use the following format for every distinct feature found in the caption:

* **Visual Feature:** [Quote the specific part of the caption, e.g., "The tree is large"]
    * **Interpretation:** [Explain what this specific feature indicates psychologically. Refer to the provided literature if applicable. e.g., "This suggests a strong ego or high energy level..."]

**Part 2: Comprehensive Synthesis**
Provide a brief summary (1-2 paragraphs) integrating the features analyzed above. Discuss the individual's potential emotional state, social orientation, and coping strategies as a whole.

**Important Guidelines:**
* Ensure every visual detail mentioned in the caption (e.g., placement, size, specific objects like stars or flowers) is analyzed in Part 1.
* Use professional psychological terminology.
* Maintain a professional, analytical, and empathetic tone.
* **Write the response in English.**
    """
    
    logger.info(f"\n📝 프롬프트 길이: {len(prompt)} characters")

    result = generate_with_qwen(prompt)
    
    logger.info(f"✅ [INTERPRET_SINGLE] 해석 완료")
    logger.info(f"생성된 해석: {result}")
    logger.info("=" * 80)
    return {"interpretation": result}

# ----------------------------- #
# 4) GPT 번역 API
# ----------------------------- #
from openai import OpenAI
client = OpenAI()

class TranslateRequest(BaseModel):
    text: str

@app.post("/translate")
def translate(req: TranslateRequest):
    """영어 텍스트를 한국어로 번역"""
    logger.info("🌐 [TRANSLATE] 번역 시작")
    logger.info(f"원문 (영어): {req.text[:100]}...")
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a professional translator. Translate the given English text to natural Korean. Only provide the translation, nothing else."},
                {"role": "user", "content": req.text}
            ],
            temperature=0.3
        )
        
        translated = response.choices[0].message.content
        logger.info(f"번역 결과 (한국어): {translated[:100]}...")
        return {"translated": translated}
        
    except Exception as e:
        logger.error(f"❌ [TRANSLATE] 번역 실패: {str(e)}")
        return {"translated": req.text}  # 실패시 원문 반환

# ----------------------------- #
# 5) Qwen 모델로 추가 질문 생성 (영어)
# ----------------------------- #

class QuestionReq(BaseModel):
    conversation: list

@app.post("/questions")
def questions(req: QuestionReq):
    logger.info("=" * 80)
    logger.info("❓ [QUESTIONS] 추가 질문 생성 시작")
    logger.info("🤖 사용 모델: Qwen (helena29/Qwen2.5_LoRA_for_HTP)")
    logger.info(f"대화 기록 수: {len(req.conversation)}")
    
    for idx, msg in enumerate(req.conversation[-3:], 1):  # 최근 3개만 로깅
        logger.info(f"  메시지 {idx}: {msg.get('role')} - {msg.get('content')[:100]}...")
    
    # 대화 히스토리를 프롬프트로 변환
    conversation_text = "\n".join([
        f"{msg.get('role').upper()}: {msg.get('content')}" 
        for msg in req.conversation
    ])
    
    prompt = f"""
You are a professional psychologist conducting an HTP (House-Tree-Person) psychological assessment interview.

Conversation History:
{conversation_text}

Based on the conversation above, generate ONE follow-up question in English to gather more psychological insights.

Important Guidelines:
- Your response must be a single question in English only.
- The question should help understand the person's psychological state better.
- Keep the question clear, professional, and focused.
- Do not include any explanations, just the question itself.
"""
    
    result = generate_with_qwen(prompt)
    
    logger.info(f"✅ [QUESTIONS] 생성된 질문: {result}")
    logger.info("=" * 80)
    return {"question": result}

# ----------------------------- #
# 6) 최종 해석 (GPT-4o)
# ----------------------------- #
class InterpretFinal(BaseModel):
    single_results: dict
    conversation: list

@app.post("/interpret_final")
def interpret_final(req: InterpretFinal):
    logger.info("=" * 80)
    logger.info("🎯 [INTERPRET_FINAL] 최종 해석 생성 시작")
    logger.info("🤖 사용 모델: GPT-4o (OpenAI)")
    logger.info(f"집 해석: {req.single_results.get('house', '없음')[:100]}...")
    logger.info(f"나무 해석: {req.single_results.get('tree', '없음')[:100]}...")
    logger.info(f"사람 해석: {req.single_results.get('person', '없음')[:100]}...")
    logger.info(f"대화 기록 수: {len(req.conversation)}")
    
    # GPT 메시지 구성
    messages = [
        {
            "role": "system",
            "content": "You are a professional psychological counselor specializing in HTP (House-Tree-Person) test interpretation. Provide comprehensive, insightful psychological analysis in Korean."
        },
        {
            "role": "user",
            "content": f"""
당신은 전문 심리상담사입니다. 아래 HTP 검사 결과를 종합하여 최종 심리 해석을 작성해주세요.

집 해석 (House Interpretation):
{req.single_results.get('house','N/A')}

나무 해석 (Tree Interpretation):
{req.single_results.get('tree','N/A')}

사람 해석 (Person Interpretation):
{req.single_results.get('person','N/A')}

사용자와 나눈 대화:
{req.conversation}

위 정보를 종합하여 최종 HTP 심리 해석을 5개 문단으로 작성하세요.

중요 지침:
- 반드시 한국어로 작성하세요
- 각 그림(집, 나무, 사람)의 개별 해석을 통합하여 전체적인 심리 상태를 분석하세요
- 사용자와의 대화 내용을 참고하여 더 깊이 있는 해석을 제공하세요
- 전문적이고 따뜻한 어조로 작성하세요
- 5개 문단으로 구성하세요
"""
        }
    ]
    
    logger.info(f"📝 GPT 요청 전송 중...")
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.7,
        max_tokens=2000
    )
    
    result = response.choices[0].message.content
    
    logger.info(f"✅ [INTERPRET_FINAL] 최종 해석 완료")
    logger.info(f"생성된 최종 해석 (처음 200자): {result[:200]}...")
    logger.info("=" * 80)
    return {"final": result}
