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
    logger.info("🔍 [RAG] RAG 검색 시작")
    logger.info(f"입력 캡션: {req.caption}")
    logger.info(f"이미지 타입: {req.image_type}")
    
    try:
        result = rag.query(req.caption, req.image_type)
        
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
            "result": "검색 실패",
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
    logger.info(f"이미지 타입: {req.image_type}")
    logger.info(f"입력 캡션: {req.caption}")
    logger.info(f"RAG 문서 수: {len(req.rag_docs)}")
    
    # RAG 문서가 있으면 참고문헌으로 활용, 없으면 캡션만으로 해석
    if req.rag_docs and len(req.rag_docs) > 0:
        literature_section = f"""
        HTP 연구 참고 문헌 (한국어):
        {req.rag_docs}
        
        위 문헌을 참고하여 해석하세요.
        """
        logger.info("✅ RAG 문서를 참고하여 해석")
    else:
        literature_section = """
        특정 참고 문헌이 없습니다. 일반적인 HTP 심리학 원리와 캡션에서 관찰된 그림 특징을 기반으로 해석하세요.
        """
        logger.info("⚠️  RAG 문서 없음 - 일반적인 HTP 원리로 해석")
    
    prompt = f"""
        당신은 HTP(집-나무-사람) 심리 검사 해석 전문가입니다.
        
        그림 유형: {req.image_type}
        
        그림 캡션 (영어): {req.caption}
        
        {literature_section}
        
        그림의 특징을 바탕으로 HTP 심리 해석을 정확히 3~5문장으로 작성하세요.
        
        중요 지침:
        - 전체 응답은 반드시 한국어로만 작성하세요.
        - 영어 단어, 번역, 설명을 포함하지 마세요.
        - 영어가 단 한 단어라도 포함되면 무효입니다.
        - 그림 특징과 관련된 심리학적 통찰에 집중하세요.
        - 참고 문헌의 내용을 적절히 활용하여 전문적인 해석을 제공하세요.
    """
    
    logger.info(f"\n📝 프롬프트 길이: {len(prompt)} characters")

    result = generate_with_qwen(prompt)
    
    logger.info(f"✅ [INTERPRET_SINGLE] 해석 완료")
    logger.info(f"생성된 해석: {result}")
    logger.info("=" * 80)
    return {"interpretation": result}

# ----------------------------- #
# 4) GPT-4o-mini로 추가 질문 생성
# ----------------------------- #
from openai import OpenAI
client = OpenAI()

class QuestionReq(BaseModel):
    conversation: list

@app.post("/questions")
def questions(req: QuestionReq):
    logger.info("=" * 80)
    logger.info("❓ [QUESTIONS] 추가 질문 생성 시작")
    logger.info(f"대화 기록 수: {len(req.conversation)}")
    
    for idx, msg in enumerate(req.conversation[-3:], 1):  # 최근 3개만 로깅
        logger.info(f"  메시지 {idx}: {msg.get('role')} - {msg.get('content')[:100]}...")
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=req.conversation
    )
    
    question = response.choices[0].message.content
    logger.info(f"✅ [QUESTIONS] 생성된 질문: {question}")
    logger.info("=" * 80)
    return {"question": question}

# ----------------------------- #
# 5) 최종 해석 (Qwen + LoRA)
# ----------------------------- #
class InterpretFinal(BaseModel):
    single_results: dict
    conversation: list

@app.post("/interpret_final")
def interpret_final(req: InterpretFinal):
    logger.info("=" * 80)
    logger.info("🎯 [INTERPRET_FINAL] 최종 해석 생성 시작")
    logger.info(f"집 해석: {req.single_results.get('house', '없음')[:100]}...")
    logger.info(f"나무 해석: {req.single_results.get('tree', '없음')[:100]}...")
    logger.info(f"사람 해석: {req.single_results.get('person', '없음')[:100]}...")
    logger.info(f"대화 기록 수: {len(req.conversation)}")
    
    prompt = f"""
당신은 전문 심리상담사입니다.

집 해석:
{req.single_results.get('house','')}

나무 해석:
{req.single_results.get('tree','')}

사람 해석:
{req.single_results.get('person','')}

사용자와 나눈 대화:
{req.conversation}

위 정보를 종합한 최종 HTP 해석을 5문단으로 작성하세요. 반드시 한글로 작성하세요. rag에 포함된 설명 또한 영어가 있을경우 한글로 번역 후 작성하세요.
    """
    
    logger.info(f"📝 최종 프롬프트 길이: {len(prompt)} characters")
    
    result = generate_with_qwen(prompt)
    
    logger.info(f"✅ [INTERPRET_FINAL] 최종 해석 완료")
    logger.info(f"생성된 최종 해석 (처음 200자): {result[:200]}...")
    logger.info("=" * 80)
    return {"final": result}
