from dotenv import load_dotenv
load_dotenv()


from fastapi import FastAPI
from pydantic import BaseModel
from caption import generate_caption
from rag import rag_search
from model import generate_with_qwen
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # 프론트엔드 주소
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------- #
# 1) 이미지 캡션 생성
# ----------------------------- #
class CaptionRequest(BaseModel):
    image_base64: str

@app.post("/caption")
def caption(req: CaptionRequest):
    caption = generate_caption(req.image_base64)
    return {"caption": caption}

# ----------------------------- #
# 2) RAG 검색
# ----------------------------- #
class RagRequest(BaseModel):
    caption: str
    image_type: str  # "house" | "tree" | "person"

@app.post("/rag")
def rag(req: RagRequest):
    result = rag_search(req.caption, req.image_type)
    return {"documents": result}

# ----------------------------- #
# 3) Qwen 로라 모델 개별 해석
# ----------------------------- #
class InterpretSingle(BaseModel):
    caption: str
    rag_docs: list
    image_type: str

@app.post("/interpret_single")
def interpret_single(req: InterpretSingle):
    prompt = f"""
        You are an HTP (House-Tree-Person) psychological interpretation expert.
        
        Drawing type:
        {req.image_type}
        
        Image caption:
        {req.caption}
        
        Relevant literature:
        {req.rag_docs}
        
        Write an HTP interpretation in exactly 3–5 sentences.
        
        IMPORTANT INSTRUCTIONS:
        - Your ENTIRE response MUST be in Korean only.
        - Do NOT output English words, translations, or explanations.
        - If you output any English at all, even a single word, the answer is invalid.
        - 반드시 한국어로 작성하세요.
    """

    result = generate_with_qwen(prompt)
    print(result)
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
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=req.conversation
    )

    return {"question": response.choices[0].message.content}

# ----------------------------- #
# 5) 최종 해석 (Qwen + LoRA)
# ----------------------------- #
class InterpretFinal(BaseModel):
    single_results: dict
    conversation: list

@app.post("/interpret_final")
def interpret_final(req: InterpretFinal):
    prompt = f"""
당신은 전문 심리상담사입니다.

집 해석:
{req.single_results.get('house','')}

나무 해석:
{req.single_results.get('tree','')}

사람 해석:
{req.single_results.get('person','')}

추가 질문과 답변:
{req.conversation}

위 정보를 종합한 최종 HTP 해석을 5문단으로 작성하세요. 반드시 한글로 작성하세요. rag에 포함된 설명 또한 영어가 있을경우 한글로 번역 후 작성하세요.
    """
    result = generate_with_qwen(prompt)
    return {"final": result}
