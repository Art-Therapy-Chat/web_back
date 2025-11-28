# rag_engine.py
import torch
import json
from typing import List, Dict
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.documents import Document
from embeddings import vectorstore, cross_encoder


# ===============================================================
# 1) Multi Query Generator (OpenAI GPT-4o-mini)
# ===============================================================

class MultiQueryGenerator(BaseModel):
    queries: List[str] = Field(description="검색 쿼리 목록")


class AdvancedQueryRewriter:
    def __init__(self, model_name="gpt-4o", temperature=0):
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=1000
        )
        self.parser = JsonOutputParser(pydantic_object=MultiQueryGenerator)

        self.template = """
        당신은 사용자의 질문을 기반으로 검색용 쿼리를 재작성합니다.

        # 지침
        1. 아래의 history 에 포함된 모든 이전 질의/검색문서/답변을 참고하여 더 정확한 검색 쿼리를 생성하세요.
        2. 현재 질문이 모호하거나 생략된 경우, history 를 참고해 문맥상 완전한 쿼리로 재구성하세요.
        3. 이전 대화가 없거나 관련이 없는 경우 현재 질문만 사용하세요.
        4. 반드시 명확하고 검색에 적합한 쿼를 생성하세요.
        6. 출력은 재생성된 쿼리 문자열만 포함해야 합니다. 추가 설명이나 주석은 포함하지 마세요.
        7. 현재 문장이 여러 속성을 포함하고 있다면, 각각을 별도의 쿼리로 분리하세요.
        8. 각 쿼리는 독립적으로 벡터 DB에서 검색될 수 있도록 완전하고 명확해야 합니다.
        10. 분리된 쿼리들을 합쳤을 때 원래 쿼리의 의미를 나타낼 수 있도록 하세요.

        전체 대화: {history_text}
        사용자 질문: {current_query}

        출력 예시:
        {{
            "queries": ["쿼리1", "쿼리2"]
        }}

        예시 :
        현재 질문이 "서울은 ? 그리고 맛집은?"이고 이전 대화가 "한국의 관광지 추천해줘"라면,
        {{{{
            "queries" : ["서울의 관광지 추천해줘", "서울의 맛집 추천해줘"]
        }}}}

        단일 쿼리인 경우:
        {{{{
            "queries" : ["한국의 관광지 추천해줘"]
        }}}}

        {format_instructions}
        """

        self.prompt = PromptTemplate(
            input_variables=["history_text", "current_query"],
            template=self.template,
        )

    def rewrite_query(self, history_text: str, current_query: str) -> List[str]:
        if not history_text.strip():
            history_text = "대화 기록 없음"

        format_instructions = self.parser.get_format_instructions()

        prompt = self.prompt.format(
            history_text=history_text,
            current_query=current_query,
            format_instructions=format_instructions
        )

        llm_response = self.llm.invoke(prompt).content

        try:
            data = json.loads(llm_response)
            return data.get("queries", [current_query])
        except:
            print("⚠ JSON 파싱 실패, 원본 쿼리 사용")
            return [current_query]


# ===============================================================
# 2) Multi Query Retriever
# ===============================================================

class MultiQueryRetriever:
    def __init__(self, vectorstore, query_rewriter):
        self.vectorstore = vectorstore
        self.cross_encoder = cross_encoder
        self.history = []
        self.query_rewriter = query_rewriter

    def build_history_text(self):
        out = ""
        for h in self.history:
            out += f"[USER]\n{h['user_query']}\n"
            out += f"[REWRITTEN]\n{h['rewritten_queries']}\n"
            out += "[DOCS]\n"
            for d in h["retrieved_docs"]:
                out += f"- {d['content']}\n"
            out += f"[ANSWER]\n{h['final_answer']}\n"
            out += "-" * 30 + "\n"
        return out

    def retrieve(self, query: str, category: str, num_docs=5):

        history_text = self.build_history_text()
        rewritten_queries = self.query_rewriter.rewrite_query(history_text, query)

        print("재작성된 쿼리:", rewritten_queries)

        all_docs = []
        seen = set()

        for q in rewritten_queries:
            # 1) 쿼리당 5개 검색
            docs = self.vectorstore.similarity_search(
                q,
                k=num_docs,
                filter={"category": category}
            )

            if not docs:
                continue

            # 2) 쿼리별 CrossEncoder Top-2 추출
            pairs = [(q, d.page_content) for d in docs]
            scores = self.cross_encoder.predict(pairs)

            # numpy/torch 변환 처리
            if hasattr(scores, "detach"):
                scores = scores.detach().cpu().numpy()
            scores = scores.squeeze().tolist()

            reranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    
            # ★ 여기서 쿼리당 Top-2만 유지
            top_docs = [doc for doc, score in reranked[:2]]

            # 3) 최종 후보 리스트에 추가 (중복 제거)
            for d in top_docs:
                if d.page_content not in seen:
                    all_docs.append(d)
                    seen.add(d.page_content)

        return all_docs, rewritten_queries


# ===============================================================
# 3) RAG 검색 엔진 (검색 전용)
# ===============================================================

# RAG 검색 클래스 (해석 생성 없이 검색만 수행)
class AdvancedConversationalRAG:
    def __init__(self, vectorstore, query_model_name="gpt-4o"):
        """
        Hugging Face에 업로드된 fine-tuned 모델을 사용한 대화형 RAG 시스템
        Args:
            vectorstore: 벡터 저장소
            query_model_name: 쿼리 재작성용 OpenAI 모델 이름 (기본값: gpt-4o)
        """
        # history에 대화 저장
        self.history = []
        
        # 쿼리 재생성기 (OpenAI GPT 사용)
        self.query_rewriter = AdvancedQueryRewriter(model_name=query_model_name)
        
        # 각각의 검색어를 따로 검색한 뒤에 검색결과를 취합하는 멀티쿼리 리트리버
        self.retriever = MultiQueryRetriever(vectorstore=vectorstore, query_rewriter=self.query_rewriter)
        
        print("✅ RAG 엔진 초기화 완료 (검색 전용 모드)")
    
    def search_only(self, current_query: str, category: str) -> Dict:
        """
        검색만 수행하고 해석은 생성하지 않음 (프론트엔드에서 /interpret_single 사용)
        """
        print("🔍 [RAG] 검색 전용 모드 - 해석 생성 스킵")
        
        # 관련 문서검색만 수행
        docs, rewritten_queries = self.retriever.retrieve(current_query, category)
        
        # 문서 내용을 문자열 리스트로 변환
        rag_docs = [doc.page_content for doc in docs]
        
        print(f"✅ [RAG] 검색 완료 - {len(rag_docs)}개 문서 발견")
        
        # 결과 반환 (해석 없이 검색 결과만)
        return {
            "query": current_query,
            "rewritten_queries": rewritten_queries,
            "source_documents": docs,
            "rag_docs": rag_docs
        }
