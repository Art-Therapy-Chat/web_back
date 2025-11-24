# rag_engine.py
import torch
import json
from typing import List, Dict
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import JsonOutputParser
from langchain_core.documents import Document


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
        {
            "queries": ["쿼리1", "쿼리2"]
        }

        예시 :
        현재 질문이 "서울은 ? 그리고 맛집은?"이고 이전 대화가 "한국의 관광지 추천해줘"라면,
        {{
            "queries" : ["서울의 관광지 추천해줘", "서울의 맛집 추천해줘"]
        }}

        단일 쿼리인 경우:
        {{
            "queries" : ["한국의 관광지 추천해줘"]
        }}

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

    def retrieve(self, query: str, category: str, num_docs=3):

        history_text = self.build_history_text()
        rewritten_queries = self.query_rewriter.rewrite_query(history_text, query)

        print("재작성된 쿼리:", rewritten_queries)

        all_docs = []
        seen = set()

        for q in rewritten_queries:
            docs = self.vectorstore.similarity_search(
                q,
                k=num_docs,
                filter={"category": category}
            )

            for d in docs:
                if d.page_content not in seen:
                    seen.add(d.page_content)
                    all_docs.append(d)

        return all_docs, rewritten_queries


# ===============================================================
# 3) Fine-tuned Qwen2.5 HTP 모델 기반 RAG
# ===============================================================

from transformers import AutoModelForCausalLM, AutoTokenizer

class AdvancedConversationalRAG:
    def __init__(self, vectorstore, model_name="helena29/Qwen2.5_LoRA_for_HTP"):
        self.history = []
        self.query_rewriter = AdvancedQueryRewriter()
        self.retriever = MultiQueryRetriever(vectorstore, self.query_rewriter)

        print("🔥 Loading Qwen HTP Model:", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype="auto"
        )
        self.device = self.llm.device

        self.response_template = """
        You are a professional psychologist specializing in HTP interpretation.

        User Question:
        {query}

        Relevant Information:
        {context}

        Guidelines:
        1. If the user's question contains multiple queries, address each one clearly and separately.
        2. Base your answer only on the provided information. If information is insufficient, honestly state that you don't know.
        3. Provide your answer in Korean language.
        4. If there are original sources in the provided information, cite them appropriately.
        5. Explain possible psychological meanings in a professional manner.

        Answer:
        """

    def generate_response(self, prompt: str) -> str:

        messages = [
            {"role": "system", "content": "HTP 검사 전문 심리학자 역할"},
            {"role": "user", "content": prompt}
        ]

        formatted = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(formatted, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.llm.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.3,
                do_sample=True
            )

        return self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()

    def query(self, current_query: str, category: str):
        docs, rewritten_queries = self.retriever.retrieve(
            current_query,
            category=category
        )

        if docs:
            context = "\n\n".join(
                [f"문서 {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)]
            )
        else:
            context = "검색된 문서 없음"

        prompt = self.response_template.format(
            query=current_query,
            context=context
        )

        response = self.generate_response(prompt)

        record = {
            "user_query": current_query,
            "rewritten_queries": rewritten_queries,
            "retrieved_docs": [{"content": d.page_content} for d in docs],
            "final_answer": response,
        }

        self.history.append(record)
        self.retriever.history.append(record)

        return {
            "result": response,
            "rewritten_queries": rewritten_queries,
            "source_documents": docs
        }
