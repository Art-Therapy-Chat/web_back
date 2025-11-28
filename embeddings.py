# embeddings.py
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import CrossEncoder
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings


# -------------------------------
# Embedding Wrapper
# -------------------------------
class MyEmbeddings(Embeddings):
    def __init__(self, model, tokenizer, device="cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def embed_documents(self, texts):
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            emb = self.model(**inputs).last_hidden_state[:, 0, :]
            emb = emb / emb.norm(dim=1, keepdim=True)
        return emb.cpu().numpy()[0]


# -------------------------------
# Load Embedding Model
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
embedding_model_name = "HJUNN/bge-m3b-Art-Therapy-embedding-fine-tuning"

embed_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
embed_model = AutoModel.from_pretrained(embedding_model_name).to(device)

embeddings = MyEmbeddings(embed_model, embed_tokenizer, device=device)


# -------------------------------
# Load Vectorstore
# -------------------------------
vectorstore = Chroma(
    collection_name="htp_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_store"
)


# -------------------------------
# CrossEncoder Reranker
# -------------------------------
cross_encoder = CrossEncoder(
    "HJUNN/bge_BCE_cross_encoder",
    device=device
)
