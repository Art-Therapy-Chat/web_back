import chromadb
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
chroma = chromadb.PersistentClient(path="./chroma")

collections = {
    "house": chroma.get_or_create_collection("house"),
    "tree": chroma.get_or_create_collection("tree"),
    "person": chroma.get_or_create_collection("person"),
}

# 데이터 로드 (처음 실행할 때만)
def load_docs():
    files = {
        "house": "./data/H.txt",
        "tree": "./data/T.txt",
        "person": "./data/P.txt"
    }
    for key, path in files.items():
        with open(path, "r", encoding="utf8") as f:
            text = f.read()
        emb = embedding_model.encode(text).tolist()
        collections[key].add(documents=[text], embeddings=[emb], ids=[key])

load_docs()

def rag_search(query, image_type):
    emb = embedding_model.encode(query).tolist()
    result = collections[image_type].query(query_embeddings=[emb], n_results=1)
    return result["documents"][0]
