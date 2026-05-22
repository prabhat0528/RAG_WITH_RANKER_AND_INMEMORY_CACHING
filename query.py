from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import RetrievalQA
from langchain_classic.retrievers.contextual_compression import (
    ContextualCompressionRetriever
)
from langchain_community.document_compressors import FlashrankRerank

# OLD CACHE IMPORTS
# from langchain_core.caches import InMemoryCache
# from langchain_core.outputs import Generation

# NEW SEMANTIC CACHE IMPORTS
from gptcache.adapter.api import init_similar_cache
from gptcache import Cache
from langchain_community.cache import GPTCache
from langchain_core.globals import set_llm_cache

import hashlib
import os


# ---------------- LOAD ENV ----------------

load_dotenv()

GEMINI_KEY = os.getenv("GEMINI_KEY")


# ---------------- SAME EMBEDDING MODEL ----------------

class CustomEmbeddings(Embeddings):

    def __init__(self):

        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

    def embed_documents(self, texts):

        return self.model.encode(texts).tolist()

    def embed_query(self, text):

        return self.model.encode(text).tolist()


embedding_function = CustomEmbeddings()


# ---------------- LOAD EXISTING CHROMA DB ----------------

db = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_function
)


# ---------------- CREATE RETRIEVER ----------------

retriever = db.as_retriever(
    search_kwargs={"k": 3}
)


# ---------------- PROMPT TEMPLATE ----------------

prompt = PromptTemplate.from_template(
    """
You are an AI assistant for Lovely Professional University.

Answer ONLY from the provided context.

If the answer is not present in the context,
reply with:
"Sorry! I don't know."

Context:
{context}

Question:
{question}

Answer:
"""
)


# ---------------- GEMINI MODEL ----------------

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GEMINI_KEY,
    temperature=0.3
)


# ---------------- RERANKER ----------------

compressor = FlashrankRerank()

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)


# ---------------- RAG CHAIN ----------------

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=compression_retriever,
    chain_type="stuff",
    chain_type_kwargs={
        "prompt": prompt
    }
)


# =========================================================
# ---------------- OLD  MATCH CACHE ------------------
# =========================================================

# cache = InMemoryCache()

# llm_string = "gemini-2.5-flash"


# =========================================================
# ---------------- NEW SEMANTIC CACHE ---------------------
# =========================================================

def get_hashed_name(name):

    return hashlib.sha256(
        name.encode()
    ).hexdigest()


def init_gptcache(cache_obj: Cache, llm: str):

    hashed_llm = get_hashed_name(llm)

    init_similar_cache(
        cache_obj=cache_obj,

        data_dir=f"./similar_cache/{hashed_llm}",

        # similarity_threshold=0.78
    )


# SET SEMANTIC CACHE
set_llm_cache(
    GPTCache(init_gptcache)
)


# ---------------- QUERY LOOP ----------------

print("\nLPU RAG Bot Ready with Semantic Cache!\n")

while True:

    query = input("Ask Question: ")

    if query.lower() == "exit":
        break


    # =====================================================
    # -------------- OLD CACHE LOGIC ----------------
    # =====================================================

    # result = cache.lookup(
    #     prompt=query,
    #     llm_string=llm_string
    # )

    # if result:

    #     print("\nAnswer (From Cache):\n")

    #     print(result[0].text)

    #     print("\n" + "-" * 50 + "\n")

    #     continue


    # =====================================================
    # ------------------ NEW RAG FLOW ---------------------
    # =====================================================

    # GPTCache automatically:
    # 1. Checks semantic similarity
    # 2. Returns cached response if found
    # 3. Otherwise calls Gemini
    # 4. Stores response automatically

    response = qa_chain.invoke(
        {"query": query}
    )

    answer = response["result"]


    # =====================================================
    # ------------ OLD MANUAL CACHE UPDATE ----------------
    # =====================================================

    # cache.update(
    #     prompt=query,
    #     llm_string=llm_string,
    #     return_val=[Generation(text=answer)]
    # )


    print("\nAnswer:\n")

    print(answer)

    print("\n" + "-" * 50 + "\n")