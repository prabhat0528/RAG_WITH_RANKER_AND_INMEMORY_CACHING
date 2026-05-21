from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import RetrievalQA
import os
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank
from langchain_core.caches import InMemoryCache
from langchain_core.outputs import Generation

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

compressor = FlashrankRerank()
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
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

#---------------------Caching----------------------
cache = InMemoryCache()
llm_string = "gemini-2.5-flash"


# ---------------- QUERY LOOP ----------------

print("\nLPU RAG Bot Ready!\n")

while True:

    query = input("Ask Question: ")

    if query.lower() == "exit":
        break

    # Check cache
    result = cache.lookup(
        prompt=query,
        llm_string=llm_string
    )

    if result:

        print("\nAnswer (From Cache):\n")

        print(result[0].text)

        print("\n" + "-" * 50 + "\n")

        continue

    # Generate fresh response
    response = qa_chain.invoke(
        {"query": query}
    )

    answer = response["result"]

    # Store in cache
    cache.update(
        prompt=query,
        llm_string=llm_string,
        return_val=[Generation(text=answer)]
    )

    print("\nAnswer:\n")

    print(answer)

    print("\n" + "-" * 50 + "\n")