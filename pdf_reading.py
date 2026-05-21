from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
import os

# Load environment variables
load_dotenv()

pdf = "LPU.pdf"

GEMINI_KEY = os.getenv("GEMINI_KEY")


# ---------------- PDF TEXT EXTRACTION ----------------

def get_pdf_text(pdf_path):

    text = ""

    reader = PdfReader(pdf_path)

    for page in reader.pages:

        extracted_text = page.extract_text()

        if extracted_text:
            text += extracted_text

    return text


pdf_text = get_pdf_text(pdf)


# ---------------- TEXT CHUNKING ----------------

def get_chunks(pdf_text):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = text_splitter.split_text(pdf_text)

    return chunks


chunks = get_chunks(pdf_text)

print(f"Total Chunks: {len(chunks)}")


# ---------------- EMBEDDING CREATION ----------------

def create_embeddings(chunks):

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    vectors = model.encode(chunks)

    return vectors


embeddings = create_embeddings(chunks)

# print(embeddings.shape)

# # First embedding vector
# print(embeddings[0])


# ---------------- CHROMA DB STORAGE ----------------

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


vector_store = Chroma.from_texts(
    texts=chunks,
    embedding=embedding_function,
    persist_directory="./chroma_db"
)


vector_store.persist()

print("Vectors stored successfully in ChromaDB")