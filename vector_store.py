# (Handles Vector Embeddings & Search)
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

def create_vector_store(raw_documents) -> Chroma:
    """Create a Chroma vector store from documents."""
    text_splitter = CharacterTextSplitter(chunk_size=0, chunk_overlap=0, separator="\n")
    documents = text_splitter.split_documents(raw_documents)

    return Chroma.from_documents(
        documents,
        embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    )
