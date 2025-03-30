# (Handles Data Loading)

import pandas as pd
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv

def load_books_data(file_path: str) -> pd.DataFrame:
    """Load book data from CSV file."""
    return pd.read_csv(file_path)

def load_text_data(file_path: str) -> list:
    """Load text data from a file."""
    raw_documents = TextLoader(file_path, encoding="utf-8").load()
    return raw_documents

# Load environment variables
load_dotenv()
