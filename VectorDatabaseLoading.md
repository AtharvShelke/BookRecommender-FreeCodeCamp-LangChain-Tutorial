# **Semantic Book Recommendation System - Detailed Explanation**

## **Project Overview**
This project implements a **semantic book recommendation system** using **LangChain, ChromaDB, and Google Generative AI Embeddings**. The system:
- Loads a dataset of books.
- Processes text descriptions into vector embeddings.
- Performs similarity search to recommend books based on user queries.

## **Project Structure**
```
/project
│── main.py                 # Entry point of the application
│── data_loader.py           # Handles data loading (CSV and text files)
│── vector_store.py          # Manages vector embeddings & similarity search
│── recommendation.py        # Handles recommendation logic
│── requirements.txt         # Lists required dependencies
│── book_cleaned.csv         # Dataset of books
│── tagged_description.txt   # Processed descriptions
```

---

## **1️⃣ `data_loader.py` - Data Loading**
This module loads book data from a CSV file and text data for embedding generation.

### **Functions:**
```python
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
```
### **How It Works:**
- `load_books_data(file_path)`: Loads the **book_cleaned.csv** file into a Pandas DataFrame.
- `load_text_data(file_path)`: Loads **tagged_description.txt** for text processing.

---

## **2️⃣ `vector_store.py` - Vector Embedding & Storage**
This module converts book descriptions into embeddings and stores them in **ChromaDB**.

### **Functions:**
```python
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
```
### **How It Works:**
1. **Splits text data** into chunks using `CharacterTextSplitter`.
2. **Embeds text** using `GoogleGenerativeAIEmbeddings`.
3. **Stores embeddings** in `Chroma` for efficient retrieval.

---

## **3️⃣ `recommendation.py` - Semantic Search & Recommendation**
This module finds books that are most similar to a given user query.

### **Functions:**
```python
import pandas as pd

def retrieve_semantic_recommendations(db_books, books, query: str, top_k: int) -> pd.DataFrame:
    """Retrieve top K semantic recommendations based on a query."""
    recs = db_books.similarity_search(query, k=50)
    books_list = []

    for rec in recs:
        try:
            isbn = rec.page_content.split()[0].strip().strip('"').strip("'")
            books_list.append(int(isbn))
        except ValueError as e:
            print(f"Skipping invalid ISBN: {rec.page_content.split()[0]} - Error: {e}")

    return books[books["isbn13"].isin(books_list)].head(top_k)
```
### **How It Works:**
1. **Performs a similarity search** in `db_books`.
2. **Extracts ISBNs** from retrieved descriptions.
3. **Filters the books dataset** to return relevant results.
4. **Handles invalid ISBN errors gracefully**.

---

## **4️⃣ `main.py` - Main Script**
This script ties everything together and executes the recommendation process.

### **Code:**
```python
import pandas as pd
from data_loader import load_books_data, load_text_data
from vector_store import create_vector_store
from recommendation import retrieve_semantic_recommendations

# Load book data
books = load_books_data("book_cleaned.csv")

# Load text data
raw_documents = load_text_data("tagged_description.txt")

# Create vector store
db_books = create_vector_store(raw_documents)

# Query example
query = "A book to teach children about nature"
docs = db_books.similarity_search(query, k=10)

try:
    isbn = docs[0].page_content.split()[0].strip().strip('"').strip("'")
    result = books[books["isbn13"] == int(isbn)]
    print(result)
except ValueError as e:
    print("Error converting ISBN:", e)

# Get recommendations
recommendations = retrieve_semantic_recommendations(db_books, books, query, 10)
print(recommendations)
```
### **How It Works:**
1. **Loads book and text data**.
2. **Creates a vector database**.
3. **Runs a query for book recommendations**.
4. **Handles ISBN conversion errors**.
5. **Prints the final recommendations**.

---

## **5️⃣ `requirements.txt` - Dependencies**
```
pandas
langchain_community
langchain_text_splitters
langchain_google_genai
langchain_chroma
python-dotenv
```
### **How to Install Dependencies:**
Run the following command:
```bash
pip install -r requirements.txt
```

---

## **🚀 Running the Project in Google Colab**
### **Steps to Run in Colab:**
1. **Upload necessary files** (`book_cleaned.csv`, `tagged_description.txt`).
2. **Install dependencies**:
   ```bash
   !pip install -r requirements.txt
   ```
3. **Run the script**:
   ```bash
   !python main.py
   ```

---

## **💡 Key Features & Improvements**
✅ **Modular Design** → Easy to maintain and expand.
✅ **Efficient Search** → Uses **ChromaDB** for fast retrieval.
✅ **Google AI Embeddings** → Improves semantic understanding.
✅ **Error Handling** → Prevents crashes due to incorrect data.
✅ **Colab Compatible** → Easily runs in Jupyter/Colab environments.

---

## **🔗 Future Enhancements**
🔹 **Improve query processing** (e.g., spell check, synonym expansion).  
🔹 **Enhance search efficiency** using optimized embedding models.  
🔹 **UI Integration** for a web-based book recommendation system.  

---

This document provides a complete guide to understanding and running the **Semantic Book Recommendation System**. 🚀 Happy coding! 🎯

