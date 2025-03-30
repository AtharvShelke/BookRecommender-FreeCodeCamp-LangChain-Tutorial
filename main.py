import pandas as pd
from data_loader import load_books_data, load_text_data
from vector_store import create_vector_store
from recommendations import retrieve_semantic_recommendations

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
