# (Handles Semantic Search & Recommendation)



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
