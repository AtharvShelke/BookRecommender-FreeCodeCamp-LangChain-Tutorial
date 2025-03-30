# Data Preprocessing for Books Metadata Dataset

## 📌 Overview
This project processes and cleans a dataset of books with metadata. The dataset is sourced from Kaggle and includes various attributes such as title, description, number of pages, published year, and categories.

The preprocessing steps involve:
- Handling missing values
- Feature engineering
- Filtering meaningful descriptions
- Data export for further analysis

---

## 📂 Dataset Source
The dataset is downloaded from Kaggle using `kagglehub`:
```python
import kagglehub
path = kagglehub.dataset_download("dylanjcastillo/7k-books-with-metadata")
print("Path to dataset files:", path)
```

The main file used is `books.csv`, which is loaded as a Pandas DataFrame:
```python
import pandas as pd
books = pd.read_csv(f"{path}/books.csv")
```

---

## 🛠 Data Cleaning & Preprocessing

### 1️⃣ **Handling Missing Data**
To understand missing data, we visualize it using a heatmap:
```python
import seaborn as sns
import matplotlib.pyplot as plt

ax = plt.axes()
sns.heatmap(books.isna().transpose(), cbar=False, ax=ax)
plt.xlabel("Columns")
plt.ylabel("Missing values")
plt.show()
```

We filter out books with missing values in key columns:
```python
book_missing = books[
    ~(books["description"].isna()) &
    ~(books["num_pages"].isna()) &
    ~(books["average_rating"].isna()) &
    ~(books["published_year"].isna())
]
book_missing = book_missing.copy()
```

### 2️⃣ **Feature Engineering**
#### **Adding Derived Features**
- `missing_description`: Indicates whether a book lacks a description (1 = missing, 0 = present)
- `age_of_book`: Computes book age using `2025 - published_year`
```python
import numpy as np
books["missing_description"] = np.where(books["description"].isna(), 1, 0)
books["age_of_book"] = 2025 - books["published_year"]
```

#### **Correlation Analysis**
We compute the Spearman correlation between key numeric features:
```python
columns_of_interest = ["num_pages", "age_of_book", "missing_description", "average_rating"]
correlation_matrix = books[columns_of_interest].corr(method="spearman")

sns.set_theme(style="white")
plt.figure(figsize=(8, 6))
heatmap = sns.heatmap(
    correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5,
    cbar_kws={"label": "Spearman correlation"}
)
plt.title("Correlation Matrix")
plt.show()
```

### 3️⃣ **Filtering Books with Meaningful Descriptions**
We count words in descriptions:
```python
book_missing["words_in_description"] = book_missing["description"].str.split().str.len()
```
Filter books with **at least 25 words** in their description:
```python
book_missing_25_words = book_missing[book_missing["words_in_description"] >= 25].copy()
```

### 4️⃣ **Combining Title and Subtitle**
We create a `title_and_subtitle` column:
```python
book_missing_25_words.loc[:, "title_and_subtitle"] = np.where(
    book_missing_25_words["subtitle"].isna(),
    book_missing_25_words["title"],
    book_missing_25_words[["title", "subtitle"]].astype(str).agg(": ".join, axis=1)
)
```

### 5️⃣ **Creating a Tagged Description for NLP**
Merging `isbn13` and `description` into a single column:
```python
book_missing_25_words.loc[:, "tagged_description"] = book_missing_25_words[["isbn13", "description"]].astype(str).agg(" ".join, axis=1)
```

### 6️⃣ **Saving the Cleaned Dataset**
We remove unnecessary columns and save the final dataset as a CSV file:
```python
book_missing_25_words.drop(
    ["subtitle", "missing_description", "age_of_book", "words_in_description"], axis=1
).to_csv("book_cleaned.csv", index=False)
```

---

## 📊 Summary
✔ **Filtered out books with missing values in critical fields**  
✔ **Created new features (`age_of_book`, `words_in_description`)**  
✔ **Selected books with meaningful descriptions (≥25 words)**  
✔ **Merged title and subtitle for better readability**  
✔ **Created a `tagged_description` column for NLP applications**  
✔ **Exported cleaned dataset for further analysis**  

This dataset is now optimized for **machine learning, text analysis, and further exploration!** 🚀

---

## 📌 Next Steps
- Perform **natural language processing (NLP)** on book descriptions.
- Train a **recommender system** using book metadata.
- Visualize **trends in book publishing over time**.


