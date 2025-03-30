# Import necessary libraries
import kagglehub
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Download the latest dataset version
path = kagglehub.dataset_download("dylanjcastillo/7k-books-with-metadata")
print("Path to dataset files:", path)

# Load the dataset
books = pd.read_csv(f"{path}/books.csv")

# Visualizing missing values
ax = plt.axes()
sns.heatmap(books.isna().transpose(), cbar=False, ax=ax)
plt.xlabel("Columns")
plt.ylabel("Missing values")
plt.show()

# Feature Engineering: Adding new columns
books["missing_description"] = np.where(books["description"].isna(), 1, 0)
books["age_of_book"] = 2025 - books["published_year"]

# Selecting relevant columns for correlation analysis
columns_of_interest = ["num_pages", "age_of_book", "missing_description", "average_rating"]
correlation_matrix = books[columns_of_interest].corr(method="spearman")

# Setting seaborn theme
sns.set_theme(style="white")

# Plot correlation heatmap
plt.figure(figsize=(8, 6))
heatmap = sns.heatmap(
    correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, 
    cbar_kws={"label": "Spearman correlation"}
)
plt.title("Correlation Matrix")
plt.show()

# Filtering books with no missing critical data
book_missing = books[
    ~(books["description"].isna()) &
    ~(books["num_pages"].isna()) &
    ~(books["average_rating"].isna()) &
    ~(books["published_year"].isna())
]

# Creating a copy to avoid SettingWithCopyWarning
book_missing = book_missing.copy()

# Count category occurrences
book_missing["categories"].value_counts().reset_index().sort_values("count", ascending=False)

# Counting words in description
book_missing["words_in_description"] = book_missing["description"].str.split().str.len()

# Filtering books with descriptions between 1 and 4 words (though not used later)
short_descriptions = book_missing.loc[book_missing["words_in_description"].between(1, 4), "description"]

# Selecting books with descriptions of at least 25 words
book_missing_25_words = book_missing[book_missing["words_in_description"] >= 25].copy()

# Creating "title_and_subtitle" column
book_missing_25_words.loc[:, "title_and_subtitle"] = np.where(
    book_missing_25_words["subtitle"].isna(), 
    book_missing_25_words["title"], 
    book_missing_25_words[["title", "subtitle"]].astype(str).agg(": ".join, axis=1)
)

# Creating "tagged_description" column
book_missing_25_words.loc[:, "tagged_description"] = book_missing_25_words[["isbn13", "description"]].astype(str).agg(" ".join, axis=1)

# Dropping unnecessary columns and saving the cleaned dataset
book_missing_25_words.drop(
    ["subtitle", "missing_description", "age_of_book", "words_in_description"], axis=1
).to_csv("book_cleaned.csv", index=False)

print("Data cleaning complete. Processed file saved as 'book_cleaned.csv'.")
