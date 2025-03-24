# -*- coding: utf-8 -*-
"""
Created on Sat Mar 15 13:02:57 2025

@author: oisin
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS

# Read in the cleaned dataset
input_path = r"recommendation-system/data/processed/cleaned_dataset.xlsx"
df = pd.read_excel(input_path)

# Checking dataset info
print("Dataset Info:")
print(df.info())

print("\nFirst 5 rows of the dataset:")
print(df.head())

# Checking for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Drop rows with missing values in critical columns
df.dropna(subset=['ratings', 'text', 'target'], inplace=True)

# Verifying
df.reset_index(drop=True, inplace=True)
print("\nMissing values after cleaning:")
print(df.isnull().sum())
print("\nDataset Shape:", df.shape)

# Ensure correct data types
df["ratings"] = pd.to_numeric(df["ratings"], errors='coerce')
df["target"] = df["target"].astype(str)  # Ensure 'target' is string

# Summary statistics
print("\nSummary Statistics:")
print(df.describe(include="all"))

# Checking unique values
print("\nProduct Name Distribution:\n", df["product name"].value_counts().head(10))
print("\nCategory Distribution:\n", df["category"].value_counts())
print("\nSentiment Target Distribution:\n", df["target"].value_counts())

## Rating Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["ratings"], bins=5, kde=True)
plt.xlabel("Ratings")
plt.ylabel("Count")
plt.title("Distribution of Ratings")
plt.show()

## Sentiments vs Rating Analysis
plt.figure(figsize=(6, 4))
sns.boxplot(x=df["target"], y=df["ratings"])
plt.xlabel("Sentiment (Negative=0, Positive=1)")
plt.ylabel("Ratings")
plt.title("Ratings Distribution by Sentiment Target")
plt.show()

## Category-Wise Ratings Distribution (Top 10 Categories)
top_categories = df["category"].value_counts().nlargest(10).index
df_top_categories = df[df["category"].isin(top_categories)]

plt.figure(figsize=(10, 5))
sns.boxplot(x="category", y="ratings", data=df_top_categories, order=top_categories)
plt.xlabel("Top Product Categories")
plt.ylabel("Ratings")
plt.title("Ratings Distribution by Category")
plt.xticks(rotation=45)
plt.show()

## WordCloud for Positive & Negative Reviews
# Drop NaN values before joining
positive_reviews = " ".join(df[df['target'] == 'p']['text'].dropna().astype(str))
negative_reviews = " ".join(df[df['target'] == 'n']['text'].dropna().astype(str))

stopwords = set(STOPWORDS)

positive_wordcloud = WordCloud(width=600, height=400, background_color='white', stopwords=stopwords).generate(positive_reviews)
negative_wordcloud = WordCloud(width=600, height=400, background_color='black', stopwords=stopwords).generate(negative_reviews)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].imshow(positive_wordcloud, interpolation='bilinear')
axes[0].set_title("Positive Reviews Word Cloud")
axes[0].axis("off")

axes[1].imshow(negative_wordcloud, interpolation='bilinear')
axes[1].set_title("Negative Reviews Word Cloud")
axes[1].axis("off")

plt.tight_layout()
plt.show()