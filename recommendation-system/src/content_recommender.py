# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 22:47:29 2025

@author: mohsin

Content-Based Filtering Recommender System
"""

import os
import pandas as pd
import numpy as np
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Loading the cleaned dataset
file_path = os.path.abspath(os.path.join("data", "processed", "cleaned_dataset.xlsx"))
df = pd.read_excel(file_path)

# Text cleaning function
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(f"[{string.punctuation}]", "", text)  # Removes punctuation
    text = re.sub(r"\s+", " ", text).strip()  # Removes extra spaces
    return text

# Apply cleaning
df["cleaned_text"] = df["text"].astype(str).apply(clean_text)

# Aggregate all reviews for each product into a single text entry
df_grouped = df.groupby("product name")["cleaned_text"].apply(lambda x: " ".join(x)).reset_index()

# Convert cleaned text into TF-IDF vectors with bigrams and more features
tfidf = TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1,2))
tfidf_matrix = tfidf.fit_transform(df_grouped["cleaned_text"])

# Computing cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# recommendation function
def recommend_products(product_name, top_n=3):
    if product_name not in df_grouped["product name"].values:
        print("Product not found in dataset!")
        return []
    
    # index of the product
    idx = df_grouped[df_grouped["product name"] == product_name].index[0]
    
    # similarity scores for the product
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # similarity score (highest first)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Top N similar products
    sim_scores = sim_scores[1:top_n+1]
    
    # product indices
    product_indices = [i[0] for i in sim_scores]
    
    # top recommended product names
    return df_grouped.iloc[product_indices]["product name"].tolist()

# Example usage
print("Recommended products:", recommend_products("Sparkling ICE Sparkling Water, Variety Pack", top_n=3))

# Testing recommendations for different products
test_products = [
    "Becoming",  # A book
    "Echo Dot (3rd Gen) - Smart speaker with Alexa - Sandstone",  # An electronic item
    "Nautica Voyage By Nautica For Men. Eau De Toilette Spray, 100 ml"  # A perfume
]

# Checking recommendations
for product in test_products:
    print(f"\n Recommended products for: {product}")
    print(recommend_products(product, top_n=5))