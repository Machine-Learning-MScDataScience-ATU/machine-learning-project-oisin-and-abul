# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 05:16:16 2025

@author: mohsin
"""

import pandas as pd
import re
import string
#import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns

# Download necessary NLTK resources
#nltk.download("vader_lexicon")
#nltk.download("stopwords")
#nltk.download("punkt")
#nltk.download("wordnet")

# Loading dataset
input_path = r"C:/Users/mohsi/OneDrive/Documents/GitHub/machine-learning-project-oisin-and-abul/recommendation-system/data/processed/cleaned_dataset.xlsx"
df = pd.read_excel(input_path)

# Droping missing values
df.dropna(subset=["text"], inplace=True)

# Initialize NLTK components
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
sia = SentimentIntensityAnalyzer()

# Function to clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

# Applying text cleaning
df["clean_text"] = df["text"].swifter.apply(clean_text)

# Function to get sentiment score using VADER
def get_vader_sentiment(text):
    score = sia.polarity_scores(text)["compound"]
    return 1 if score > 0.05 else -1 if score < -0.05 else 0

# Applying sentiment analysis
df["sentiment_label"] = df["clean_text"].swifter.apply(get_vader_sentiment)

# Saving the dataset with sentiments
output_path = r"C:/Users/mohsi/OneDrive/Documents/GitHub/machine-learning-project-oisin-and-abul/recommendation-system/data/processed/sentiment_analysis.xlsx"
df.to_excel(output_path, index=False)

print(f"Sentiment analysis saved to: {output_path}")

# Sentiment Distribution Visualization
order = ["Negative", "Neutral", "Positive"]

# Convert numeric sentiment labels to actual text labels
df["sentiment_category"] = df["sentiment_label"].map({-1: "Negative", 0: "Neutral", 1: "Positive"})

# Plot 
sns.countplot(x="sentiment_category", data=df, order=order, palette=["#FF6F61", "#6B8E23", "#4682B4"])
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()



