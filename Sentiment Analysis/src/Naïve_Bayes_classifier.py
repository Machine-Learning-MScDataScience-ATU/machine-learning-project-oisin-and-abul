# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 05:52:54 2025

@author: mohsin
"""

import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV

# Loading the dataset
input_path = r"C:/Users/mohsi/OneDrive/Documents/GitHub/machine-learning-project-oisin-and-abul/recommendation-system/data/processed/sentiment_analysis.xlsx"
df = pd.read_excel(input_path)

# Droping rows with missing values in 'clean_text' or 'helpful'
df.dropna(subset=['clean_text', 'helpful'], inplace=True)

# Converting text to numerical features using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["clean_text"])
y = df["sentiment_label"]

# Balancing classes using SMOTE
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Training the model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluating the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# confusion matrix
plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap="Blues",
            xticklabels=["Negative", "Neutral", "Positive"],
            yticklabels=["Negative", "Neutral", "Positive"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix After SMOTE")
plt.show()

# Hyperparameter tuning
param_grid = {
    'alpha': [0.01, 0.1, 0.5, 1, 2, 5],
    'fit_prior': [True, False]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best hyperparameters
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Evaluating the tuned model
y_pred = best_model.predict(X_test)
print("Accuracy with best model:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Visualize confusion matrix for the tuned model
plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap="Blues",
            xticklabels=["Negative", "Neutral", "Positive"],
            yticklabels=["Negative", "Neutral", "Positive"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix After Hyperparameter Tuning")
plt.show()
