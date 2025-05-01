# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 20:24:12 2025

@author: oisin
"""

#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, \
    confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve

#read in the cleaned dataset
input_path = r"sentiment_analysis.xlsx"
df = pd.read_excel(input_path)

#converts text into token counts and passes it through a tf-idf transformer
#reduces the impact of tokens that occur very frequently
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
counts = vectorizer.fit_transform(df['reviews'].values)
#stores the token counts in x
x = counts

#encodes the target labels from n and p into numbers and stores the values in y
le = LabelEncoder()
y = le.fit_transform(df['sentiment_label'].values)  # Converts categories into numbers

#uses smote to synthetically oversample the neutral and negative values in
#sentiment_label as they areunderrepresented
smote = SMOTE(random_state=42)
x, y = smote.fit_resample(x, y)

#split the data
X_train, X_test, y_train, y_test = \
    train_test_split(x, y,
                     test_size=0.2,
                     shuffle=True,
                     random_state=42)

#create and fit the model with the given parameters
model = LogisticRegression(C=10, class_weight='balanced')
model.fit(X_train, y_train)

#calculate accuracy score and classification report on the fitted model
y_pred = model.predict(X_test)
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

#calculates a cross validation score of the model
scores = cross_val_score(model, X_train, y_train, cv=5)
print("Cross-validation accuracy:", scores.mean())

#creates a confusion matrix of the test and predicted values
cm = confusion_matrix(y_test, y_pred, normalize='true')

#generates a ConfusionMatrixDisplay of the predicted values
fig, axs = plt.subplots(figsize=(8, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap=plt.cm.Blues, ax=axs)
plt.title('Normalized Confusion Matrix')
plt.show()

#calculates a predict_probability score, used for precision and recall curve
y_test_prob = model.predict_proba(X_test)

#plots precision-recall curves for each class in the target values
#first class
precision, recall, thresholds = precision_recall_curve(
y_test, y_test_prob[:, 0], pos_label=1)

plt.figure(figsize=(10,8))
plt.plot(thresholds, precision[:-1], label="Precision", color='blue')
plt.plot(thresholds, recall[:-1], label="Recall", color='red')
plt.xlabel("Threshold")
plt.ylabel("Scores of Precision and Recall")
plt.legend()
plt.title("Precision and Recall vs. Threshold")
plt.show()

#second class
precision, recall, thresholds = precision_recall_curve(
y_test, y_test_prob[:, 1], pos_label=1)

plt.figure(figsize=(10,8))
plt.plot(thresholds, precision[:-1], label="Precision", color='blue')
plt.plot(thresholds, recall[:-1], label="Recall", color='red')
plt.xlabel("Threshold")
plt.ylabel("Scores of Precision and Recall")
plt.legend()
plt.title("Precision and Recall vs. Threshold")
plt.show()

#third class
precision, recall, thresholds = precision_recall_curve(
y_test, y_test_prob[:, 2], pos_label=1)

plt.figure(figsize=(10,8))
plt.plot(thresholds, precision[:-1], label="Precision", color='blue')
plt.plot(thresholds, recall[:-1], label="Recall", color='red')
plt.xlabel("Threshold")
plt.ylabel("Scores of Precision and Recall")
plt.legend()
plt.title("Precision and Recall vs. Threshold")
plt.show()
