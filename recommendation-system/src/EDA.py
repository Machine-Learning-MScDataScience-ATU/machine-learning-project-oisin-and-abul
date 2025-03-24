# -*- coding: utf-8 -*-
"""
Created on Sat Mar 15 13:02:57 2025

@author: oisin
"""

import pandas as pd
import numpy as np#
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


#read in the cleaned dataset
input_path = r"C:\Users\oisin\OneDrive\Desktop\5th Year-Masters\Machine Learning\Project Work\cleaned_dataset.xlsx"
df = pd.read_excel(input_path)

#create a pivot table to compare average positive and negative scores in each category
rating_table = df.pivot_table(index=['category'],
                              columns=['target'],
                              values='ratings')
print(rating_table)

#crrealation matrix of positive and negative scores
corr_matrix = rating_table.corr()
corr_matrix.head()

#groups by the product's name and calculates an average rating score and how many
#reviews of the product there are
product_stats = df.groupby('category').agg({'ratings': [np.size, np.mean]})
product_stats.head()

#plot of the mean score of ratings per category
fig, axs = plt.subplots(figsize=(8, 8))
sns.scatterplot(product_stats, s=100, color="blue")

#creates titles and labels for the chart
plt.title("Product Ratings Scatter Plot", fontsize=14)
plt.xlabel("Products", fontsize=12)
plt.ylabel("Ratings", fontsize=12)
plt.ylim(0, 5)  # Ratings typically range from 0 to 5
plt.grid(axis='y', linestyle="--", alpha=0.7)

plt.show()

#scales the product rating values to be passed through PCA
scaler = StandardScaler()
scaled_rating = scaler.fit_transform(df['ratings'].values.reshape(-1, 1))

pca = PCA(n_components=1)
pca_rating = pca.fit_transform(scaled_rating)

#stores the new pca data in a new dataframe
df_pca = pd.DataFrame(pca_rating)
print("\nPCA Transformed Data:")
print(df_pca)


