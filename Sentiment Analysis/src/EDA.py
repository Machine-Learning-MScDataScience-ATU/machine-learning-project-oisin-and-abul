# -*- coding: utf-8 -*-
"""
Created on Sat Mar 15 13:02:57 2025

@author: oisin
"""

#import libraries
import pandas as pd
import numpy as np#
import matplotlib.pyplot as plt
import seaborn as sns

#read in the cleaned dataset
input_path = r"cleaned_dataset.xlsx"
df = pd.read_excel(input_path)

#create a pivot table to compare average positive and negative scores in each category
rating_table = df.pivot_table(index=['category'],
                              columns=['target'],
                              values='ratings')
print(rating_table)
print(rating_table.describe())

#create a bar chart of the count of each target per category
rating_table.plot(kind='bar', figsize=(10, 6))

plt.xlabel("Category")
plt.ylabel("Ratings")
plt.title("Ratings by Category and Target")
plt.legend(title="Target")
plt.xticks(rotation=45)
plt.show()

#correalation matrix of positive and negative scores
corr_matrix = rating_table.corr()
corr_matrix.head()

#groups by the product's name and calculates an average rating score and how many
#reviews of the product there are
product_stats = df.groupby('category').agg({'ratings': [np.size, np.mean]})
product_stats.head()

#redefines the column headings and ensures category is not the index
product_stats.columns = ['count', 'mean_rating']
product_stats = product_stats.reset_index()

#plot of the mean score of ratings per category
fig, axs = plt.subplots(figsize=(8, 8))
sns.scatterplot(product_stats, s=100, color="blue")
plt.title("Product Ratings Scatter Plot", fontsize=14)
plt.xlabel("Products", fontsize=12)
plt.ylabel("Ratings", fontsize=12)
#reduces the range of the y-axis from 3 to 5
plt.ylim(3, 5)
plt.grid(axis='y', linestyle="--", alpha=0.7)
plt.show()

#counts how much of each target is in the target column
counts = pd.Series(df['target']).value_counts()
print(counts)

#plots a bar chart of the total target counts
counts.plot(kind='bar', figsize=(10, 6))
plt.xlabel("Target")
plt.ylabel("Counts")
plt.title("Target Counts")
plt.legend(title="Target")
plt.show()

#read in the new dataset with the updated sentiment_label
input_path2 = r"sentiment_analysis.xlsx"
df2 = pd.read_excel(input_path2)

#counts each value in the sentiment_label column
new_counts = pd.Series(df2['sentiment_label']).value_counts()
print(new_counts)

#plots a bar chart of the new target values
new_counts.plot(kind='bar', figsize=(10, 6))
plt.xlabel("Target")
plt.ylabel("Counts")
plt.title("Updated Target Counts")
plt.legend(title="Target")
plt.show()
