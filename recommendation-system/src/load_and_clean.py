# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 03:13:21 2025

@author: mohsi
"""
import os
import pandas as pd

# Defining data path
data_path = "C:/Users/mohsi/OneDrive/Documents/GitHub/machine-learning-project-oisin-and-abul/recommendation-system/data/"  # path from 'src' folder to 'data' folder

# List of dataset files
files = ["export_book.csv", "export_curcumin.csv", "export_electronics.csv", 
         "export_food.csv", "export_mask.csv", "export_movie.csv", "export_perfume.csv"]

# Mapping filenames to categories
category_map = {
    "export_book.csv": "Books",
    "export_curcumin.csv": "Curcumin",
    "export_electronics.csv": "Electronics",
    "export_food.csv": "Food",
    "export_mask.csv": "Masks",
    "export_movie.csv": "Movies",
    "export_perfume.csv": "Perfumes"
}

# Merging datasets with the 'category' column
df_all = pd.concat([
    pd.read_csv(os.path.join(data_path, file)).assign(category=category_map[file]) 
    for file in files
], ignore_index=True)

# Save the combined dataset to an Excel file in a new 'processed' folder under 'data/'
output_path = "data/processed/combined_dataset.xlsx"
df_all.to_excel(output_path, index=False)

print(f"Dataset saved to {output_path}")

# Dataset info
print("Combined Dataset Info:")
print(df_all.info())

# First few rows
print(df_all.head())



