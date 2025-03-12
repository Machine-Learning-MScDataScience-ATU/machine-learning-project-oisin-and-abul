# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 00:34:17 2025

@author: abul mohsin (l00187574)
"""
import os
import pandas as pd

# Print current working directory
print("Current Working Directory:", os.getcwd())

# Load CSV with correct relative path
csv_path = "data/export_book.csv"  # Make sure this path is correct
df = pd.read_csv(csv_path)

# Display first few rows
print(df.head())  
