# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 21:44:11 2025

@author: mohsi
"""
import pandas as pd

# Loading the merged dataset
input_path = "C:/Users/mohsi/OneDrive/Documents/GitHub/machine-learning-project-oisin-and-abul/recommendation-system/data/processed/combined_dataset.xlsx"
df = pd.read_excel(input_path)

# 2. Checking initial dataset info
print(df.info())

# 3. Dropping unnecessary columns
df = df.drop(columns=["Unnamed: 0", "Unnamed: 6"], errors="ignore")

# 4. Checking for missing values
print("Missing values before handling:\n", df.isnull().sum())

# 5. Checking for duplicate rows
print("Duplicate rows:", df.duplicated().sum())

# 6. Removing duplicate rows
df = df.drop_duplicates()
print("Duplicate rows after removal:", df.duplicated().sum())

# 7. Converting data types
df["ratings"] = df["ratings"].astype(float)
df["helpful"] = pd.to_numeric(df["helpful"], errors="coerce").astype("Int64")

# 8. Handling the date column
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# 9. Checking for invalid dates
print("Number of NaT (invalid dates):", df["date"].isna().sum())

# 10. Filling missing dates with the most frequent date
most_frequent_date = df["date"].mode()[0]
df.loc[:, "date"] = df["date"].fillna(most_frequent_date)

# 11. Verifying if there are still missing values
print("Number of NaT (invalid dates) after filling:", df["date"].isna().sum())

# 12. Checking final data types
print(df.dtypes)

# 13. Saving the cleaned dataset
df.to_excel("C:/Users/mohsi/OneDrive/Documents/GitHub/machine-learning-project-oisin-and-abul/recommendation-system/data/processed/cleaned_dataset.xlsx", index=False)
print("Cleaned dataset saved successfully!")

