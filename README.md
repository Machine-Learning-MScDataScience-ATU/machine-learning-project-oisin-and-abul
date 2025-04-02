# **Sentiment Analysis Project**

## **Overview**
Welcome to the **Sentiment Analysis Project**! 🚀 This project, created by **Abul** and **Oisin**, aims to delve into the fascinating world of text mining and sentiment analysis. Using **Amazon Customer Review Data**, our goal was to analyze and predict the sentiment of customer reviews—whether they are **Positive**, **Neutral**, or **Negative**.

### **Goal**
The primary goal of this project was to build a machine learning pipeline that:
1. **Processes** and **cleans** raw text data from Amazon customer reviews.
2. **Analyzes sentiment** using powerful natural language processing (NLP) techniques.
3. **Compares models** to determine the most effective approach for sentiment classification.

## **Approach**

### **1. Data Collection & Preprocessing** 
We began by working with the **Amazon Customer Review Dataset**, which contains a variety of customer reviews, including text and sentiment labels. The steps taken for data preprocessing were as follows:
- **Lowercasing**: Converting all text to lowercase to ensure uniformity.
- **Removing Noise**: Stripping out numbers, special characters, and punctuation that could affect sentiment classification.
- **Tokenization**: Breaking the text into words (tokens).
- **Stopwords Removal**: Eliminating common words that don't add much value to sentiment analysis.
- **Lemmatization**: Reducing words to their root form for more effective processing.

### **2. Sentiment Analysis with VADER**
For sentiment analysis, we leveraged the **VADER Sentiment Analysis Tool** from **NLTK**. VADER is known for being effective in analyzing sentiments from social media posts and reviews, where sentiments are often expressed in informal language. The sentiment scores were categorized into:
- **Positive** = Positive Sentiment 😊
- **Neutral** = Neutral Sentiment 😐
- **Negative** = Negative Sentiment 😔

### **3. Text Vectorization**
We then used **TF-IDF (Term Frequency-Inverse Document Frequency)** vectorization to convert the cleaned text data into numerical features. This method helps:
- **Downweight** common words that appear too frequently across the reviews.
- **Highlight** more important words based on their uniqueness in the dataset.

### **4. Balancing the Classes with SMOTE**
To handle the class imbalance (where certain sentiment classes dominate), we applied **SMOTE (Synthetic Minority Oversampling Technique)**. This technique generates synthetic examples for the underrepresented classes, ensuring the dataset remains balanced and the model learns to classify all sentiments accurately.

### **5. Model Training & Evaluation**
We experimented with two classic machine learning models for text classification:
1. **Logistic Regression**: A widely-used algorithm known for its simplicity and efficiency in binary and multi-class classification tasks.
2. **Naive Bayes**: A probabilistic model that works exceptionally well for text classification tasks like sentiment analysis.

After training both models, we evaluated their performance using key metrics:
- **Accuracy**
- **Precision**
- **Recall**

Additionally, we visualized the **confusion matrix** for both models, which helped us assess how well each model predicted the sentiment categories.

### **6. Visualizations**
To make our findings easier to interpret, we visualized:
- **Sentiment Distribution**: Using a count plot to visualize how many reviews belong to each sentiment category.
- **Confusion Matrices**: To evaluate how well each model performed in predicting the correct sentiment.
- **Precision and Recall Curves**: To visualize the trade-off between precision and recall for each model.

### **7. Model Comparison**
Once the models were evaluated, we compared them based on **accuracy**, **precision**, and **recall**. This comparison helped us identify the best-performing model and provided insights into how to improve future iterations.

## **Outcome**
By the end of the project, both **Logistic Regression** and **Naive Bayes** showed promising results. Through careful tuning and evaluation, we managed to build a robust sentiment analysis pipeline.

### **Future Improvements**
In the future, we aim to:
- Experiment with more sophisticated models like **XGBoost** and **Random Forest**.
- Perform **hyperparameter tuning** to further optimize model performance.
- Explore advanced **text preprocessing** techniques and **feature engineering** to improve prediction accuracy.

---

## **Technologies Used**
- **Python** 🐍
- **pandas** (for data manipulation)
- **NLTK** (for text processing and sentiment analysis)
- **scikit-learn** (for machine learning models and evaluation)
- **SMOTE** (for balancing the dataset)
- **matplotlib** & **seaborn** (for visualizations)
