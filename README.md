# IMDB Sentiment Analysis with Logistic Regression

This project is a Natural Language Processing (NLP) implementation that classifies IMDB movie reviews as either **Positive** or **Negative**. It utilizes TF-IDF vectorization and a Logistic Regression model to achieve an accuracy of approximately **83.8%**.

A key feature of this project is the **interpretability analysis**, extracting the specific words that weigh most heavily towards positive or negative sentiment.

## 📊 Dataset
* **Source:** IMDB Dataset of 50,000 Movie Reviews.
* **Structure:** Two columns (`review`, `sentiment`).
* **Balance:** The dataset is perfectly balanced with 25,000 positive and 25,000 negative reviews.

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:**
    * `pandas` (Data manipulation)
    * `numpy` (Numerical operations)
    * `scikit-learn` (Modeling, Vectorization, Metrics)
    * `re` (Regular Expressions for text cleaning)

## ⚙️ Methodology

### 1. Data Preprocessing
Raw text data is noisy. A custom cleaning pipeline was implemented using **Regex** to:
* Convert text to lowercase.
* Remove non-word characters (punctuation/symbols).
* Remove numbers/digits.
* Remove single characters (orphaned letters like 'a', 's' often left after punctuation removal).
* Remove multiple consecutive spaces.
* **Stop Word Removal:** A custom list of stop words (e.g., "the", "is", "at") was used to filter out non-meaningful tokens.

### 2. Feature Extraction (TF-IDF)
The text was converted into numerical vectors using **TfidfVectorizer**.
* **Configuration:** `max_features=500`
* **Logic:** The model focuses only on the top 500 most important words across the corpus to reduce dimensionality while retaining predictive power.

### 3. Model Training
* **Algorithm:** Logistic Regression.
* **Target Encoding:** Label Encoded (0 = Negative, 1 = Positive).
* **Split:** 80% Training / 20% Testing (`random_state=42`).

## 📈 Results & Performance

| Metric | Score |
| :--- | :--- |
| **Accuracy** | **83.85%** |

### Model Interpretation (Feature Importance)
Since Logistic Regression is a linear model, we can interpret the model's coefficients (`model.coef_`) to understand which words drive sentiment.

* **Positive Coefficients:** Push the prediction toward **1 (Positive)**.
* **Negative Coefficients:** Push the prediction toward **0 (Negative)**.

#### Top Predictors Found by the Model:

| Top Positive Words | Top Negative Words |
| :--- | :--- |
| **excellent** (+6.19) | **worst** (-9.78) |
| **great** (+5.50) | **waste** (-8.05) |
| **amazing** (+4.87) | **awful** (-7.58) |
| **wonderful** (+4.80) | **boring** (-6.62) |
| **perfect** (+4.61) | **terrible** (-6.10) |

## 🚀 How to Run

1.  **Clone the repository:**
    
2.  **Install dependencies:**
    ```bash
    pip install pandas numpy scikit-learn
    ```
3.  **Ensure the dataset is present:**
    Place `IMDB_Dataset.csv` in the root directory.
4.  **Run the Notebook:**
    Open the Jupyter Notebook and execute all cells.

## 📝 Conclusion
Despite limiting the vocabulary to only the top 500 words, the model achieved high accuracy. The feature importance analysis confirms that the model successfully learned semantic meaning, identifying strong adjectives ("excellent", "waste", "boring") as the primary drivers of sentiment.
