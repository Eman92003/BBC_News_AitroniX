# BBC News Classification (AitroniX)

This notebook performs **classification of BBC News articles** into 5 categories using a complete pipeline starting from data loading and exploration, through text preprocessing, and ending with training and evaluating different machine learning and deep learning models.

The notebook experiments with two main approaches:

1. **TF‑IDF + Multinomial Naive Bayes** (classic machine learning)  
2. **LSTM with learned embeddings** (deep learning)

---

## 📊 Dataset

- File used: `bbc-news-data.csv`  
- Source: Kaggle input `../kaggle/input/bbc-news-archive/bbc-news-data.csv`  
- Shape: 2225 rows × 4 columns

**Columns:**

| Column    | Description                       |
|-----------|-----------------------------------|
| category  | News category (target)            |
| filename  | Original filename of the article  |
| title     | Article title                     |
| content   | Full text of the article          |

**Class distribution:** Approximately balanced across the 5 categories (visualized using a Pie Chart).

---

## 🧹 Data Preprocessing

- Dropped unnecessary columns (`filename`, `title`), keeping only `content`.  
- Converted text to lowercase.  
- Removed punctuation.  
- Removed URLs, hashtags, and unwanted characters using regex.  
- Removed stopwords using NLTK.  
- Converted `category` labels to numerical values using `LabelEncoder`.  

---

## 🧠 Modeling Approaches

| Approach        | Representation           | Model                            | Result |
|-----------------|-------------------------|---------------------------------|--------|
| Classic ML      | TF‑IDF                  | Multinomial Naive Bayes (80/20) | Accuracy = 0.9708 |
| Deep Learning   | Learned embeddings       | LSTM (from scratch)              | Train Accuracy = 0.98, Test Accuracy = 0.87 |
| Deep Learning   | Pretrained embeddings (FastText) | LSTM                     | Train Accuracy = 0.95, Test Accuracy = 0.93 |

- The deep learning models were trained using **Tokenizer + padding** to prepare the vocabulary and sequences.  
- One LSTM model used embeddings learned from scratch, achieving **high train accuracy (~98%)** but with overfitting.  
- Another LSTM used **pretrained embeddings from FastText**, achieving **train accuracy 95% and test accuracy 93%**.  

---

## 📑 Notebook Contents

1. **Reading Data:** Load the dataset and display sample rows.  
2. **Dataset Info:** Check shape, info, and missing values.  
3. **Class Distribution:** Count values and visualize with a Pie Chart.  
4. **Data Preprocessing:** Lowercasing, punctuation removal, regex cleaning, stopword removal, and label encoding.  
5. **Modeling:**  
   - TF‑IDF + Multinomial Naive Bayes  
   - LSTM with embeddings (from scratch and pretrained)  
6. **Results & Evaluation:** Compare accuracy for classic ML vs deep learning models.

---

## 🛠 Libraries Used

- `pandas`, `matplotlib`  
- `scikit-learn` (TF‑IDF, train_test_split, MultinomialNB, metrics, LabelEncoder)  
- `nltk` (stopwords)  
- `tensorflow` / `keras` (Tokenizer, pad_sequences, LSTM)

---
