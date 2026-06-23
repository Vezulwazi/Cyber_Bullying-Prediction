# 🛡️ Cyber-Bullying Prediction

A Natural Language Processing (NLP) and Machine Learning project that detects cyberbullying in Formspring Q&A posts.

The system processes text data, extracts features using TF-IDF, and applies multiple classification models with hyperparameter tuning to identify cyberbullying content.

---

## 📌 Dataset

- **Source:** Formspring Q&A dataset  
- **File:** `Formspring.csv`  
- **Size:** 13,147 labeled text entries  
- **Task:** Binary classification (Cyberbullying vs Non-cyberbullying)

---

## ⚙️ Methodology

### 🧹 Text Preprocessing
- Stopword removal (NLTK)
- Stemming (SnowballStemmer)
- Text cleaning and normalization

### 🔢 Feature Extraction
- TF-IDF Vectorization (`TfidfVectorizer`)

### 🤖 Models Used
- Support Vector Classifier (SVC - sigmoid kernel)
- Multinomial Naive Bayes
- Decision Tree Classifier
- Logistic Regression
- Ridge Classifier
- SGD Classifier

### 🔧 Model Optimization
- Hyperparameter tuning using `GridSearchCV`

---

## 🛠 Tech Stack

- pandas  
- numpy  
- NLTK (stopwords, SnowballStemmer)  
- scikit-learn  
  - TfidfVectorizer  
  - SVC  
  - MultinomialNB  
  - DecisionTreeClassifier  
  - LogisticRegression  
  - RidgeClassifier  
  - SGDClassifier  
  - LabelEncoder  
- matplotlib  

---

## 📈 Results

| Model | Accuracy | Precision | Recall |
|------|----------|-----------|--------|
| SVC (Sigmoid Kernel) | 95.08% | 80.52% | 25.73% |
| Multinomial Naive Bayes | 94.02% | 53.62% | 15.35% |
| Decision Tree | 93.11% | 41.88% | 33.20% |

---

## 📊 Key Insight

- The dataset is **highly imbalanced**, with most samples labeled as non-cyberbullying.
- As a result:
  - Accuracy appears high across models
  - Recall for the bullying class remains low
- This highlights the importance of:
  - Class balancing techniques
  - F1-score or recall-focused evaluation

---

## 🚀 How to Run

```bash
pip install pandas numpy nltk scikit-learn matplotlib
