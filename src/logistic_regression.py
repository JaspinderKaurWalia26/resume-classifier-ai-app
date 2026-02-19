import pandas as pd
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score
import joblib


# Loading Data
data = pd.read_csv("data/input/resume_dataset.csv")

# Cleaning function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\S+@\S+', ' ', text)     
    text = re.sub(r'\d+', ' ', text)         
    text = re.sub(r'[^a-zA-Z ]', ' ', text)  
    text = re.sub(r'\s+', ' ', text)         
    return text.strip()

data["cleaned_text"] = data["Resume_Text"].apply(clean_text)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens)

data["final_text"] = data["cleaned_text"].apply(preprocess)


# Train / Validation / Test Split

X = data["final_text"]
y = data["Category"]

# Split: 70% Train, 30% Temp
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30,  random_state=42
)
# Split Temp: 15% Validation, 15% Test 
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)


# TF-IDF Vectorization (with bigrams)

tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1,2))
X_train_vec = tfidf.fit_transform(X_train)
X_val_vec = tfidf.transform(X_val)
X_test_vec = tfidf.transform(X_test)
joblib.dump(tfidf, 'data/output/tfidf.pkl')


# Logistic Regression Model

lr_model = LogisticRegression(max_iter=1000, class_weight="balanced")
lr_model.fit(X_train_vec, y_train)
lr_pred = lr_model.predict(X_test_vec)

print("Logistic Regression Test Report ")
print(classification_report(y_test, lr_pred))
lr_acc = accuracy_score(y_test, lr_pred)
lr_f1 = f1_score(y_test, lr_pred, average="weighted")
print("Test Accuracy:", lr_acc)
print("Weighted F1 Score:", round(lr_f1, 2))

joblib.dump(lr_model,'data/output/model.pkl')
