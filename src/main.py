import streamlit as st
import joblib
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

import logging


logging.basicConfig(
    filename='logs/predictions.log',      
    level=logging.INFO,              
    format='%(asctime)s - %(levelname)s - %(message)s'  
)



model = joblib.load('data/output/model.pkl')
tfidf = joblib.load('data/output/tfidf.pkl')  


stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def preprocess(text):
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens)


def predict_resume(text):
    cleaned = clean_text(text)
    preprocessed = preprocess(cleaned)
    vector = tfidf.transform([preprocessed])
    result = model.predict(vector)
    logging.info(f"Resume Text Input: {text} | Predicted Role: {result}")
    return result[0]

st.title("Resume Job Role Classifier")
resume_text = st.text_area("Paste Resume Text Here")

if st.button("Predict Category"):
    text = resume_text.strip()
    
    if text == "":
        st.warning("Resume text is empty")
        logging.warning("Rejected: Empty resume input")
    elif len(text) < 5:
        st.error("Rejected as length is too small")
        logging.error(f"Rejected Resume (too short): {text}")
    else:
        prediction = predict_resume(text)
        st.success(f"Predicted Role: {prediction}")
