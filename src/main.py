import streamlit as st
import joblib
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

import logging

# logging setup
logging.basicConfig(
    filename='logs/predictions.log',      
    level=logging.INFO,              
    format='%(asctime)s - %(levelname)s - %(message)s'  
)


# loading the trained model and tfidf vectors
model = joblib.load('data/output/model.pkl')
tfidf = joblib.load('data/output/tfidf.pkl')  


stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# cleaning text function
def clean_text(text):
    text = str(text).lower()                   # lowering text
    text = re.sub(r'\S+@\S+', ' ', text)       # removing mails
    text = re.sub(r'\d+', ' ', text)           # removing digits 
    text = re.sub(r'[^a-zA-Z ]', ' ', text)    # Removes all characters except alphabets and spaces
    text = re.sub(r'\s+', ' ', text)           # Replaces multiple spaces with a single space
    return text.strip()

# preprocessing function
def preprocess(text):
    tokens = text.split()                                 # tokenization
    tokens = [w for w in tokens if w not in stop_words]   # stopwords removal
    tokens = [lemmatizer.lemmatize(w) for w in tokens]    # lemmatization
    return " ".join(tokens)

# predict resume function: Takes input as text and returns the category
def predict_resume(text):
    cleaned = clean_text(text)
    preprocessed = preprocess(cleaned)
    vector = tfidf.transform([preprocessed])
    result = model.predict(vector)
    logging.info(f"Resume Text Input: {text} | Predicted Role: {result}")
    return result[0]

# App title shown on the Streamlit UI
st.title("Resume Job Role Classifier")
# Creating Text box where user pastes resume content
resume_text = st.text_area("Paste Resume Text Here")

# Button to trigger prediction
if st.button("Predict Category"):
    # Removing extra spaces from start and end
    text = resume_text.strip()
    
    if text == "":
        st.warning("Resume text is empty")
        # Log empty input rejection
        logging.warning("Rejected: Empty resume input")
    elif len(text) < 5:
        st.error("Rejected as length is too small")
        # Log short resume rejection
        logging.error(f"Rejected Resume (too short): {text}")
    else:
        prediction = predict_resume(text)
        st.success(f"Predicted Role: {prediction}")
