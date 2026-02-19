# Resume Job Role Classifier

## Project Explanation

The **Resume Job Role Classifier** is a mini project that predicts the job category of a resume automatically. It helps in identifying the most suitable job role based on resume content.

---

## Training Phase(`src/logistic_regression.py`)

- The resume dataset is first **loaded** and **cleaned** by removing emails, numbers, and special characters.
- The text is then **tokenized**, **lemmatized**, and **stopwords are removed** to simplify and normalize the resume content.
- Using the cleaned text, a **Logistic Regression model** is trained to learn patterns associated with different job roles.
- After successful training, the **trained model** and **TF-IDF vectorizer** are saved for future predictions.

---

## Prediction Pipeline (`src/main.py`)

- The previously saved **model** and **TF-IDF vectorizer** are loaded.
- A `predict_resume` function is used to handle the complete preprocessing pipeline:
  - Text cleaning  
  - Tokenization  
  - Stopwords removal  
  - Vectorization  
- The processed resume text is then passed to the trained model to **predict the job role**.

---

## Streamlit Application(`src/main.py`)

- A user-friendly **Streamlit web application** is created.
- Users can paste resume text into the input area.
- When the **“Predict Category”** button is clicked:
  - Empty resume input shows a warning.
  - Very short resume text shows an error.
  - Valid resume text is processed and the **predicted job role is displayed**.
- All predictions and rejected inputs are **logged** for monitoring and debugging.

---

## Project Structure 
```
MINI_AI_PRODUCT_DEPLOYMENT/
│
├── data/
│   ├── input/
│   │   └── resume_dataset.csv
│   │
│   └── output/
│       ├── model.pkl
│       └── tfidf.pkl
│
├── logs/
│   └── predictions.log
│
├── src/
│   ├── logistic_regression.py
│   └── main.py
│
└── README.md

```
## How to Run 
### 1. Clone the repository
```
git clone https://github.com/JaspinderKaurWalia26/resume-classifier-ai-app.git
cd resume-classifier-ai-app
```
### 2. Create a virtual environment (optional)
```
python -m venv venv
```
### 3. Activate the virtual environment
- Windows:
```
venv\Scripts\activate
```
- Linux/Mac:
```
source venv/bin/activate
```
### 4. Install dependencies
```
pip install -r requirements.txt
```
### 4. Train the Model
```
python src/logistic_regression.py
```

### 5. Run the Streamlit App
```
streamlit run src/main.py
```
### 6. Check outputs

- Logs: logs/predictions.log




