import streamlit as st
st.write("")


import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
import os

# ---------- SAFE NLTK SETUP FOR STREAMLIT CLOUD ----------
@st.cache_resource
def load_stopwords():
    try:
        return set(stopwords.words('english'))
    except LookupError:
        nltk.download('stopwords')
        return set(stopwords.words('english'))

stop_words = load_stopwords()
# --------------------------------------------------------

# ---------- LOAD MODEL FILES (FAIL FAST IF MISSING) ------
model_path = "model/classifier.pkl"
tfidf_path = "model/tfidf.pkl"

if not os.path.exists(model_path) or not os.path.exists(tfidf_path):
    st.error("Model files not found. Deployment failed.")
    st.stop()

model = pickle.load(open(model_path, "rb"))
tfidf = pickle.load(open(tfidf_path, "rb"))
# --------------------------------------------------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

# ---------- STREAMLIT UI ----------
st.set_page_config(page_title="Job Role Recommendation", layout="centered")

st.title("Job Role Recommendation System")
st.write("Paste your resume text or skills below:")

user_input = st.text_area("Resume / Skills", height=180)

if st.button("Predict Job Role"):
    if user_input.strip():
        cleaned = clean_text(user_input)
        vect = tfidf.transform([cleaned])
        prediction = model.predict(vect)
        st.success(f"Recommended Job Role: {prediction[0]}")
    else:
        st.warning("Please enter some text.")

