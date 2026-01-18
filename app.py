import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords

# Ensure stopwords are available
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load trained artifacts
model = pickle.load(open("model/classifier.pkl", "rb"))
tfidf = pickle.load(open("model/tfidf.pkl", "rb"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

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
