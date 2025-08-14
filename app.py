import streamlit as st
import pickle
import string
import nltk
import os
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# ---------------------------
# Setup NLTK data path
# ---------------------------
nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
nltk.data.path.append(nltk_data_dir)

ps = PorterStemmer()

# ---------------------------
# Text preprocessing
# ---------------------------
def transform_text(text):
    text = text.lower()  # lowercase
    text = nltk.word_tokenize(text)  # tokenization
    y = [i for i in text if i.isalnum()]  # remove non-alphanumeric
    text = y[:]
    y.clear()
    for i in text:  # remove stopwords and punctuation
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:  # stemming
        y.append(ps.stem(i))
    return " ".join(y)

# ---------------------------
# Load model and vectorizer
# ---------------------------
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# ---------------------------
# Streamlit app
# ---------------------------
st.title("SMS Spam Classifier")
input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    transformed_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transformed_sms])
    result = model.predict(vector_input)[0]
    st.header("Spam" if result == 1 else "Not Spam")
