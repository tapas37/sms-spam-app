import streamlit as st
import pickle
import string
import nltk
import os
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

# ---------------------------
# Setup NLTK data for Streamlit Cloud
# ---------------------------
nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

# Tell NLTK to look in this directory
nltk.data.path.append(nltk_data_dir)

# Download required datasets if not present
nltk.download('stopwords', download_dir=nltk_data_dir)
nltk.download('punkt', download_dir=nltk_data_dir)

# ---------------------------
# Text preprocessing
# ---------------------------
def transform_text(text):
    # lower case
    text = text.lower()

    # tokenization
    text = nltk.word_tokenize(text)

    # remove non-alphanumeric
    y = [i for i in text if i.isalnum()]
    text = y[:]
    y.clear()

    # remove stopwords and punctuation
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()

    # stemming
    for i in text:
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
    # 1. Preprocess
    transformed_sms = transform_text(input_sms)

    # 2. Vectorize
    vector_input = tfidf.transform([transformed_sms])

    # 3. Predict
    result = model.predict(vector_input)[0]

    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
