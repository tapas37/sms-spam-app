import streamlit as st
import pickle
import string

# ---------------------------
# Manual stopwords list
# ---------------------------
STOPWORDS = set("""
a about above after again against all am an and any are aren't as at be because been
before being below between both but by can't cannot could couldn't did didn't do does
doesn't doing don't down during each few for from further had hadn't has hasn't have
haven't having he he'd he'll he's her here here's hers herself him himself his how
how's i i'd i'll i'm i've if in into is isn't it it's its itself let's me more most
mustn't my myself no nor not of off on once only or other ought our ours ourselves out
over own same shan't she she'd she'll she's should shouldn't so some such than that
that's the their theirs them themselves then there there's these they they'd they'll
they're they've this those through to too under until up very was wasn't we we'd we'll
we're we've were weren't what what's when when's where where's which while who who's
whom why why's with won't would wouldn't you you'd you'll you're you've your yours
yourself yourselves
""".split())

# ---------------------------
# Simple stemmer function
# ---------------------------
def simple_stem(word):
    suffixes = ['ing', 'ly', 'ed', 's', 'es', 'ment']
    for suffix in suffixes:
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            return word[:-len(suffix)]
    return word

# ---------------------------
# Text preprocessing
# ---------------------------
def transform_text(text):
    # lowercase
    text = text.lower()

    # basic tokenization
    tokens = text.split()

    # remove punctuation and stopwords
    tokens = [word.strip(string.punctuation) for word in tokens]
    tokens = [word for word in tokens if word not in STOPWORDS and word]

    # simple stemming
    tokens = [simple_stem(word) for word in tokens]

    return " ".join(tokens)

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
    st.header("Spam" if result == 1 else "Not Spam")
