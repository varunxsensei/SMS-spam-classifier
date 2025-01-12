import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Load the trained model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Initialize NLTK resources
nltk.download('stopwords')
nltk.download('punkt')


# Function to process and transform the text
def text_tokenization(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    li = []
    for i in text:
        if i.isalnum():
            li.append(i)
    text = li[:]
    li.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            li.append(i)
    text = li[:]
    li.clear()
    ps = PorterStemmer()
    for i in text:
        li.append(ps.stem(i))
    return " ".join(li)


# Streamlit app UI
st.title("SMS Spam Classifier")
st.subheader("Classify your SMS as Spam or Not Spam")

# Get user input
sms_text = st.text_area("Enter the SMS text:")

# Button to classify the SMS
if st.button("Classify SMS"):
    if sms_text:
        # Transform the input text
        processed_text = text_tokenization(sms_text)
        transformed_text = vectorizer.transform([processed_text]).toarray()

        # Predict the class (0 = Not Spam, 1 = Spam)
        prediction = model.predict(transformed_text)

        if prediction == 0:
            st.write("The SMS is **Not Spam**")
        else:
            st.write("The SMS is **Spam**")
    else:
        st.error("Please enter an SMS text to classify.")
