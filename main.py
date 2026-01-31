import streamlit as st

st.markdown("""
<style>
textarea + div {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)


import string
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    words = text.split()   # 👈 NO nltk.word_tokenize

    y = []
    for word in words:
        if word.isalnum():
            y.append(word)

    y = [ps.stem(word) for word in y
         if word not in stopwords.words('english')
         and word not in string.punctuation]

    return " ".join(y)


import pickle

tfidf = pickle.load(open('models/vectorizer.pkl', 'rb'))
model = pickle.load(open('models/model.pkl', 'rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area(
    label=" ",

    placeholder="Type your SMS or email here...",
    height=120
)

if st.button("Predict"):
    if input_sms.strip() == "":
        st.warning("Please enter a message")
    else:
        transform_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transform_sms])
        result = model.predict(vector_input)[0]

        if result == 1:
            st.error("🚨 Spam")
        else:
            st.success("✅ Not Spam")
