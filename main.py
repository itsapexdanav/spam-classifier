import streamlit as st
import string
from nltk.stem.porter import PorterStemmer
import pickle

# Hide "Ctrl + Enter to apply"
st.markdown("""
<style>
textarea + div {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

# Hardcoded stopwords (deployment-safe)
STOPWORDS = set("""
a about above after again against all am an and any are aren't as at
be because been before being below between both but by
can can't cannot could couldn't did didn't do does doesn't doing don't down during
each few for from further
had hadn't has hasn't have haven't having he he'd he'll he's her here here's hers herself him himself his how how's
i i'd i'll i'm i've if in into is isn't it it's its itself
let's me more most mustn't my myself
no nor not of off on once only or other ought our ours ourselves out over own
same shan't she she'd she'll she's should shouldn't so some such
than that that's the their theirs them themselves then there there's these they they'd they'll they're they've this those through to too
under until up very
was wasn't we we'd we'll we're we've were weren't what what's when when's where where's which while who who's whom why why's with won't would wouldn't
you you'd you'll you're you've your yours yourself yourselves
""".split())

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    words = text.split()

    y = []
    for word in words:
        if word.isalnum() and word not in STOPWORDS:
            y.append(ps.stem(word))

    return " ".join(y)

# Load model & vectorizer
tfidf = pickle.load(open("models/vectorizer.pkl", "rb"))
model = pickle.load(open("models/model.pkl", "rb"))

# UI
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
