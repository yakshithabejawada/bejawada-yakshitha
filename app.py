import streamlit as st
import joblib
import numpy as np

@st.cache_resource
def load_assets():
    vectorizer = joblib.load('tfidf_vectorizer.joblib')
    model = joblib.load('logreg_sentiment.joblib')
    return vectorizer, model

# Load vectorizer and model
vectorizer, model = load_assets()

st.set_page_config(page_title="Tweet Sentiment Predictor", page_icon="🕊️")
st.title("Tweet Sentiment Predictor")
st.write("Enter a tweet (or multiple tweets separated by `||`) and get sentiment predictions.")

# Single tweet input
user_input = st.text_area("Tweet text", height=120, placeholder="Type a tweet like: I love this product!")

# Option: accept multiple tweets separated by ||
st.caption("Tip: For multiple tweets at once, separate tweets with `||` (e.g. `I love it! || I hate this`)")

if st.button("Predict"):
    if not user_input or user_input.strip() == "":
        st.warning("Please enter some text to predict.")
    else:
        raw_items = [t.strip() for t in user_input.split("||") if t.strip()]
        cleaned = [t.lower() for t in raw_items]  # basic cleaning
        X = vectorizer.transform(cleaned)

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)
            preds = model.predict(X)
            for i, txt in enumerate(raw_items):
                pred_label = "Positive" if preds[i] == 1 else "Negative"
                positive_prob = probs[i][1]
                negative_prob = probs[i][0]
                st.write("---")
                st.markdown(f"**Tweet:** {txt}")
                st.markdown(f"**Prediction:** {pred_label}")
                st.markdown(f"**Confidence:** Positive={positive_prob:.3f} | Negative={negative_prob:.3f}")
        else:
            preds = model.predict(X)
            for i, txt in enumerate(raw_items):
                pred_label = "Positive" if preds[i] == 1 else "Negative"
                st.write("---")
                st.markdown(f"**Tweet:** {txt}")
                st.markdown(f"**Prediction:** {pred_label}")



