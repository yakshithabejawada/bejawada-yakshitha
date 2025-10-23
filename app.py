# app.py
import streamlit as st
import joblib
import numpy as np

@st.cache_resource
def load_assets():
    vect = joblib.load('tfidf_vectorizer.joblib')
    model = joblib.load('logreg_sentiment.joblib')
    return vect, model

vectorizer, model = load_assets()

st.set_page_config(page_title="Tweet Sentiment Predictor", page_icon="🕊️")
st.title("Tweet Sentiment Predictor")
st.write("Enter a tweet (or multiple tweets separated by `||`) and get sentiment predictions.")

# Single tweet input
user_input = st.text_area("Tweet text", height=120, placeholder="Type a tweet like: I love this product!")

# Option: accept multiple tweets separated by || (helpful for batch)
st.caption("Tip: For multiple tweets at once, separate tweets with `||` (e.g. `I love it! || I hate this`)")

if st.button("Predict"):
    if not user_input or user_input.strip() == "":
        st.warning("Please enter some text to predict.")
    else:
        # support multiple tweets separated by "||"
        raw_items = [t.strip() for t in user_input.split("||") if t.strip()]
        cleaned = [t.lower() for t in raw_items]  # same simple cleaning as training
        X = vectorizer.transform(cleaned)

        # If model has predict_proba:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)  # shape (n, 2) -> prob for class 0 and 1
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
            # fallback for models without predict_proba (e.g., LinearSVC)
            preds = model.predict(X)
            for i, txt in enumerate(raw_items):
                pred_label = "Positive" if preds[i] == 1 else "Negative"
                st.write("---")
                st.markdown(f"**Tweet:** {txt}")
                st.markdown(f"**Prediction:** {pred_label}")
