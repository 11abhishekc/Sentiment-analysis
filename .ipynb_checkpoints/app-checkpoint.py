import json
import re
import string
from pathlib import Path

import joblib
import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

ARTIFACT_DIR = Path("model_artifacts")

@st.cache_resource
def load_artifacts():
    required = [
        "bow_vectorizer.joblib",
        "nb_bow_model.joblib",
        "tfidf_vectorizer.joblib",
        "nb_tfidf_model.joblib",
        "logistic_tfidf_model.joblib",
        "emotion_labels.json",
    ]
    missing = [name for name in required if not (ARTIFACT_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing model files: " + ", ".join(missing)
            + ". Run the export cell in your notebook first."
        )

    with open(ARTIFACT_DIR / "emotion_labels.json", encoding="utf-8") as f:
        labels = json.load(f)

    return {
        "labels": {int(k): v for k, v in labels.items()},
        "bow_vectorizer": joblib.load(ARTIFACT_DIR / "bow_vectorizer.joblib"),
        "nb_bow_model": joblib.load(ARTIFACT_DIR / "nb_bow_model.joblib"),
        "tfidf_vectorizer": joblib.load(ARTIFACT_DIR / "tfidf_vectorizer.joblib"),
        "nb_tfidf_model": joblib.load(ARTIFACT_DIR / "nb_tfidf_model.joblib"),
        "logistic_tfidf_model": joblib.load(ARTIFACT_DIR / "logistic_tfidf_model.joblib"),
    }

@st.cache_resource
def get_stop_words():
    try:
        return set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        return set(stopwords.words("english"))

def clean_text(text):
    # Mirrors the preprocessing steps in the notebook.
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = "".join(ch for ch in text if not ch.isdigit())
    text = "".join(ch for ch in text if ch.isascii())
    try:
        tokens = word_tokenize(text)
    except LookupError:
        nltk.download("punkt", quiet=True)
        try:
            tokens = word_tokenize(text)
        except LookupError:
            nltk.download("punkt_tab", quiet=True)
            tokens = word_tokenize(text)

    stop_words = get_stop_words()
    return " ".join(token for token in tokens if token not in stop_words)

st.set_page_config(page_title="Emotion Classifier", page_icon="💬", layout="centered")
st.title("💬 Emotion Classifier")
st.write("Enter a sentence and predict its emotion using a model trained in your notebook.")

try:
    artifacts = load_artifacts()
except Exception as exc:
    st.error(str(exc))
    st.info("In your notebook, run the export code provided with this app, then start Streamlit from the same project folder.")
    st.stop()

model_choice = st.selectbox(
    "Choose a model",
    [
        "Logistic Regression (TF-IDF)",
        "Naive Bayes (TF-IDF)",
        "Naive Bayes (Bag of Words)",
    ],
)
sentence = st.text_area(
    "Your sentence",
    placeholder="Example: I am feeling wonderful today!",
    height=120,
)

if st.button("Predict emotion", type="primary", use_container_width=True):
    if not sentence.strip():
        st.warning("Please enter a sentence first.")
    else:
        cleaned = clean_text(sentence)
        if not cleaned.strip():
            st.warning("The sentence became empty after preprocessing. Try a longer sentence with more descriptive words.")
        else:
            if model_choice == "Logistic Regression (TF-IDF)":
                vectorizer = artifacts["tfidf_vectorizer"]
                model = artifacts["logistic_tfidf_model"]
            elif model_choice == "Naive Bayes (TF-IDF)":
                vectorizer = artifacts["tfidf_vectorizer"]
                model = artifacts["nb_tfidf_model"]
            else:
                vectorizer = artifacts["bow_vectorizer"]
                model = artifacts["nb_bow_model"]

            features = vectorizer.transform([cleaned])
            prediction = int(model.predict(features)[0])
            emotion = artifacts["labels"].get(prediction, f"Class {prediction}")

            st.success(f"Predicted emotion: **{emotion}**")
            with st.expander("See preprocessing details"):
                st.write("**Original sentence:**", sentence)
                st.write("**Processed sentence:**", cleaned)
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(features)[0]
                class_ids = model.classes_
                ranked = sorted(
                    [(artifacts["labels"].get(int(class_id), str(class_id)), float(prob))
                     for class_id, prob in zip(class_ids, probabilities)],
                    key=lambda item: item[1],
                    reverse=True,
                )
                st.subheader("Model confidence scores")
                for label, probability in ranked[:5]:
                    st.write(f"{label}: {probability:.1%}")
                    st.progress(min(max(probability, 0.0), 1.0))
                st.caption("These are model probability scores, not guaranteed measures of real-world certainty.")
