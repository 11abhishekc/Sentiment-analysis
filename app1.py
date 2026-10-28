import json
import re
import string
from collections import defaultdict
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
            "Missing model files: "
            + ", ".join(missing)
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
        "logistic_tfidf_model": joblib.load(
            ARTIFACT_DIR / "logistic_tfidf_model.joblib"
        ),
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


# ---------------------------------------------------------------------------
# Negation handling (inference-time heuristic).
# The training labels ignore negation, and the stopword step also drops "not",
# so "I am not good" is seen as just "good". We detect negation on the RAW text
# and flip the emotion's valence when the negated word is the main evidence.
# ---------------------------------------------------------------------------
NEGATORS = {
    "not",
    "no",
    "never",
    "dont",
    "didnt",
    "cant",
    "cannot",
    "wont",
    "wasnt",
    "isnt",
    "arent",
    "doesnt",
    "couldnt",
    "wouldnt",
    "shouldnt",
    "havent",
    "hasnt",
    "hadnt",
    "werent",
    "aint",
}
NEGATION_BREAKS = {"but", "however", "though", "although", "yet", "because"}
NOT_NEGATION = {("not", "only"), ("no", "matter")}
VALENCE_FLIP = {
    "joy": "sadness",
    "love": "sadness",
    "sadness": "joy",
    "anger": "joy",
    "fear": "joy",
    "surprise": "surprise",
}
MIN_EVIDENCE = (
    1.0  # minimum logistic-regression weight for a word to count as emotion evidence
)


def find_negated_words(text, artifacts):
    """Return [(word, emotion_it_points_to)] for words within 3 tokens after a negator."""
    vocab = artifacts["tfidf_vectorizer"].vocabulary_
    lr = artifacts["logistic_tfidf_model"]
    labels = artifacts["labels"]
    tokens = re.findall(r"[a-z]+", text.lower().replace("\u2019", "'").replace("'", ""))
    found = []
    for i, tok in enumerate(tokens):
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        if tok not in NEGATORS or (tok, nxt) in NOT_NEGATION:
            continue
        for word in tokens[i + 1 : i + 4]:
            if word in NEGATION_BREAKS:
                break
            if word in vocab:
                weights = lr.coef_[:, vocab[word]]
                best = int(weights.argmax())
                if weights[best] >= MIN_EVIDENCE:
                    found.append((word, labels[int(lr.classes_[best])]))
    return found


def strongest_evidence_word(cleaned, emotion, artifacts):
    """The word that pushes the logistic model hardest towards `emotion`."""
    vectorizer = artifacts["tfidf_vectorizer"]
    lr = artifacts["logistic_tfidf_model"]
    labels = artifacts["labels"]
    class_pos = [labels[int(c)] for c in lr.classes_].index(emotion)
    features = vectorizer.transform([cleaned])
    inverse_vocab = {i: w for w, i in vectorizer.vocabulary_.items()}
    scores = {
        inverse_vocab[i]: lr.coef_[class_pos, i] * features[0, i]
        for i in features.nonzero()[1]
    }
    return max(scores, key=scores.get) if scores else None


def apply_negation(distribution, sentence, cleaned, artifacts):
    """Flip valence if the top emotion's strongest evidence word is negated."""
    top = max(distribution, key=distribution.get)
    key_word = strongest_evidence_word(cleaned, top, artifacts)
    negated = [
        w
        for w, emo in find_negated_words(sentence, artifacts)
        if emo == top and w == key_word
    ]
    if not negated:
        return distribution, None
    flipped = defaultdict(float)
    for emotion, prob in distribution.items():
        flipped[VALENCE_FLIP.get(emotion, emotion)] += prob
    return dict(flipped), key_word


st.set_page_config(page_title="Emotion Classifier", page_icon="💬", layout="centered")
st.title("💬 Emotion Classifier")
st.write(
    "Enter a sentence and predict its emotion using a model trained in your notebook."
)

try:
    artifacts = load_artifacts()
except Exception as exc:
    st.error(str(exc))
    st.info(
        "In your notebook, run the export code provided with this app, then start Streamlit from the same project folder."
    )
    st.stop()

model_choice = st.selectbox(
    "Choose a model",
    [
        "Logistic Regression (TF-IDF)",
        "Naive Bayes (TF-IDF)",
        "Naive Bayes (Bag of Words)",
    ],
)
handle_negation = st.checkbox(
    "Handle negation (heuristic)",
    value=True,
    help="Detects words like 'not' / \"don't\" before the key emotion word and flips the emotion's valence.",
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
            st.warning(
                "The sentence became empty after preprocessing. Try a longer sentence with more descriptive words."
            )
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
            probabilities = model.predict_proba(features)[0]
            distribution = {
                artifacts["labels"].get(int(c), str(c)): float(p)
                for c, p in zip(model.classes_, probabilities)
            }

            negated_word = None
            if handle_negation:
                distribution, negated_word = apply_negation(
                    distribution, sentence, cleaned, artifacts
                )

            emotion = max(distribution, key=distribution.get)
            st.success(f"Predicted emotion: **{emotion}**")
            if negated_word:
                st.info(
                    f'Negation detected before "{negated_word}", so the emotion was flipped. '
                    "This is a rule-based adjustment on top of the model."
                )
            with st.expander("See preprocessing details"):
                st.write("**Original sentence:**", sentence)
                st.write("**Processed sentence:**", cleaned)

            ranked = sorted(
                distribution.items(), key=lambda item: item[1], reverse=True
            )
            st.subheader("Model confidence scores")
            for label, probability in ranked[:5]:
                st.write(f"{label}: {probability:.1%}")
                st.progress(min(max(probability, 0.0), 1.0))
            st.caption(
                "These are model probability scores, not guaranteed measures of real-world certainty."
            )
