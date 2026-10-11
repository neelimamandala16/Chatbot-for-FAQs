"""A local, retrieval-based FAQ chatbot built with Streamlit and scikit-learn."""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import streamlit as st
from nltk.stem import SnowballStemmer
from nltk.tokenize import TreebankWordTokenizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


FAQ_FILE = Path(__file__).parent / "faqs.json"
FALLBACK_ANSWER = (
    "Sorry, I could not find a suitable answer. "
    "Please try asking another question."
)
TOKENIZER = TreebankWordTokenizer()
STEMMER = SnowballStemmer("english")
GENERIC_QUESTION_WORDS = {
    "available",
    "offer",
    "offered",
    "offering",
    "provide",
    "provided",
}
TOPIC_ALIASES = {
    "admiss": "admission",
    "cours": "course",
    "examin": "exam",
    "fee": "fee",
}


def preprocess_text(text: str) -> List[str]:
    """Normalize question wording so TF-IDF compares its meaningful topics."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = []
    for token in TOKENIZER.tokenize(text):
        if token in ENGLISH_STOP_WORDS or token in GENERIC_QUESTION_WORDS:
            continue
        stem = STEMMER.stem(token)
        tokens.append(TOPIC_ALIASES.get(stem, stem))
    return tokens


@st.cache_data
def load_faqs() -> List[Dict[str, str]]:
    """Load and validate the FAQ records from the project's JSON file."""
    with FAQ_FILE.open("r", encoding="utf-8") as faq_file:
        faqs = json.load(faq_file)

    if not isinstance(faqs, list) or not faqs:
        raise ValueError("The FAQ file must contain a non-empty JSON list.")

    for index, faq in enumerate(faqs, start=1):
        if (
            not isinstance(faq, dict)
            or not isinstance(faq.get("question"), str)
            or not faq["question"].strip()
            or not isinstance(faq.get("answer"), str)
            or not faq["answer"].strip()
        ):
            raise ValueError(
                f"FAQ entry {index} must have non-empty question and answer text."
            )

    return faqs


@st.cache_resource
def create_faq_search(faq_questions: Tuple[str, ...]):
    """Create a TF-IDF matrix once and reuse it across Streamlit reruns."""
    vectorizer = TfidfVectorizer(
        analyzer=preprocess_text,
        lowercase=False,
    )
    question_matrix = vectorizer.fit_transform(faq_questions)
    return vectorizer, question_matrix


def find_answer(
    user_question: str,
    faqs: List[Dict[str, str]],
    vectorizer: TfidfVectorizer,
    question_matrix,
    threshold: float,
) -> str:
    """Return the closest FAQ answer, or the fallback if it misses the threshold."""
    user_vector = vectorizer.transform([user_question])
    scores = cosine_similarity(user_vector, question_matrix).flatten()
    best_index = int(scores.argmax())
    best_score = float(scores[best_index])

    if best_score == 0.0 or best_score < threshold:
        return FALLBACK_ANSWER
    return faqs[best_index]["answer"]


def main() -> None:
    st.set_page_config(page_title="College FAQ Chatbot", page_icon="🎓")
    st.title("🎓 College FAQ Chatbot")
    st.write(
        "Ask a question about admissions, courses, fees, schedules, exams, "
        "or college services. Answers are selected from a local FAQ list."
    )

    threshold = st.sidebar.slider(
        "Similarity threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.05,
        help=(
            "A higher threshold requires a closer match. If the best match is "
            "below this value, the chatbot will ask you to try another question."
        ),
    )

    try:
        faqs = load_faqs()
    except FileNotFoundError:
        st.error(
            f"Could not find {FAQ_FILE.name}. Make sure it is in the same "
            "folder as app.py."
        )
        st.stop()
    except (json.JSONDecodeError, OSError, ValueError) as error:
        st.error(f"Could not load the FAQ data: {error}")
        st.stop()

    try:
        vectorizer, question_matrix = create_faq_search(
            tuple(faq["question"] for faq in faqs)
        )
    except ValueError as error:
        st.error(f"Could not prepare FAQ search: {error}")
        st.stop()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if st.button("Clear chat", help="Remove all messages from this conversation"):
        st.session_state.chat_history = []
        st.rerun()

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    with st.form("question_form", clear_on_submit=True):
        question = st.text_input(
            "Your question",
            placeholder="For example: How do I apply to the college?",
        )
        submitted = st.form_submit_button("Send")

    if submitted:
        if not question.strip():
            st.warning("Please enter a question before sending.")
        else:
            answer = find_answer(
                question,
                faqs,
                vectorizer,
                question_matrix,
                threshold,
            )
            st.session_state.chat_history.extend(
                [
                    {"role": "user", "content": question.strip()},
                    {"role": "assistant", "content": answer},
                ]
            )
            st.rerun()


if __name__ == "__main__":
    main()
