"""
╔══════════════════════════════════════════════════════════════╗
║         FAQ CHATBOT - CodeAlpha Internship Project           ║
║         Built with NLP, TF-IDF & Cosine Similarity          ║
╚══════════════════════════════════════════════════════════════╝

Author      : Subhajit Roy
Project     : FAQ Chatbot
Technology  : Python, Streamlit, NLTK, Scikit-learn
Description : An intelligent FAQ Chatbot that matches user queries
              to the most relevant question in a FAQ database using
              TF-IDF Vectorization and Cosine Similarity.
Copyright   : © 2026 Subhajit Roy. All Rights Reserved.
"""

# ──────────────────────────────────────────────────────────────
# Standard Library Imports
# ──────────────────────────────────────────────────────────────
import html
import json
import os
import re
from datetime import datetime

# ──────────────────────────────────────────────────────────────
# Third-Party Imports
# ──────────────────────────────────────────────────────────────
import nltk
import numpy as np
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ──────────────────────────────────────────────────────────────
# SSL Certificate Fix (macOS)
# On macOS, Python's bundled SSL certs are often missing, causing
# NLTK downloads to fail with CERTIFICATE_VERIFY_FAILED. This
# bypass is the standard fix recommended by the Python docs and
# is safe for downloading public NLTK corpus data from nltk.org.
# ──────────────────────────────────────────────────────────────
import ssl as _ssl
try:
    _create_unverified_ctx = _ssl._create_unverified_context
except AttributeError:
    pass
else:
    _ssl._create_default_https_context = _create_unverified_ctx

# ──────────────────────────────────────────────────────────────
# NLTK Resource Download — runs at MODULE LEVEL on every start
# This guarantees corpora exist before any class is instantiated,
# regardless of Streamlit's @st.cache_resource execution order.
# ──────────────────────────────────────────────────────────────
for _resource in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
    nltk.download(_resource, quiet=True)


# ──────────────────────────────────────────────────────────────
# Constants & Configuration
# ──────────────────────────────────────────────────────────────
FAQ_FILE_PATH         = "faq_data.json"
CONFIDENCE_THRESHOLD  = 0.15   # Minimum similarity score to return an answer
APP_TITLE             = "College FAQ Chatbot"
APP_ICON              = "🎓"
BOT_AVATAR            = "🤖"
USER_AVATAR           = "👤"
LOW_CONFIDENCE_MSG    = (
    "I'm sorry, I couldn't find a relevant answer to your question. "
    "Please try rephrasing your question or contact the admissions office directly."
)


# ══════════════════════════════════════════════════════════════
#  DATA LOADING LAYER
# ══════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_faq_data(filepath: str) -> list[dict]:
    """
    Load and validate the FAQ dataset from a JSON file.

    Args:
        filepath (str): Relative or absolute path to the JSON file.

    Returns:
        list[dict]: A list of FAQ dictionaries with 'question' and 'answer' keys.

    Raises:
        FileNotFoundError: If the FAQ file does not exist.
        ValueError: If the JSON is malformed or fails schema validation.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"FAQ data file not found at '{filepath}'. "
            "Please ensure 'faq_data.json' is in the project root directory."
        )

    try:
        with open(filepath, "r", encoding="utf-8") as faq_file:
            data = json.load(faq_file)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"The FAQ file '{filepath}' contains invalid JSON. "
            f"Please verify the file format.\nDetails: {e}"
        )

    # Validate that it's a non-empty list
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError(
            "The FAQ file must contain a non-empty JSON array of objects."
        )

    # Validate each entry has 'question' and 'answer' keys
    validated_data = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"FAQ entry at index {idx} is not a valid object.")
        if "question" not in item or "answer" not in item:
            raise ValueError(
                f"FAQ entry at index {idx} is missing 'question' or 'answer' key."
            )
        if not str(item["question"]).strip() or not str(item["answer"]).strip():
            continue  # Skip blank entries silently
        validated_data.append(
            {
                "question": str(item["question"]).strip(),
                "answer":   str(item["answer"]).strip(),
            }
        )

    if len(validated_data) == 0:
        raise ValueError("The FAQ file contains no valid question-answer pairs.")

    return validated_data


# ══════════════════════════════════════════════════════════════
#  NLP PREPROCESSING LAYER
# ══════════════════════════════════════════════════════════════

class TextPreprocessor:
    """
    Handles all NLP text preprocessing steps:
      1. Lowercasing
      2. Punctuation & special character removal
      3. Tokenization
      4. Stopword removal
      5. Lemmatization
    """

    def __init__(self) -> None:
        self.lemmatizer  = WordNetLemmatizer()
        self.stop_words  = set(stopwords.words("english"))
        # Keep question words — they can be meaningful in FAQ matching
        self.stop_words -= {"what", "when", "where", "who", "how", "why", "which"}

    def clean_text(self, text: str) -> str:
        """
        Full preprocessing pipeline for a given text string.

        Steps:
            1. Lowercase the text.
            2. Remove URLs, email addresses, and special characters.
            3. Tokenize into individual words.
            4. Remove stopwords.
            5. Lemmatize each remaining token.
            6. Rejoin tokens into a single clean string.

        Args:
            text (str): Raw input text.

        Returns:
            str: Preprocessed, cleaned text string.
        """
        if not text or not isinstance(text, str):
            return ""

        # Step 1: Lowercase
        text = text.lower()

        # Step 2: Remove URLs and email addresses
        text = re.sub(r"http\S+|www\.\S+", " ", text)
        text = re.sub(r"\S+@\S+", " ", text)

        # Step 3: Remove special characters and digits, keep alphabets and spaces
        text = re.sub(r"[^a-z\s]", " ", text)

        # Step 4: Collapse multiple whitespace into single space
        text = re.sub(r"\s+", " ", text).strip()

        # Step 5: Tokenize
        try:
            tokens = word_tokenize(text)
        except Exception:
            tokens = text.split()  # Fallback to simple split

        # Step 6: Remove stopwords and short tokens
        tokens = [t for t in tokens if t not in self.stop_words and len(t) > 1]

        # Step 7: Lemmatize
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens]

        return " ".join(tokens)


# ══════════════════════════════════════════════════════════════
#  TF-IDF VECTORIZATION & SIMILARITY ENGINE
# ══════════════════════════════════════════════════════════════

class FAQMatcher:
    """
    Core matching engine that uses TF-IDF vectorization and
    cosine similarity to find the most relevant FAQ answer
    for a given user query.
    """

    def __init__(self, faq_data: list[dict]) -> None:
        """
        Initialize the FAQ matcher by preprocessing all FAQ questions
        and fitting the TF-IDF vectorizer.

        Args:
            faq_data (list[dict]): List of FAQ dictionaries.
        """
        self.faq_data     = faq_data
        self.preprocessor = TextPreprocessor()

        # Preprocess all FAQ questions
        self.processed_questions = [
            self.preprocessor.clean_text(item["question"])
            for item in faq_data
        ]

        # Initialize and fit TF-IDF Vectorizer on the FAQ question corpus
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),       # Unigrams and bigrams for better matching
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,        # Apply sublinear TF scaling
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.processed_questions)

    def find_best_match(self, user_query: str) -> dict:
        """
        Find the most relevant FAQ entry for the given user query.

        Process:
            1. Preprocess the user query.
            2. Transform it using the fitted TF-IDF vectorizer.
            3. Compute cosine similarity with all FAQ questions.
            4. Return the best matching FAQ with its confidence score.

        Args:
            user_query (str): The raw user question.

        Returns:
            dict: A dictionary containing:
                - 'answer'       (str)  : The best matching answer.
                - 'matched_question' (str): The original matched FAQ question.
                - 'confidence'   (float): Cosine similarity score (0.0–1.0).
                - 'found'        (bool) : Whether confidence > threshold.
        """
        result = {
            "answer":           LOW_CONFIDENCE_MSG,
            "matched_question": None,
            "confidence":       0.0,
            "found":            False,
        }

        # Guard: empty or whitespace-only query
        if not user_query or not user_query.strip():
            result["answer"] = "Please enter a valid question."
            return result

        try:
            # Step 1: Preprocess user query
            cleaned_query = self.preprocessor.clean_text(user_query)

            if not cleaned_query.strip():
                return result  # Nothing left after cleaning

            # Step 2: Vectorize user query using the fitted TF-IDF model
            query_vector = self.vectorizer.transform([cleaned_query])

            # Step 3: Compute cosine similarity between query and all FAQ questions
            similarity_scores = cosine_similarity(query_vector, self.tfidf_matrix)

            # Step 4: Flatten and find the best match index
            scores_array = similarity_scores.flatten()
            best_idx     = int(np.argmax(scores_array))
            best_score   = float(scores_array[best_idx])

            # Step 5: Apply confidence threshold
            if best_score >= CONFIDENCE_THRESHOLD:
                result["answer"]           = self.faq_data[best_idx]["answer"]
                result["matched_question"] = self.faq_data[best_idx]["question"]
                result["confidence"]       = round(best_score, 4)
                result["found"]            = True

        except Exception as e:
            # Graceful fallback for unexpected runtime errors
            result["answer"] = (
                "An unexpected error occurred while processing your query. "
                "Please try again. If the issue persists, contact support."
            )

        return result

    def get_statistics(self) -> dict:
        """
        Return statistical summary of the loaded FAQ database.

        Returns:
            dict: Keys include total FAQs, vocabulary size, and topic categories.
        """
        vocab_size    = len(self.vectorizer.vocabulary_)
        total_faqs    = len(self.faq_data)
        avg_q_length  = (
            sum(len(q.split()) for q in self.processed_questions) / total_faqs
            if total_faqs > 0 else 0
        )
        return {
            "total_faqs":    total_faqs,
            "vocabulary":    vocab_size,
            "avg_q_length":  round(avg_q_length, 1),
        }


# ══════════════════════════════════════════════════════════════
#  STREAMLIT SESSION STATE MANAGEMENT
# ══════════════════════════════════════════════════════════════

def initialize_session_state() -> None:
    """
    Initialize all required Streamlit session state variables.
    This function is idempotent — safe to call multiple times.
    """
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "total_queries" not in st.session_state:
        st.session_state.total_queries = 0

    if "successful_matches" not in st.session_state:
        st.session_state.successful_matches = 0


def add_message(role: str, content: str, metadata: dict | None = None) -> None:
    """
    Append a message to the chat history.

    Args:
        role     (str)  : Either 'user' or 'assistant'.
        content  (str)  : The message text.
        metadata (dict) : Optional dict with extra info (confidence, matched_question).
    """
    st.session_state.chat_history.append(
        {
            "role":      role,
            "content":   content,
            "metadata":  metadata or {},
            "timestamp": datetime.now().strftime("%I:%M %p"),
        }
    )


def clear_chat() -> None:
    """Clear the entire chat history and reset counters."""
    st.session_state.chat_history    = []
    st.session_state.total_queries   = 0
    st.session_state.successful_matches = 0


# ══════════════════════════════════════════════════════════════
#  UI RENDERING LAYER
# ══════════════════════════════════════════════════════════════

def render_custom_css() -> None:
    """
    Inject custom CSS for a professional dark-themed chatbot UI.
    Implements chat bubbles, typography, animations, and responsive design.
    """
    st.markdown(
        """
        <style>
        /* ── Google Font ── */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        /* ── Global Reset ── */
        * { box-sizing: border-box; margin: 0; padding: 0; }

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif !important;
        }

        /* ── App Background ── */
        .stApp {
            background: linear-gradient(135deg, #0f0c29, #1a1a3e, #141428) !important;
            min-height: 100vh;
        }

        /* ── Streamlit header: transparent + no black bar ── */
        [data-testid="stHeader"] {
            background-color: transparent !important;
            background: transparent !important;
            border-bottom: none !important;
            box-shadow: none !important;
        }
        /* Hide only the hamburger menu & deploy button inside the header */
        #MainMenu                        { display: none !important; }
        [data-testid="stToolbar"]        { display: none !important; }
        [data-testid="stDecoration"]     { display: none !important; }
        footer                           { display: none !important; }

        /* Force the sidebar toggle arrow to always be visible */
        [data-testid="stSidebarCollapsedControl"] {
            visibility: visible !important;
            display: flex !important;
            opacity: 1 !important;
            z-index: 999999 !important;
        }
        section[data-testid="stSidebarCollapsedControl"] button {
            background: rgba(108,99,255,0.25) !important;
            border: 1px solid rgba(108,99,255,0.5) !important;
            border-radius: 8px !important;
            color: #fff !important;
        }
        .block-container { padding-top: 0.5rem !important; padding-bottom: 2rem !important; }


        /* ── App Header ── */
        .app-header {
            background: linear-gradient(90deg, #6c63ff 0%, #e040fb 100%);
            border-radius: 16px;
            padding: 20px 28px;
            margin-bottom: 24px;
            display: flex;
            align-items: center;
            gap: 16px;
            box-shadow: 0 8px 32px rgba(108, 99, 255, 0.35);
        }
        .app-header h1 {
            color: #ffffff;
            font-size: 1.7rem;
            font-weight: 700;
            letter-spacing: -0.5px;
            line-height: 1.2;
        }
        .app-header p {
            color: rgba(255,255,255,0.82);
            font-size: 0.85rem;
            margin-top: 4px;
        }
        .header-icon { font-size: 2.5rem; }

        /* ── Chat Container ── */
        .chat-container {
            display: flex;
            flex-direction: column;
            gap: 14px;
            padding: 8px 0;
            max-height: 62vh;
            overflow-y: auto;
            scrollbar-width: thin;
            scrollbar-color: #6c63ff transparent;
        }
        .chat-container::-webkit-scrollbar { width: 5px; }
        .chat-container::-webkit-scrollbar-thumb {
            background: #6c63ff;
            border-radius: 10px;
        }

        /* ── Chat Bubbles ── */
        .chat-message {
            display: flex;
            align-items: flex-start;
            gap: 10px;
            animation: fadeSlideIn 0.3s ease-out;
        }
        @keyframes fadeSlideIn {
            from { opacity: 0; transform: translateY(10px); }
            to   { opacity: 1; transform: translateY(0); }
        }

        /* User bubble (right-aligned) */
        .chat-message.user { flex-direction: row-reverse; }
        .chat-message.user .bubble {
            background: linear-gradient(135deg, #6c63ff, #8b5cf6);
            color: #ffffff;
            border-radius: 18px 4px 18px 18px;
            box-shadow: 0 4px 16px rgba(108, 99, 255, 0.3);
        }

        /* Bot bubble (left-aligned) */
        .chat-message.bot .bubble {
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.12);
            color: #e2e8f0;
            border-radius: 4px 18px 18px 18px;
            backdrop-filter: blur(8px);
        }

        .bubble {
            max-width: 75%;
            padding: 12px 16px;
            font-size: 0.9rem;
            line-height: 1.6;
            position: relative;
        }
        .bubble .timestamp {
            font-size: 0.68rem;
            opacity: 0.55;
            margin-top: 6px;
            display: block;
        }

        /* ── Avatar ── */
        .avatar {
            width: 36px;
            height: 36px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1rem;
            flex-shrink: 0;
        }
        .avatar.user-av { background: linear-gradient(135deg, #6c63ff, #8b5cf6); }
        .avatar.bot-av  { background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2); }

        /* ── Confidence Badge ── */
        .confidence-badge {
            display: inline-flex;
            align-items: center;
            gap: 4px;
            font-size: 0.7rem;
            font-weight: 600;
            padding: 2px 8px;
            border-radius: 999px;
            margin-top: 8px;
        }
        .confidence-high   { background: rgba(52, 211, 153, 0.15); color: #34d399; }
        .confidence-medium { background: rgba(251, 191, 36, 0.15);  color: #fbbf24; }
        .confidence-low    { background: rgba(248, 113, 113, 0.15); color: #f87171; }

        /* ── Matched Question ── */
        .matched-q {
            font-size: 0.72rem;
            color: rgba(255,255,255,0.4);
            font-style: italic;
            margin-top: 4px;
            padding-top: 6px;
            border-top: 1px solid rgba(255,255,255,0.08);
        }

        /* ── Input Area ── */
        .stTextInput > div > div > input {
            background: rgba(255,255,255,0.06) !important;
            border: 1px solid rgba(108, 99, 255, 0.5) !important;
            border-radius: 12px !important;
            color: #e2e8f0 !important;
            font-size: 0.9rem !important;
            padding: 12px 16px !important;
            transition: border-color 0.2s;
        }
        .stTextInput > div > div > input:focus {
            border-color: #6c63ff !important;
            box-shadow: 0 0 0 2px rgba(108, 99, 255, 0.25) !important;
        }
        .stTextInput > div > div > input::placeholder { color: rgba(255,255,255,0.3) !important; }

        /* ── Buttons ── */
        .stButton > button {
            background: linear-gradient(135deg, #6c63ff, #8b5cf6) !important;
            color: #fff !important;
            border: none !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
            font-size: 0.85rem !important;
            padding: 10px 20px !important;
            transition: all 0.25s ease !important;
            box-shadow: 0 4px 12px rgba(108, 99, 255, 0.3) !important;
        }
        .stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 20px rgba(108, 99, 255, 0.45) !important;
        }
        .stButton > button:active { transform: translateY(0) !important; }

        /* ── Sidebar ── */
        [data-testid="stSidebar"] {
            background: rgba(255,255,255,0.03) !important;
            border-right: 1px solid rgba(255,255,255,0.08) !important;
        }
        [data-testid="stSidebar"] * { color: #e2e8f0 !important; }

        /* ── Metric Cards ── */
        [data-testid="metric-container"] {
            background: rgba(108, 99, 255, 0.12) !important;
            border: 1px solid rgba(108, 99, 255, 0.25) !important;
            border-radius: 12px !important;
            padding: 14px !important;
        }
        [data-testid="metric-container"] label {
            color: rgba(255,255,255,0.55) !important;
            font-size: 0.75rem !important;
        }
        [data-testid="metric-container"] [data-testid="stMetricValue"] {
            color: #ffffff !important;
            font-weight: 700 !important;
        }

        /* ── Divider ── */
        hr { border-color: rgba(255,255,255,0.08) !important; }

        /* ── Empty chat notice ── */
        .empty-chat {
            text-align: center;
            padding: 50px 20px;
            color: rgba(255,255,255,0.25);
        }
        .empty-chat .big-icon { font-size: 3.5rem; margin-bottom: 12px; }
        .empty-chat p { font-size: 0.9rem; line-height: 1.6; }

        /* ── Sample question chips ── */
        .chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 8px;
        }
        .chip {
            background: rgba(108, 99, 255, 0.15);
            border: 1px solid rgba(108, 99, 255, 0.35);
            border-radius: 999px;
            padding: 5px 14px;
            font-size: 0.78rem;
            color: #a5b4fc;
            cursor: default;
        }

        /* ── Info banner ── */
        .info-banner {
            background: rgba(108, 99, 255, 0.1);
            border: 1px solid rgba(108, 99, 255, 0.3);
            border-radius: 10px;
            padding: 10px 16px;
            font-size: 0.8rem;
            color: rgba(255,255,255,0.65);
            margin-bottom: 12px;
        }

        /* ── Scrollbar for stApp ── */
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: rgba(108,99,255,0.4); border-radius: 10px; }

        /* ── Footer ── */
        .app-footer {
            text-align: center;
            padding: 18px 0 8px;
            margin-top: 24px;
            border-top: 1px solid rgba(255,255,255,0.07);
            color: rgba(255,255,255,0.35);
            font-size: 0.78rem;
            letter-spacing: 0.3px;
        }
        .app-footer .author {
            color: rgba(165, 180, 252, 0.8);
            font-weight: 600;
        }
        .app-footer .cursor {
            display: inline-block;
            width: 2px;
            height: 0.85em;
            background: #a5b4fc;
            margin-left: 2px;
            border-radius: 1px;
            vertical-align: middle;
            animation: blink 1s step-end infinite;
        }
        @keyframes blink {
            0%, 100% { opacity: 1; }
            50%       { opacity: 0; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    """Render the top application header with title and subtitle."""
    st.markdown(
        """
        <div class="app-header">
            <div class="header-icon">🎓</div>
            <div>
                <h1>College Admissions FAQ Chatbot</h1>
                <p>Powered by NLP · TF-IDF · Cosine Similarity &nbsp;|&nbsp; CodeAlpha Internship Project</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_confidence_class(score: float) -> str:
    """Return the CSS class for the confidence badge based on the score."""
    if score >= 0.45:
        return "confidence-high"
    elif score >= 0.25:
        return "confidence-medium"
    else:
        return "confidence-low"


def get_confidence_label(score: float) -> str:
    """Return a human-readable label for the confidence score."""
    if score >= 0.45:
        return "High Confidence"
    elif score >= 0.25:
        return "Medium Confidence"
    else:
        return "Low Confidence"


def render_chat_message(message: dict) -> None:
    """
    Render a single chat message as a styled HTML bubble.

    Args:
        message (dict): A message dict with keys: role, content, metadata, timestamp.

    Security:
        User-supplied content is HTML-escaped before injection to prevent
        any risk of HTML/script injection through the chat input.
    """
    role      = message["role"]
    content   = message["content"]
    timestamp = message.get("timestamp", "")
    is_user   = (role == "user")

    css_role   = "user" if is_user else "bot"
    avatar_css = "user-av" if is_user else "bot-av"
    avatar_ico = USER_AVATAR if is_user else BOT_AVATAR

    # Escape user content to prevent HTML injection; bot answers are trusted strings
    safe_content = html.escape(content) if is_user else content

    st.markdown(
        f"""
        <div class="chat-message {css_role}">
            <div class="avatar {avatar_css}">{avatar_ico}</div>
            <div class="bubble">
                {safe_content}
                <span class="timestamp">{timestamp}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(matcher: FAQMatcher) -> None:
    """
    Render the sidebar with chatbot info, FAQ statistics, and session metrics.

    Args:
        matcher (FAQMatcher): The FAQ matcher instance for statistics.
    """
    with st.sidebar:
        # Logo / Branding
        st.markdown(
            """
            <div style="text-align:center; padding:10px 0 20px;">
                <div style="font-size:3rem;">🤖</div>
                <h2 style="font-weight:700; font-size:1.1rem; margin:6px 0 2px;">FAQ Assistant</h2>
                <p style="font-size:0.75rem; color:rgba(255,255,255,0.45);">CodeAlpha Internship</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()

        # FAQ Statistics
        st.markdown("#### 📊 FAQ Database Stats")
        stats = matcher.get_statistics()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total FAQs",  stats["total_faqs"])
        with col2:
            st.metric("Vocab Size",  stats["vocabulary"])
        st.metric("Avg. Q Length (tokens)", stats["avg_q_length"])
        st.divider()

        # Session Metrics
        st.markdown("#### 💬 Session Stats")
        total   = st.session_state.get("total_queries", 0)
        success = st.session_state.get("successful_matches", 0)
        rate    = f"{round((success / total) * 100)}%" if total > 0 else "N/A"
        col3, col4 = st.columns(2)
        with col3:
            st.metric("Queries",  total)
        with col4:
            st.metric("Matched",  success)
        st.metric("Match Rate", rate)
        st.divider()

        # Topic coverage
        st.markdown("#### 🎯 Topic Coverage")
        topics = [
            "🏫 Admissions", "📋 Eligibility",
            "💰 Fees",        "🎓 Scholarships",
            "🏠 Hostel",      "💼 Placements",
            "📄 Documents",   "📚 Courses",
            "📅 Attendance",  "📝 Examinations",
        ]
        for topic in topics:
            st.markdown(
                f"<span style='font-size:0.8rem;color:rgba(255,255,255,0.7);'>{topic}</span>",
                unsafe_allow_html=True,
            )
        st.divider()

        # About
        st.markdown(
            """
            <div style="font-size:0.75rem; color:rgba(255,255,255,0.4); line-height:1.7;">
            <b>Technology Stack</b><br>
            🐍 Python 3.11+<br>
            🌐 Streamlit<br>
            🔤 NLTK (NLP)<br>
            📐 Scikit-learn (TF-IDF)<br>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_welcome_message() -> None:
    """Render the initial empty-chat placeholder with sample questions."""
    st.markdown(
        """
        <div class="empty-chat">
            <div class="big-icon">💬</div>
            <p>
                Hi! I'm your College Admissions FAQ Assistant.<br>
                Ask me anything about <strong>admissions, fees, scholarships,
                hostel, placements,</strong> and more!
            </p>
        </div>
        <div class="chip-row" style="justify-content:center; margin-top:0; padding:0 20px 16px;">
            <span class="chip">What is the last date to apply?</span>
            <span class="chip">What documents are required?</span>
            <span class="chip">Are scholarships available?</span>
            <span class="chip">Tell me about hostel facilities</span>
            <span class="chip">What is the placement record?</span>
            <span class="chip">What courses are offered?</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════
#  CORE APPLICATION ENTRY POINT
# ══════════════════════════════════════════════════════════════

def main() -> None:
    """
    Main application entry point.

    Execution flow:
        1. Configure the Streamlit page.
        2. Download NLTK resources.
        3. Load & validate FAQ data.
        4. Initialize the FAQMatcher.
        5. Initialize session state.
        6. Render sidebar, header, and chat history.
        7. Handle user input and generate bot response.
    """

    # ── Page Configuration ──────────────────────────────────
    st.set_page_config(
        page_title = f"{APP_TITLE} | CodeAlpha",
        page_icon  = APP_ICON,
        layout     = "wide",
        initial_sidebar_state = "expanded",
    )

    # ── Inject Custom CSS ───────────────────────────────────
    render_custom_css()

    # ── Force Sidebar Open on Every Load (JS) ───────────────
    # Streamlit saves sidebar state in localStorage. This script
    # clears it on load so the sidebar is always open at start.
    st.markdown(
        """
        <script>
        (function() {
            // Clear Streamlit's saved sidebar collapsed state
            const keys = Object.keys(localStorage);
            keys.forEach(k => {
                if (k.includes('sidebar') || k.includes('Sidebar')) {
                    localStorage.removeItem(k);
                }
            });

            // Also programmatically click toggle if sidebar is collapsed
            function openSidebar() {
                const collapsed = document.querySelector(
                    '[data-testid="stSidebarCollapsedControl"] button'
                );
                if (collapsed) {
                    collapsed.click();
                }
            }
            // Try after short delays to ensure DOM is ready
            setTimeout(openSidebar, 300);
            setTimeout(openSidebar, 800);
        })();
        </script>
        """,
        unsafe_allow_html=True,
    )

    # ── Load FAQ Data ───────────────────────────────────────
    try:
        faq_data = load_faq_data(FAQ_FILE_PATH)
    except FileNotFoundError as e:
        st.error(f"⚠️ **File Not Found:** {e}")
        st.info("Please place `faq_data.json` in the same directory as `app.py` and restart.")
        st.stop()
    except ValueError as e:
        st.error(f"⚠️ **Data Error:** {e}")
        st.info("Please verify the format of `faq_data.json` and restart.")
        st.stop()

    # ── Initialize FAQ Matcher ──────────────────────────────
    # Use st.cache_resource so the TF-IDF model is built only once per session
    @st.cache_resource(show_spinner="🔧 Initializing NLP engine...")
    def get_matcher(data_tuple):
        # Convert back from tuple-of-tuples to list-of-dicts for caching compatibility
        data = [{"question": q, "answer": a} for q, a in data_tuple]
        return FAQMatcher(data)

    # Serialize FAQ data for caching (Streamlit can't hash dicts directly)
    data_tuple = tuple((item["question"], item["answer"]) for item in faq_data)
    matcher    = get_matcher(data_tuple)

    # ── Session State ───────────────────────────────────────
    initialize_session_state()

    # ── Sidebar ─────────────────────────────────────────────
    render_sidebar(matcher)

    # ── Header ──────────────────────────────────────────────
    render_header()

    # ── Greeting on first load ──────────────────────────────
    if len(st.session_state.chat_history) == 0:
        render_welcome_message()
    else:
        # Render existing chat history
        for msg in st.session_state.chat_history:
            render_chat_message(msg)

    st.markdown("")  # Spacer

    # ── Info Banner ─────────────────────────────────────────
    st.markdown(
        '<div class="info-banner">'
        "💡 <strong>Tip:</strong> Ask questions in plain English. "
        "The chatbot uses NLP to understand the intent and match the best answer."
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Input Row ───────────────────────────────────────────
    col_input, col_btn, col_clear = st.columns([7, 1.2, 1.2])

    with col_input:
        user_input = st.text_input(
            label        = "Your question",
            placeholder  = "e.g. What documents do I need for admission?",
            label_visibility = "collapsed",
            key          = "user_query_input",
        )

    with col_btn:
        send_clicked = st.button("Send ➤", use_container_width=True, key="send_btn")

    with col_clear:
        clear_clicked = st.button("🗑 Clear", use_container_width=True, key="clear_btn")

    # ── Handle Clear ────────────────────────────────────────
    if clear_clicked:
        clear_chat()
        st.rerun()

    # ── Handle Send ─────────────────────────────────────────
    if send_clicked:
        query = user_input.strip() if user_input else ""

        if query:
            # Update session counters
            st.session_state.total_queries += 1

            # Add user message to history
            add_message("user", query)

            # Query the FAQ matcher
            result = matcher.find_best_match(query)

            # Update success counter
            if result.get("found"):
                st.session_state.successful_matches += 1

            # Add bot response to history
            add_message(
                role     = "assistant",
                content  = result["answer"],
                metadata = {
                    "found":            result.get("found", False),
                    "confidence":       result.get("confidence", 0.0),
                    "matched_question": result.get("matched_question"),
                },
            )

            # Rerun to update UI with new messages
            st.rerun()

        else:
            st.warning("⚠️ Please enter a question before clicking Send.")

    # ── Footer ──────────────────────────────────────────────
    st.markdown(
        """
        <div class="app-footer">
            Made with ❤️ by <span class="author">Subhajit Roy</span><span class="cursor"></span>
            &nbsp;·&nbsp; © 2026 All Rights Reserved
            &nbsp;·&nbsp; CodeAlpha AI Internship
        </div>
        """,
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
