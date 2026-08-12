# -*- coding: utf-8 -*-
from __future__ import annotations

import ast
import base64
import html
import json
import re
import unicodedata
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# PATHS — DEPLOY-SAFE FOR GITHUB / STREAMLIT CLOUD
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "linear_svm_calibrated.joblib"
STOPWORD_PATH = BASE_DIR / "final_stopwords_tokens.pkl"

# Public repo used only for storing user-submitted prediction reports.
GITHUB_OWNER = "vuduc1711"
GITHUB_REPO = "Sentiment-demo"
GITHUB_BRANCH = "main"
REPORTS_DIR = "reports"


# ============================================================
# RAW TEXT CLEANING
# ============================================================

KEEP_PUNCT = set("%+-")

HTML_TAG_PATTERN = re.compile(r"<[^>]+>")

URL_PATTERN = re.compile(
    r"""
    (?:
        https?://\S+
        |
        www\.\S+
        |
        \b[a-zA-Z0-9.-]+\.
        (?:com|net|org|vn|com\.vn|edu|gov|io|co|me|info|
           biz|tv|ly|app|dev|ai|finance|stock)
        \b(?:/\S*)?
    )
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)

WHITESPACE_PATTERN = re.compile(r"\s+")
REPEATED_SPECIAL_PATTERN = re.compile(r"([^\w\s])\1{2,}")


def remove_html_tag(text: Any) -> str:
    text = "" if text is None else str(text)
    text = html.unescape(text)
    return HTML_TAG_PATTERN.sub(" ", text)


def remove_urls(text: Any) -> str:
    text = "" if text is None else str(text)
    return URL_PATTERN.sub(" ", text)


def clean_special_chars_v2(
    text: Any,
    keep_punct: set[str] = KEEP_PUNCT,
) -> str:
    text = "" if text is None else str(text)

    text = unicodedata.normalize("NFKC", text)
    text = unicodedata.normalize("NFC", text)

    cleaned = []
    for ch in text:
        if ch.isalnum() or ch.isspace():
            cleaned.append(ch)
        elif ch in keep_punct:
            cleaned.append(ch)
        else:
            cleaned.append(" ")

    text = "".join(cleaned)
    text = REPEATED_SPECIAL_PATTERN.sub(" ", text)
    text = WHITESPACE_PATTERN.sub(" ", text).strip()
    return text


def preprocess_sentence_text(
    text: Any,
    keep_punct: set[str] = KEEP_PUNCT,
) -> str:
    if text is None:
        return ""

    try:
        if pd.isna(text):
            return ""
    except (TypeError, ValueError):
        pass

    text = str(text)
    text = remove_html_tag(text)
    text = remove_urls(text)
    text = text.lower()
    text = clean_special_chars_v2(text, keep_punct=keep_punct)
    return text


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize_with_underthesea(text_clean: Any) -> list[str]:
    if text_clean is None:
        return []

    text_clean = str(text_clean).strip()
    if not text_clean:
        return []

    from underthesea import word_tokenize

    tokenized_text = word_tokenize(text_clean, format="text")

    return [
        token.strip()
        for token in tokenized_text.split()
        if token.strip()
    ]


# ============================================================
# STOPWORDS
# ============================================================

@st.cache_resource
def load_stopwords() -> set[str]:
    if not STOPWORD_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy stopword file: {STOPWORD_PATH}"
        )

    loaded = pd.read_pickle(STOPWORD_PATH)

    if isinstance(loaded, pd.Series):
        stopwords = set(
            loaded.dropna()
            .astype(str)
            .str.strip()
            .str.lower()
        )

    elif isinstance(loaded, (list, tuple, set, np.ndarray)):
        stopwords = {
            str(x).strip().lower()
            for x in loaded
            if str(x).strip()
        }

    else:
        raise TypeError(
            f"Stopword object có type không dự kiến: {type(loaded)}"
        )

    stopwords.discard("")
    return stopwords


def remove_stopwords_from_tokens(
    tokens: Any,
    stopwords: set[str],
) -> list[str]:
    if not isinstance(tokens, (list, tuple)):
        return []

    return [
        str(token).strip()
        for token in tokens
        if (
            str(token).strip()
            and str(token).strip().lower() not in stopwords
        )
    ]


# ============================================================
# FINAL TOKEN CLEANING
# ============================================================

def ensure_token_list(
    value: Any,
    *,
    lowercase: bool = True,
) -> list[str]:
    if value is None:
        return []

    if isinstance(value, float) and np.isnan(value):
        return []

    if isinstance(value, (list, tuple, set)):
        tokens = list(value)

    elif isinstance(value, str):
        value = value.strip()

        if not value:
            return []

        if value.startswith("[") and value.endswith("]"):
            try:
                parsed = ast.literal_eval(value)
                if isinstance(parsed, (list, tuple, set)):
                    tokens = list(parsed)
                else:
                    tokens = value.split()
            except (ValueError, SyntaxError):
                tokens = value.split()
        else:
            tokens = value.split()

    else:
        return []

    tokens = [
        str(token).strip()
        for token in tokens
        if str(token).strip()
    ]

    if lowercase:
        tokens = [token.lower() for token in tokens]

    return tokens


def is_number_token(token: str) -> bool:
    return bool(re.fullmatch(r"\d+", str(token)))


def is_symbols_only_token(token: str) -> bool:
    return bool(
        re.fullmatch(
            r"[\W_]+",
            str(token),
            flags=re.UNICODE,
        )
    )


def is_url_like_token(token: str) -> bool:
    token = str(token).lower()

    return (
        token.startswith("http")
        or token.startswith("www")
        or ".com" in token
        or ".vn" in token
        or ".html" in token
        or ".aspx" in token
        or "http_" in token
        or "https_" in token
    )


def has_repeat_char_spam(
    token: str,
    max_repeat: int = 4,
) -> bool:
    pattern = r"(.)\1{" + str(max_repeat - 1) + r",}"
    return bool(re.search(pattern, str(token)))


def is_spam_like_token(token: str) -> bool:
    token = str(token).lower().strip()

    if not token:
        return True

    if len(token) > 80:
        return True

    if token.count("/") >= 2 or token.count(".") >= 3:
        return True

    if len(set(token)) == 1 and len(token) >= 4:
        return True

    return False


def clean_forum_tokens_final(
    tokens: Any,
    *,
    min_len: int = 2,
    remove_numbers: bool = True,
    remove_symbols_only: bool = True,
    remove_url_like: bool = True,
    remove_spam_like: bool = True,
    remove_repeat_char: bool = True,
    lowercase: bool = True,
) -> list[str]:
    tokens = ensure_token_list(tokens, lowercase=lowercase)

    cleaned_tokens = []

    for token in tokens:
        token = str(token).strip()

        if lowercase:
            token = token.lower()

        if min_len is not None and len(token) < min_len:
            continue

        if remove_numbers and is_number_token(token):
            continue

        if remove_symbols_only and is_symbols_only_token(token):
            continue

        if remove_url_like and is_url_like_token(token):
            continue

        if remove_spam_like and is_spam_like_token(token):
            continue

        if remove_repeat_char and has_repeat_char_spam(token):
            continue

        cleaned_tokens.append(token)

    return cleaned_tokens


# ============================================================
# FULL RAW SENTENCE -> content_tok_final
# ============================================================

def preprocess_one_sentence(
    sentence_raw: str,
    stopwords: set[str],
) -> str:
    sentence_clean = preprocess_sentence_text(sentence_raw)

    sentence_tokens = tokenize_with_underthesea(sentence_clean)

    sentence_tokens_sw = remove_stopwords_from_tokens(
        sentence_tokens,
        stopwords=stopwords,
    )

    sentence_tokens_final = clean_forum_tokens_final(
        sentence_tokens_sw,
        min_len=2,
        remove_numbers=True,
        remove_symbols_only=True,
        remove_url_like=True,
        remove_spam_like=True,
        remove_repeat_char=True,
        lowercase=True,
    )

    return " ".join(sentence_tokens_final)


# ============================================================
# MODEL
# ============================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy saved SVM: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    if not hasattr(model, "predict"):
        raise TypeError("Model không có predict().")

    if not hasattr(model, "predict_proba"):
        raise TypeError("Model không có predict_proba().")

    return model


def get_vectorizer(model):
    """
    Find the TF-IDF vectorizer inside the saved sklearn Pipeline.
    Used only to detect a completely OOV / zero-feature input.
    """
    if hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            if hasattr(step, "vocabulary_") and hasattr(step, "transform"):
                return step
    return None


def predict_sentence(
    text: str,
    model,
    stopwords: set[str],
):
    model_text = preprocess_one_sentence(
        text,
        stopwords=stopwords,
    )

    if not model_text:
        return {
            "label": "uncertain",
            "p_positive": None,
            "p_negative": None,
        }

    # Important: do not interpret a zero TF-IDF vector as negative.
    vectorizer = get_vectorizer(model)
    if vectorizer is not None:
        x_tfidf = vectorizer.transform([model_text])
        if x_tfidf.nnz == 0:
            return {
                "label": "uncertain",
                "p_positive": None,
                "p_negative": None,
            }

    probs = model.predict_proba([model_text])[0]
    classes = list(model.classes_)

    if 1 not in classes or 0 not in classes:
        raise ValueError(
            f"Expected model classes [0, 1], got {classes}"
        )

    p_pos = float(probs[classes.index(1)])
    p_neg = float(probs[classes.index(0)])

    label = "positive" if p_pos >= p_neg else "negative"

    return {
        "label": label,
        "p_positive": p_pos,
        "p_negative": p_neg,
    }



# ============================================================
# REPORT PREDICTION -> GITHUB
# ============================================================

def get_github_token() -> str | None:
    """Read the GitHub token from Streamlit Secrets."""
    try:
        token = st.secrets.get("GITHUB_TOKEN", None)
    except Exception:
        token = None

    if token is None:
        return None

    token = str(token).strip()
    return token or None


def save_report_to_github(
    *,
    text: str,
    result: dict,
    reason: str,
    corrected_label: str | None,
    note: str,
) -> str:
    """
    Save one report as its own JSON file under reports/.
    This avoids append conflicts when multiple people report at once.
    """
    token = get_github_token()
    if not token:
        raise RuntimeError(
            "Reporting is not configured yet. Missing GITHUB_TOKEN in Streamlit Secrets."
        )

    created_at = datetime.now(timezone.utc)
    report_id = uuid.uuid4().hex[:10]

    report = {
        "report_id": report_id,
        "created_at_utc": created_at.isoformat(),
        "text": text,
        "predicted_label": result.get("label"),
        "p_positive": result.get("p_positive"),
        "p_negative": result.get("p_negative"),
        "reason": reason,
        "corrected_label": corrected_label,
        "note": note.strip() if note else "",
    }

    filename = f"{created_at.strftime('%Y%m%dT%H%M%SZ')}_{report_id}.json"
    repo_path = f"{REPORTS_DIR}/{filename}"

    api_url = (
        f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}"
        f"/contents/{repo_path}"
    )

    raw = json.dumps(
        report,
        ensure_ascii=False,
        indent=2,
    ).encode("utf-8")

    payload = {
        "message": f"Add prediction report {report_id}",
        "content": base64.b64encode(raw).decode("ascii"),
        "branch": GITHUB_BRANCH,
    }

    req = urllib.request.Request(
        api_url,
        data=json.dumps(payload).encode("utf-8"),
        method="PUT",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "Content-Type": "application/json",
            "User-Agent": "sentiment-demo-streamlit",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=20) as response:
            if response.status not in (200, 201):
                raise RuntimeError(
                    f"GitHub returned HTTP {response.status}"
                )
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"GitHub API error {exc.code}: {body}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not connect to GitHub API: {exc}"
        ) from exc

    return repo_path


# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(
    page_title="Vietnamese Investor Sentiment",
    page_icon="📈",
    layout="centered",
)

st.title("Vietnamese Investor Sentiment")
st.caption("Prediction model for Vietnamese investor sentiment")

try:
    model = load_model()
    stopwords = load_stopwords()
except Exception as exc:
    st.error(f"Không load được model/data: {exc}")
    st.stop()

text = st.text_area(
    "Nhập câu hoặc đoạn post:",
    height=170,
    placeholder="Ví dụ: múc mạnh anh em ơi",
    key="input_text",
)

if st.button("Predict", type="primary", use_container_width=True):
    if not text.strip():
        st.warning("Nhập text trước.")
        st.session_state.pop("last_prediction", None)
    else:
        try:
            result = predict_sentence(
                text=text,
                model=model,
                stopwords=stopwords,
            )
            st.session_state["last_prediction"] = {
                "text": text,
                "result": result,
            }
        except Exception as exc:
            st.error(f"Prediction error: {exc}")
            st.session_state.pop("last_prediction", None)

last_prediction = st.session_state.get("last_prediction")

if last_prediction is not None:
    predicted_text = last_prediction["text"]
    result = last_prediction["result"]
    label = result["label"]

    if label == "positive":
        st.success("POSITIVE")
    elif label == "negative":
        st.error("NEGATIVE")
    else:
        st.warning("UNCERTAIN")

    c1, c2 = st.columns(2)

    if result["p_positive"] is None:
        c1.metric("P(Positive)", "N/A")
        c2.metric("P(Negative)", "N/A")
        st.caption(
            "Model không nhận được feature phù hợp từ input này."
        )
    else:
        c1.metric(
            "P(Positive)",
            f"{result['p_positive']:.1%}",
        )
        c2.metric(
            "P(Negative)",
            f"{result['p_negative']:.1%}",
        )

    st.divider()

    with st.expander("🚩 Report this prediction"):
        st.caption(
            "Nếu prediction sai hoặc đáng ngờ, bạn có thể gửi report."
        )

        with st.form("report_form", clear_on_submit=True):
            reason = st.radio(
                "Lý do report",
                [
                    "Wrong label",
                    "Suspicious / uncertain",
                    "Slang misunderstood",
                    "Negation misunderstood",
                    "Other",
                ],
                index=0,
            )

            corrected = st.selectbox(
                "Theo bạn label đúng là gì? (optional)",
                ["Không chắc", "positive", "negative", "uncertain"],
                index=0,
            )

            note = st.text_area(
                "Ghi chú thêm (optional)",
                height=80,
                placeholder="Ví dụ: câu này là slang bullish.",
            )

            submit_report = st.form_submit_button(
                "Submit report",
                use_container_width=True,
            )

            if submit_report:
                corrected_label = (
                    None if corrected == "Không chắc" else corrected
                )

                try:
                    repo_path = save_report_to_github(
                        text=predicted_text,
                        result=result,
                        reason=reason,
                        corrected_label=corrected_label,
                        note=note,
                    )
                    st.success("Report submitted. Cảm ơn bạn!")
                    st.caption(f"Saved as: {repo_path}")
                except Exception as exc:
                    st.error(f"Không gửi được report: {exc}")
