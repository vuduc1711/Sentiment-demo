#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Predict raw Vietnamese sentences with the SAVED consensus SVM.

Expected location of this script:
    ...\0_TechnicalSteps\3_ASUM\svm_predict.py

Saved model:
    ./0_Data/consensus_classifier/svm/linear_svm_calibrated.joblib

Model from Classification.ipynb:
    Pipeline(
        TF-IDF word 1-2 grams
        +
        CalibratedClassifierCV(LinearSVC)
    )

The model was trained on `content_tok_final`, so raw terminal input must
be preprocessed in the same way:
    raw text
      -> HTML/URL removal
      -> lowercase + special-char cleaning
      -> Underthesea word tokenization
      -> old stopword removal
      -> final forum-token cleaning
      -> joined `content_tok_final`
      -> saved SVM

Usage
-----
Interactive:
    python svm_predict.py

One sentence:
    python svm_predict.py "múc mạnh anh em ơi"

Debug preprocessing:
    python svm_predict.py --debug "múc mạnh anh em ơi"

Optional uncertainty/reject region:
    python svm_predict.py --min-confidence 0.60 "múc mạnh anh em ơi"

Without reject region, the SVM remains a pure binary positive/negative model.
"""

from __future__ import annotations

import argparse
import ast
import html
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "0_Data"
    / "consensus_classifier"
    / "svm"
    / "linear_svm_calibrated.joblib"
)

# Exact path used by ASUM.ipynb.
ABS_STOPWORD_PATH = Path(
    r"C:\Users\Vu Duc\OneDrive\Desktop\0_AnalystReport"
    r"\1_ Analyst report and investor sentiment"
    r"\0_TechnicalSteps\0_Clustering"
    r"\0_Stopwords\final_stopwords_tokens.pkl"
)

# Also try sensible relative locations so the script remains portable.
STOPWORD_CANDIDATES = [
    ABS_STOPWORD_PATH,
    BASE_DIR.parent / "0_Clustering" / "0_Stopwords" / "final_stopwords_tokens.pkl",
    BASE_DIR / "0_Data" / "final_stopwords_tokens.pkl",
]


# ============================================================
# RAW TEXT CLEANING — copied from ASUM.ipynb preprocessing
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

    cleaned: list[str] = []

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

    try:
        from underthesea import word_tokenize
    except ImportError as exc:
        raise RuntimeError(
            "Thiếu package underthesea.\n"
            "Cài bằng:\n"
            "    pip install underthesea"
        ) from exc

    tokenized_text = word_tokenize(
        text_clean,
        format="text",
    )

    return [
        token.strip()
        for token in tokenized_text.split()
        if token.strip()
    ]


# ============================================================
# STOPWORDS
# ============================================================

def resolve_stopword_path(custom_path: str | None = None) -> Path:
    if custom_path:
        p = Path(custom_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"Không thấy stopword file: {p}")

    for p in STOPWORD_CANDIDATES:
        if p.exists():
            return p

    attempted = "\n".join(f"  - {p}" for p in STOPWORD_CANDIDATES)

    raise FileNotFoundError(
        "Không tìm thấy final_stopwords_tokens.pkl.\n"
        "Đã thử:\n"
        f"{attempted}\n\n"
        "Có thể truyền thủ công:\n"
        '    python svm_predict.py --stopwords "FULL_PATH_TO_PKL"'
    )


def load_stopwords(custom_path: str | None = None):
    p = resolve_stopword_path(custom_path)
    loaded = pd.read_pickle(p)

    if isinstance(loaded, pd.Series):
        stopwords = set(
            loaded.dropna()
            .astype(str)
            .str.strip()
            .str.lower()
        )

    elif isinstance(
        loaded,
        (list, tuple, set, np.ndarray),
    ):
        stopwords = {
            str(x).strip().lower()
            for x in loaded
            if str(x).strip()
        }

    else:
        raise TypeError(
            "Stopword object có type không dự kiến: "
            f"{type(loaded)}"
        )

    stopwords.discard("")
    return stopwords, p


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
# FINAL TOKEN CLEANING — same as ASUM.ipynb
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
) -> tuple[list[str], dict[str, int]]:
    tokens = ensure_token_list(
        tokens,
        lowercase=lowercase,
    )

    cleaned_tokens: list[str] = []

    removal_counts = {
        "too_short": 0,
        "number": 0,
        "symbols_only": 0,
        "url_like": 0,
        "spam_like": 0,
        "repeat_char": 0,
    }

    for token in tokens:
        token = str(token).strip()

        if lowercase:
            token = token.lower()

        if min_len is not None and len(token) < min_len:
            removal_counts["too_short"] += 1
            continue

        if remove_numbers and is_number_token(token):
            removal_counts["number"] += 1
            continue

        if remove_symbols_only and is_symbols_only_token(token):
            removal_counts["symbols_only"] += 1
            continue

        if remove_url_like and is_url_like_token(token):
            removal_counts["url_like"] += 1
            continue

        if remove_spam_like and is_spam_like_token(token):
            removal_counts["spam_like"] += 1
            continue

        if remove_repeat_char and has_repeat_char_spam(token):
            removal_counts["repeat_char"] += 1
            continue

        cleaned_tokens.append(token)

    return cleaned_tokens, removal_counts


# ============================================================
# FULL RAW SENTENCE -> content_tok_final
# ============================================================

def preprocess_one_sentence(
    sentence_raw: str,
    stopwords: set[str],
):
    sentence_clean = preprocess_sentence_text(sentence_raw)

    sentence_tokens = tokenize_with_underthesea(
        sentence_clean
    )

    sentence_tokens_sw = remove_stopwords_from_tokens(
        sentence_tokens,
        stopwords=stopwords,
    )

    sentence_tokens_final, removed = clean_forum_tokens_final(
        sentence_tokens_sw,
        min_len=2,
        remove_numbers=True,
        remove_symbols_only=True,
        remove_url_like=True,
        remove_spam_like=True,
        remove_repeat_char=True,
        lowercase=True,
    )

    content_tok_final = " ".join(sentence_tokens_final)

    return {
        "raw_text": sentence_raw,
        "sentence_clean": sentence_clean,
        "sentence_tokens": sentence_tokens,
        "sentence_tokens_sw": sentence_tokens_sw,
        "sentence_tokens_final": sentence_tokens_final,
        "content_tok_final": content_tok_final,
        "n_tokens": len(sentence_tokens),
        "n_tokens_sw": len(sentence_tokens_sw),
        "n_tokens_final": len(sentence_tokens_final),
        "n_stopwords_removed": (
            len(sentence_tokens) - len(sentence_tokens_sw)
        ),
        "removed": removed,
    }


# ============================================================
# LOAD SAVED SVM
# ============================================================

def load_svm(model_path: str | None = None):
    p = Path(model_path) if model_path else MODEL_PATH

    if not p.exists():
        raise FileNotFoundError(
            "Không thấy saved SVM:\n"
            f"    {p}\n\n"
            "Classification.ipynb save model tại:\n"
            "    ./0_Data/consensus_classifier/svm/"
            "linear_svm_calibrated.joblib"
        )

    model = joblib.load(p)

    if not hasattr(model, "predict"):
        raise TypeError(
            "Object load được không có predict()."
        )

    if not hasattr(model, "predict_proba"):
        raise TypeError(
            "Model không có predict_proba(). "
            "Expected calibrated LinearSVC pipeline."
        )

    return model, p


# ============================================================
# PREDICT
# ============================================================

def predict_sentence(
    text: str,
    model,
    stopwords: set[str],
    min_confidence: float | None = None,
):
    prep = preprocess_one_sentence(
        text,
        stopwords=stopwords,
    )

    model_text = prep["content_tok_final"]

    if not model_text:
        return {
            **prep,
            "svm_y": None,
            "svm_label_binary": None,
            "svm_label": "uncertain",
            "svm_prob_positive": None,
            "svm_prob_negative": None,
            "svm_confidence": 0.0,
            "svm_margin": 0.0,
            "warning": (
                "Không còn token nào sau preprocessing; "
                "không gọi SVM."
            ),
        }

    X = [model_text]

    y = int(model.predict(X)[0])

    prob = model.predict_proba(X)[0]

    # Do not assume column 1 blindly; resolve class positions.
    classes = list(model.classes_)

    if 1 not in classes or 0 not in classes:
        raise ValueError(
            f"Expected model classes [0,1], got {classes}"
        )

    p_pos = float(prob[classes.index(1)])
    p_neg = float(prob[classes.index(0)])

    binary_label = (
        "positive"
        if y == 1
        else "negative"
    )

    confidence = max(p_pos, p_neg)
    margin = abs(p_pos - 0.5) * 2.0

    if (
        min_confidence is not None
        and confidence < min_confidence
    ):
        final_label = "uncertain"
    else:
        final_label = binary_label

    return {
        **prep,
        "svm_y": y,
        "svm_label_binary": binary_label,
        "svm_label": final_label,
        "svm_prob_positive": p_pos,
        "svm_prob_negative": p_neg,
        "svm_confidence": confidence,
        "svm_margin": margin,
        "warning": None,
    }


# ============================================================
# OUTPUT
# ============================================================

def print_result(
    result: dict,
    *,
    debug: bool = False,
    as_json: bool = False,
):
    if as_json:
        minimal = {
            "label": result["svm_label"],
            "p_positive": result["svm_prob_positive"],
            "p_negative": result["svm_prob_negative"],
        }
        print(
            json.dumps(
                minimal,
                ensure_ascii=False,
                default=str,
            )
        )
        return

    print(f"label      : {result['svm_label']}")

    if result["svm_prob_positive"] is None:
        print("p_positive : None")
        print("p_negative : None")
    else:
        print(
            f"p_positive : "
            f"{result['svm_prob_positive']:.4f}"
        )
        print(
            f"p_negative : "
            f"{result['svm_prob_negative']:.4f}"
        )


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Load the saved calibrated Linear SVM from "
            "Classification.ipynb and predict new raw sentences."
        )
    )

    parser.add_argument(
        "text",
        nargs="*",
        help=(
            "Raw Vietnamese sentence. "
            "If omitted, interactive mode is used."
        ),
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override saved model path.",
    )

    parser.add_argument(
        "--stopwords",
        type=str,
        default=None,
        help="Override final_stopwords_tokens.pkl path.",
    )

    parser.add_argument(
        "--min-confidence",
        type=float,
        default=None,
        help=(
            "Optional reject threshold. "
            "Example 0.60 => confidence < 0.60 becomes uncertain. "
            "Default is None, preserving the original binary SVM."
        ),
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show all preprocessing stages.",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON output.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if (
        args.min_confidence is not None
        and not (0.5 <= args.min_confidence <= 1.0)
    ):
        print(
            "ERROR: --min-confidence phải nằm trong [0.5, 1.0].",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        model, model_path = load_svm(args.model)

        stopwords, stopword_path = load_stopwords(
            args.stopwords
        )

        # Minimal show-off mode: no startup diagnostics.


        if args.text:
            text = " ".join(args.text).strip()

            result = predict_sentence(
                text=text,
                model=model,
                stopwords=stopwords,
                min_confidence=args.min_confidence,
            )

            print_result(
                result,
                debug=args.debug,
                as_json=args.json,
            )
            return

        print("Nhập câu cần predict. Ctrl+C / Ctrl+D để thoát.\n")

        while True:
            try:
                text = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not text:
                continue

            result = predict_sentence(
                text=text,
                model=model,
                stopwords=stopwords,
                min_confidence=args.min_confidence,
            )

            print_result(
                result,
                debug=args.debug,
                as_json=args.json,
            )
            print()

    except Exception as exc:
        print(
            f"ERROR: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
