"""
Automated sentiment analysis for Elevate Retail Solutions.

This script simulates a small e-commerce review dataset, performs text
preprocessing, extracts TF-IDF features, trains a linear SVM model, and
evaluates its performance. Run with `python ecommerce_sentiment_svm.py`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

import nltk
from nltk.corpus import stopwords


@dataclass
class SentimentDataset:
    """Container for review text and labels."""

    reviews: pd.Series
    labels: pd.Series


DATA_DIR = Path(__file__).with_name("data")
CURATED_DATASET = DATA_DIR.joinpath("ecommerce_reviews.csv")
# Updated: train.csv is now in the same directory as the script
EXTERNAL_DATASET = Path(__file__).with_name("train.csv")


def load_external_reviews(
    csv_path: Path, max_per_class: Optional[int] = 10000
) -> pd.DataFrame:
    """Load Amazon-style CSV with rating/title/review into review/sentiment columns."""
    # Optimize: Read in chunks and sample early to avoid loading entire file
    sentiment_map = {1: "Negative", 2: "Positive"}
    chunk_size = 50000
    collected = {sent: [] for sent in sentiment_map.values()}
    target_per_class = max_per_class if max_per_class else float('inf')
    
    print(f"Loading dataset in chunks (max {max_per_class} per class)...")
    for chunk in pd.read_csv(
        csv_path, 
        header=None, 
        names=["rating", "title", "review"],
        chunksize=chunk_size,
        low_memory=False
    ):
        chunk = chunk[chunk["rating"].isin(sentiment_map)].dropna(subset=["review"])
        if len(chunk) == 0:
            continue
        chunk["sentiment"] = chunk["rating"].map(sentiment_map)
        chunk["review"] = (
            chunk["title"].fillna("").astype(str).str.strip() + " " + chunk["review"].astype(str)
        ).str.strip()
        
        # Collect samples per sentiment
        for sent, group in chunk.groupby("sentiment"):
            current_count = sum(len(df) for df in collected[sent])
            if current_count < target_per_class:
                needed = int(target_per_class - current_count)
                collected[sent].append(group.head(needed))
        
        # Early exit if we have enough samples
        if max_per_class and all(
            sum(len(df) for df in collected[sent]) >= target_per_class 
            for sent in sentiment_map.values()
        ):
            break
    
    # Combine collected samples
    frames = []
    for sent in sentiment_map.values():
        if collected[sent]:
            combined = pd.concat(collected[sent], ignore_index=True)
            if max_per_class and len(combined) > max_per_class:
                combined = combined.sample(max_per_class, random_state=42)
            frames.append(combined)
    
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["review", "sentiment"])
    return df[["review", "sentiment"]]


def load_curated_reviews(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Expected curated dataset at {csv_path}. Please recreate it."
        )
    df = pd.read_csv(csv_path)
    required_cols = {"review", "sentiment"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"{csv_path} must contain columns {required_cols}")
    return df[["review", "sentiment"]]


def load_dataset(csv_path: Path, max_per_class: Optional[int] = 10000) -> SentimentDataset:
    """Load reviews/sentiment columns from curated + external CSVs."""
    frames: List[pd.DataFrame] = []

    if EXTERNAL_DATASET.exists():
        print(f"Loading external dataset from {EXTERNAL_DATASET} ...")
        frames.append(load_external_reviews(EXTERNAL_DATASET, max_per_class))
    else:
        print(f"No external dataset found at {EXTERNAL_DATASET}.")

    if csv_path.exists():
        frames.append(load_curated_reviews(csv_path))

    if not frames:
        raise FileNotFoundError(
            f"No datasets found. Please provide either:\n"
            f"  - External dataset at {EXTERNAL_DATASET}\n"
            f"  - Curated dataset at {csv_path}"
        )

    df = pd.concat(frames, ignore_index=True)
    return SentimentDataset(reviews=df["review"], labels=df["sentiment"])


def clean_text(text: str) -> str:
    """Light text cleanup: lowercase, remove non-word chars, collapse spaces."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_vectorizer(stop_words: list[str]) -> TfidfVectorizer:
    """Create and configure the TF-IDF vectorizer."""
    return TfidfVectorizer(
        preprocessor=clean_text,
        stop_words=stop_words,
        ngram_range=(1, 2),
        min_df=1,
    )


def prepare_features(
    dataset: SentimentDataset,
    vectorizer: TfidfVectorizer,
    test_size: float = 0.3,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
    """Split data and transform text into TF-IDF matrices."""
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        dataset.reviews,
        dataset.labels,
        test_size=test_size,
        stratify=dataset.labels,
        random_state=random_state,
    )
    vectorizer.fit(X_train_text)
    X_train = vectorizer.transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)
    return X_train, X_test, y_train.reset_index(drop=True), y_test.reset_index(drop=True)


def train_svm(X_train: np.ndarray, y_train: pd.Series) -> SVC:
    """Train a linear-kernel SVM classifier."""
    model = SVC(kernel="linear", probability=False, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: SVC, X_test: np.ndarray, y_test: pd.Series) -> None:
    """Print evaluation metrics for the trained model."""
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, digits=3)
    conf_matrix = confusion_matrix(y_test, predictions, labels=["Positive", "Negative", "Neutral"])

    print("\nModel Evaluation")
    print("----------------")
    print(f"Accuracy: {accuracy:.3f}")
    print("\nClassification Report:")
    print(report)
    print("Confusion Matrix (rows=true, cols=pred):")
    print(
        pd.DataFrame(
            conf_matrix,
            index=["Positive", "Negative", "Neutral"],
            columns=["Positive", "Negative", "Neutral"],
        )
    )


ASPECT_KEYWORDS: Dict[str, List[str]] = {
    "battery": ["battery", "charge", "charging", "charger", "power"],
    "performance": ["performance", "speed", "lag", "slow", "fast", "processor"],
    "shipping": ["shipping", "delivery", "courier", "logistics", "package"],
    "design": ["design", "build", "look", "feel", "style"],
    "audio": ["audio", "sound", "speaker", "noise", "headphone"],
}


def ensure_stopwords() -> list[str]:
    """Download NLTK stopwords once and return the English stopword list."""
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")
    return stopwords.words("english")


def main() -> None:
    dataset = load_dataset(CURATED_DATASET)
    print("Preview of dataset:")
    preview = pd.DataFrame(
        {"review": dataset.reviews, "sentiment": dataset.labels}
    ).head()
    print(preview)
    print(f"\nTotal samples: {len(dataset.reviews)}")

    stop_words = ensure_stopwords()
    vectorizer = build_vectorizer(stop_words)
    X_train, X_test, y_train, y_test = prepare_features(dataset, vectorizer)

    svm_model = train_svm(X_train, y_train)
    evaluate_model(svm_model, X_test, y_test)


if __name__ == "__main__":
    main()
