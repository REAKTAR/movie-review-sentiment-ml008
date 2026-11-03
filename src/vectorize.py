"""TF-IDF feature extraction (notebook Step 5, Member 2)."""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

RANDOM_STATE = 42


def split_data(df: pd.DataFrame, test_size: float = 0.2):
    """Stratified 80/20 split. Locks the test set away before any fitting."""
    if "label" not in df:
        df = df.assign(label=(df["sentiment"] == "positive").astype(int))
    return train_test_split(df["clean_review"], df["label"], test_size=test_size,
                            random_state=RANDOM_STATE, stratify=df["label"])


def build_tfidf(X_train_text, X_test_text):
    """Fit TF-IDF on TRAIN only (prevents data leakage), then transform both."""
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
    X_train = tfidf.fit_transform(X_train_text)
    X_test = tfidf.transform(X_test_text)
    return tfidf, X_train, X_test
