"""Text cleaning, normalization and tokenization (notebook Steps 2-4, Member 1)."""
import re
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

RANDOM_STATE = 42
# Negation words are kept: "not good" must differ from "good".
KEEP = {"not", "no", "nor", "never", "cannot", "n't"}
STOPWORDS = set(ENGLISH_STOP_WORDS) - KEEP


def clean_text(text: str) -> str:
    text = text.lower()                                   # 1. lowercase
    text = re.sub(r"<[^>]+>", " ", text)                  # 2. remove HTML tags
    text = re.sub(r"[^a-z\s]", " ", text)                 # 3. remove special chars / digits
    text = re.sub(r"\s+", " ", text).strip()              # 4. remove extra spaces
    tokens = text.split()                                 # 5. tokenization
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]  # 6. stopwords
    return " ".join(tokens)


def load_and_sample(path: str, sample_per_class: int = 5000) -> pd.DataFrame:
    """Load raw CSV (columns: review, sentiment), drop nulls/duplicates, balance classes."""
    df = pd.read_csv(path)
    missing = {"review", "sentiment"} - set(df.columns)
    if missing:
        raise KeyError(f"Missing column(s): {missing}. Found: {df.columns.tolist()}")
    df["sentiment"] = df["sentiment"].astype(str).str.strip().str.lower()
    df = df[["review", "sentiment"]].dropna().drop_duplicates().reset_index(drop=True)
    n = min(sample_per_class, df["sentiment"].value_counts().min())
    df = pd.concat([df[df.sentiment == "positive"].sample(n, random_state=RANDOM_STATE),
                    df[df.sentiment == "negative"].sample(n, random_state=RANDOM_STATE)])
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    df["label"] = (df["sentiment"] == "positive").astype(int)   # 1 = positive
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["word_count"] = df["review"].str.split().str.len()
    df["clean_review"] = df["review"].apply(clean_text)
    return df


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--raw", required=True, help="raw IMDB csv with review,sentiment")
    p.add_argument("--out", default="data/processed/cleaned_movie_reviews.csv")
    p.add_argument("--sample-per-class", type=int, default=5000)
    a = p.parse_args()
    out = preprocess(load_and_sample(a.raw, a.sample_per_class))
    out[["review", "clean_review", "sentiment"]].to_csv(a.out, index=False)
    print("Saved", a.out, out.shape)
