# preprocessor.py
#import required libraries for NLP preprocessing and data manipulation

import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


# Download required NLTK data (runs once) 

def download_nltk_resources():
    resources = ["punkt", "stopwords", "wordnet", "omw-1.4", "punkt_tab"]
    for resource in resources:
        nltk.download(resource, quiet=True)


#Text cleaning functions 

def to_lowercase(text: str) -> str:
    """Convert all characters to lowercase."""
    return text.lower()


def remove_html_tags(text: str) -> str:
    """Strip any HTML tags that may have been scraped."""
    return re.sub(r"<[^>]+>", " ", text)


def remove_urls(text: str) -> str:
    """Remove any URLs."""
    return re.sub(r"http\S+|www\S+", " ", text)


def remove_punctuation(text: str) -> str:
    """Remove punctuation and special characters, keep letters and spaces."""
    return re.sub(r"[^a-z\s]", " ", text)


def remove_extra_spaces(text: str) -> str:
    """Collapse multiple spaces into one and strip leading/trailing spaces."""
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    """Split text into individual word tokens."""
    return word_tokenize(text)


def remove_stopwords(tokens: list[str]) -> list[str]:
    """Remove common English stopwords (the, is, a, an, …)."""
    stop_words = set(stopwords.words("english"))
    return [token for token in tokens if token not in stop_words]


def lemmatize(tokens: list[str]) -> list[str]:
    """
    Reduce each word to its base form.
    e.g. 'running' → 'run', 'battles' → 'battle'
    """
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]


def remove_short_words(tokens: list[str], min_length: int = 2) -> list[str]:
    """Drop single-character tokens that add no meaning."""
    return [token for token in tokens if len(token) >= min_length]


# ── Full pipeline ──────────────────────────────────────────────────────────────

def clean_storyline(text: str) -> str:
    """
    Apply the complete cleaning pipeline to a single storyline string.
    Returns a clean, space-joined string ready for vectorization.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    text = to_lowercase(text)
    text = remove_html_tags(text)
    text = remove_urls(text)
    text = remove_punctuation(text)
    text = remove_extra_spaces(text)

    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize(tokens)
    tokens = remove_short_words(tokens)

    return " ".join(tokens)

def Movie_Name(text: str) -> str:
    """
    Apply the complete cleaning pipeline to a single Movie_Name string.
    Returns a clean, space-joined string ready for vectorization.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    text = to_lowercase(text)
    text = remove_html_tags(text)
    text = remove_urls(text)
    text = remove_punctuation(text)
    text = remove_extra_spaces(text)

    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize(tokens)
    tokens = remove_short_words(tokens)

    return " ".join(tokens)


# ── Analysis helpers ──────────────────────────────────────────────────────────

def show_sample_comparison(df: pd.DataFrame, n: int = 3):
    """Print a before/after comparison for a few rows."""
    print("\n── Sample before / after cleaning ──────────────────────────────")
    for _, row in df.head(n).iterrows():
        print(f"\nMovie  : {row['Movie Name']}")
        print(f"Before : {row['Storyline'][:120]}…")
        print(f"After  : {row['Cleaned_Storyline'][:120]}…")
    print()


def show_basic_stats(df: pd.DataFrame):
    """Print dataset statistics after cleaning."""
    total      = len(df)
    with_plot  = df["Cleaned_Storyline"].apply(lambda x: len(x) > 0).sum()
    empty      = total - with_plot
    avg_words  = df["Cleaned_Storyline"].apply(lambda x: len(x.split())).mean()

    print("── Dataset statistics ──────────────────────────────────────────")
    print(f"  Total movies       : {total}")
    print(f"  With storyline     : {with_plot}")
    print(f"  Empty storylines   : {empty}  (will be dropped)")
    print(f"  Avg words (cleaned): {avg_words:.1f}")
    print()

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    INPUT_FILE  = "imdb_movies_2024.csv"
    OUTPUT_FILE = "preprocessed_movies.csv"

    print("Phase 2 — NLP Preprocessing\n")

    # 1. Download NLTK data
    print("Downloading NLTK resources…")
    download_nltk_resources()

    # 2. Load raw data
    print(f"Loading '{INPUT_FILE}'…")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"ERROR: '{INPUT_FILE}' not found. Run scraper.py first.")
        return

    print(f"  Loaded {len(df)} rows with columns: {list(df.columns)}\n")

    # 3. Basic validation
    required_cols = {"Movie Name", "Storyline"}
    if not required_cols.issubset(df.columns):
        print(f"ERROR: CSV must have columns {required_cols}. Found: {list(df.columns)}")
        return

    # 4. Drop rows with missing data
    before = len(df)
    df.dropna(subset=["Movie Name", "Storyline"], inplace=True)
    df.drop_duplicates(subset="Movie Name", inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"  Dropped {before - len(df)} duplicate/empty rows → {len(df)} remain\n")

    # 5. Apply cleaning pipeline
    print("Cleaning storylines…")
    df["Cleaned_Storyline"] = df["Storyline"].apply(clean_storyline)

    # 6. Drop rows where cleaning left an empty string
    df = df[df["Cleaned_Storyline"].str.strip() != ""]
    df.reset_index(drop=True, inplace=True)

    # 7. Show stats and samples
    show_basic_stats(df)
    show_sample_comparison(df)

    # 8. Save output
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    print(f"Saved cleaned data to '{OUTPUT_FILE}'")
    print(df[["Movie Name", "Cleaned_Storyline"]].head())


if __name__ == "__main__":
    main()
