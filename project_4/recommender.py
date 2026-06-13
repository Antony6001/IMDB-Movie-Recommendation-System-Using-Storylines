"""
Phase 3 — TF-IDF Vectorization + Cosine Similarity Recommendation Engine
Loads preprocessed_movies.csv, builds a TF-IDF matrix, and recommends
the top 5 most similar movies for any given input storyline.

Requirements:
    pip install pandas scikit-learn nltk
"""

import re
import pickle

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ── NLTK setup ────────────────────────────────────────────────────────────────

def download_nltk_resources():
    for resource in ["punkt", "stopwords", "wordnet", "omw-1.4", "punkt_tab"]:
        nltk.download(resource, quiet=True)


# ── Same cleaning pipeline as Phase 2 ────────────────────────────────────────
# (kept here so a raw user input can be cleaned before querying)

def clean_text(text: str) -> str:
    """Apply the full NLP cleaning pipeline to a raw string."""
    if not isinstance(text, str) or not text.strip():
        return ""

    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)       # remove HTML
    text = re.sub(r"http\S+|www\S+", " ", text) # remove URLs
    text = re.sub(r"[^a-z\s]", " ", text)       # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()    # collapse spaces

    tokens = word_tokenize(text)

    stop_words  = set(stopwords.words("english"))
    lemmatizer  = WordNetLemmatizer()

    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    tokens = [t for t in tokens if len(t) >= 2]

    return " ".join(tokens)


# ── TF-IDF model ──────────────────────────────────────────────────────────────

class MovieRecommender:
    """
    Builds a TF-IDF matrix from cleaned storylines and returns the top-N
    most similar movies for any input storyline.
    """

    def __init__(self,
                 max_features: int = 5000,
                 ngram_range: tuple = (1, 2),
                 top_n: int = 5):
        """
        Parameters
        ----------
        max_features : int
            Maximum number of unique terms to keep in the vocabulary.
            Higher = richer model but more memory. 5000 is a good default.
        ngram_range : tuple
            (1, 1) → single words only
            (1, 2) → single words + two-word phrases (recommended)
        top_n : int
            Number of recommendations to return.
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,   # apply log(1+tf) to dampen common terms
        )
        self.top_n      = top_n
        self.tfidf_matrix = None
        self.df           = None

    # ── Training / fitting ────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame):
        """
        Fit the TF-IDF vectorizer on the cleaned storylines and store the
        resulting matrix.

        Parameters
        ----------
        df : DataFrame with columns 'Movie_Name' and 'Cleaned_Storyline'
        """
        self.df = df.reset_index(drop=True)
        print(f"Fitting TF-IDF on {len(df)} movies…")
        self.tfidf_matrix = self.vectorizer.fit_transform(
            self.df["Cleaned_Storyline"]
        )
        print(f"  Vocabulary size : {len(self.vectorizer.vocabulary_)}")
        print(f"  Matrix shape    : {self.tfidf_matrix.shape}")  # (movies, terms)

    # ── Recommendation ────────────────────────────────────────────────────────

    def recommend(self, input_storyline: str) -> pd.DataFrame:
        """
        Given a raw or cleaned input storyline, return the top-N similar movies.

        Parameters
        ----------
        input_storyline : str
            Free-text plot description entered by the user.

        Returns
        -------
        DataFrame with columns: Rank, Movie_Name, Storyline, Similarity_Score
        """
        if self.tfidf_matrix is None:
            raise RuntimeError("Call fit() before recommend().")

        # Clean the input the same way the training data was cleaned
        cleaned_input = clean_text(input_storyline)
        if not cleaned_input:
            print("Warning: input storyline is empty after cleaning.")
            return pd.DataFrame()

        # Transform input into TF-IDF space
        input_vector  = self.vectorizer.transform([cleaned_input])

        # Compute cosine similarity against every movie in the matrix
        similarity_scores = cosine_similarity(input_vector, self.tfidf_matrix)[0]

        # Rank movies by similarity score (descending)
        top_indices = similarity_scores.argsort()[::-1][:self.top_n]

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            results.append({
                "Rank"             : rank,
                "Movie_Name"       : self.df.loc[idx, "Movie_Name"],
                "Storyline"        : self.df.loc[idx, "Storyline"],
                "Similarity_Score" : round(float(similarity_scores[idx]), 4),
            })

        return pd.DataFrame(results)

    # ── Movie-to-movie recommendations ───────────────────────────────────────

    def recommend_by_title(self, movie_title: str) -> pd.DataFrame:
        """
        Find movies similar to a title that already exists in the dataset.

        Parameters
        ----------
        movie_title : str  (case-insensitive, partial match supported)
        """
        if self.df is None:
            raise RuntimeError("Call fit() first.")

        mask = self.df["Movie_Name"].str.lower().str.contains(
            movie_title.lower(), na=False
        )
        matches = self.df[mask]

        if matches.empty:
            print(f"No movie found matching '{movie_title}'.")
            return pd.DataFrame()

        # Use the first matching movie's cleaned storyline as the query
        idx        = matches.index[0]
        title_used = self.df.loc[idx, "Movie_Name"]
        print(f"Using movie: '{title_used}'")

        input_vector      = self.tfidf_matrix[idx]
        similarity_scores = cosine_similarity(input_vector, self.tfidf_matrix)[0]
        similarity_scores[idx] = -1  # exclude the query movie itself

        top_indices = similarity_scores.argsort()[::-1][:self.top_n]

        results = []
        for rank, i in enumerate(top_indices, start=1):
            results.append({
                "Rank"             : rank,
                "Movie_Name"       : self.df.loc[i, "Movie_Name"],
                "Storyline"        : self.df.loc[i, "Storyline"],
                "Similarity_Score" : round(float(similarity_scores[i]), 4),
            })

        return pd.DataFrame(results)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str = "recommender.pkl"):
        """Pickle the fitted model for reuse in the Streamlit app."""
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"Model saved to '{path}'")

    @staticmethod
    def load(path: str = "recommender.pkl") -> "MovieRecommender":
        """Load a previously saved model."""
        with open(path, "rb") as f:
            model = pickle.load(f)
        print(f"Model loaded from '{path}'")
        return model


# ── Evaluation helper ─────────────────────────────────────────────────────────

def display_recommendations(results: pd.DataFrame):
    """Pretty-print the recommendations table."""
    if results.empty:
        print("No recommendations found.")
        return

    print("\n── Top Recommendations ──────────────────────────────────────────")
    for _, row in results.iterrows():
        print(f"\n  #{row['Rank']}  {row['Movie_Name']}  "
              f"(score: {row['Similarity_Score']})")
        storyline = row["Storyline"]
        if isinstance(storyline, str) and storyline.strip():
            preview = storyline[:150].rstrip() + ("…" if len(storyline) > 150 else "")
            print(f"      {preview}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    INPUT_FILE  = "preprocessed_movies.csv"
    MODEL_FILE  = "recommender.pkl"

    print("Phase 3 — TF-IDF Recommendation Engine\n")

    download_nltk_resources()

    # 1. Load preprocessed data
    print(f"Loading '{INPUT_FILE}'…")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"ERROR: '{INPUT_FILE}' not found. Run preprocessor.py first.")
        return

    required = {"Movie_Name", "Storyline", "Cleaned_Storyline"}
    if not required.issubset(df.columns):
        print(f"ERROR: CSV must have columns {required}. Found: {list(df.columns)}")
        return

    df.dropna(subset=["Cleaned_Storyline"], inplace=True)
    df = df[df["Cleaned_Storyline"].str.strip() != ""]
    df.reset_index(drop=True, inplace=True)
    print(f"  {len(df)} usable movies loaded\n")

    # 2. Build and fit the recommender
    recommender = MovieRecommender(
        max_features=5000,
        ngram_range=(1, 2),
        top_n=5,
    )
    recommender.fit(df)

    # 3. Save the model (used by app.py in Phase 5)
    recommender.save(MODEL_FILE)

    # 4. Demo — recommend by custom storyline
    test_storyline = (
        "A young hero discovers magical powers and must battle a dark villain "
        "to save the world, with the help of loyal friends."
    )
    print(f"\nTest query:\n  \"{test_storyline}\"\n")
    results = recommender.recommend(test_storyline)
    display_recommendations(results)

    # 5. Demo — recommend by existing movie title
    print("Recommend movies similar to 'Dune':")
    results2 = recommender.recommend_by_title("Dune")
    display_recommendations(results2)


if __name__ == "__main__":
    main()
