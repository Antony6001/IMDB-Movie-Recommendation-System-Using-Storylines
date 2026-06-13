
import pickle
import re

import nltk
import pandas as pd
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from recommender import MovieRecommender


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="CineMatch — Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=DM+Sans:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Page background */
.stApp {
    background: #0d0d12;
    color: #e8e6df;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #131319;
    border-right: 1px solid #2a2a35;
}

/* Header */
.main-header {
    text-align: center;
    padding: 2.5rem 0 1.5rem 0;
}
.main-header h1 {
    font-family: 'Playfair Display', serif;
    font-size: 3.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #f5c842 0%, #e8845a 60%, #c45fa0 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    letter-spacing: -1px;
}
.main-header p {
    color: #7a7a8a;
    font-size: 1.05rem;
    margin-top: 0.5rem;
}

/* Section label */
.section-label {
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #f5c842;
    margin-bottom: 0.5rem;
}

/* Textarea override */
textarea {
    background: #1a1a24 !important;
    border: 1px solid #2e2e3d !important;
    border-radius: 10px !important;
    color: #e8e6df !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
}
textarea:focus {
    border-color: #f5c842 !important;
    box-shadow: 0 0 0 2px rgba(245,200,66,0.15) !important;
}

/* Buttons */
div.stButton > button {
    background: linear-gradient(135deg, #f5c842, #e8845a);
    color: #0d0d12;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    font-size: 0.95rem;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 2.2rem;
    cursor: pointer;
    transition: opacity 0.2s;
    width: 100%;
}
div.stButton > button:hover { opacity: 0.88; }

/* Movie cards */
.movie-card {
    background: #16161f;
    border: 1px solid #252530;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    transition: border-color 0.2s, transform 0.15s;
    position: relative;
    overflow: hidden;
}
.movie-card:hover {
    border-color: #f5c842;
    transform: translateY(-2px);
}
.movie-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 4px; height: 100%;
    background: linear-gradient(180deg, #f5c842, #e8845a);
    border-radius: 4px 0 0 4px;
}
.card-rank {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 700;
    color: #2a2a35;
    position: absolute;
    top: 1rem; right: 1.4rem;
    line-height: 1;
}
.card-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.2rem;
    font-weight: 700;
    color: #f5f3ec;
    margin-bottom: 0.4rem;
    padding-right: 2.5rem;
}
.score-badge {
    display: inline-block;
    background: rgba(245,200,66,0.12);
    color: #f5c842;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    padding: 0.2rem 0.7rem;
    border-radius: 20px;
    border: 1px solid rgba(245,200,66,0.25);
    margin-bottom: 0.75rem;
}
.card-storyline {
    color: #8a8a9a;
    font-size: 0.9rem;
    line-height: 1.65;
}

/* Stat pills in sidebar */
.stat-pill {
    background: #1e1e2a;
    border: 1px solid #2a2a38;
    border-radius: 8px;
    padding: 0.6rem 1rem;
    margin-bottom: 0.5rem;
    font-size: 0.85rem;
    color: #9a9aaa;
}
.stat-pill span {
    color: #f5c842;
    font-weight: 500;
}

/* Divider */
.gold-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #f5c842, transparent);
    margin: 1.5rem 0;
    opacity: 0.3;
}

/* Empty state */
.empty-state {
    text-align: center;
    padding: 3rem 1rem;
    color: #3a3a4a;
}
.empty-state .icon { font-size: 3rem; margin-bottom: 1rem; }
.empty-state p { font-size: 1rem; }

/* Input label */
.input-hint {
    color: #5a5a6a;
    font-size: 0.82rem;
    margin-top: 0.4rem;
}
</style>
""", unsafe_allow_html=True)


# ── NLTK setup ────────────────────────────────────────────────────────────────

st.cache_resource(show_spinner=False)
def download_nltk():
    for r in ["punkt", "stopwords", "wordnet", "omw-1.4", "punkt_tab"]:
        nltk.download(r, quiet=True)

download_nltk()


# ── Load model ────────────────────────────────────────────────────────────────

st.cache_resource(show_spinner=False)
def load_model(path: str = "recommender.pkl"):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None


model = load_model()


# ── Text cleaning (mirrors Phase 2) ──────────────────────────────────────────

def clean_text(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = word_tokenize(text)
    sw = set(stopwords.words("english"))
    lem = WordNetLemmatizer()
    tokens = [lem.lemmatize(t) for t in tokens if t not in sw and len(t) >= 2]
    return " ".join(tokens)


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🎬 CineMatch")
    st.markdown("<div class='gold-divider'></div>", unsafe_allow_html=True)

    search_mode = st.radio(
        "Search mode",
        ["By storyline", "By movie Name"],
        index=0,
    )

    st.markdown("<div class='gold-divider'></div>", unsafe_allow_html=True)

    top_n = st.slider("Number of recommendations", 3, 10, 5)
    if model:
        model.top_n = top_n

    st.markdown("<div class='gold-divider'></div>", unsafe_allow_html=True)

    # Dataset stats
    if model and model.df is not None:
        df = model.df
        st.markdown("**Dataset stats**")
        st.markdown(
            f"<div class='stat-pill'>Movies indexed: <span>{len(df)}</span></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div class='stat-pill'>Vocabulary size: "
            f"<span>{len(model.vectorizer.vocabulary_):,}</span></div>",
            unsafe_allow_html=True,
        )
        avg_words = df["Cleaned_Storyline"].apply(lambda x: len(str(x).split())).mean()
        st.markdown(
            f"<div class='stat-pill'>Avg plot words: <span>{avg_words:.0f}</span></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div class='gold-divider'></div>", unsafe_allow_html=True)
    st.caption("Built with Selenium · NLTK · TF-IDF · Streamlit")


# ── Main header ───────────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1>CineMatch</h1>
    <p>Discover movies that match your story — powered by NLP &amp; cosine similarity</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<div class='gold-divider'></div>", unsafe_allow_html=True)


# ── Model not found ───────────────────────────────────────────────────────────

if model is None:
    st.error(
        "**Model not found.**  \n"
        "Run `python recommender.py` first to generate `recommender.pkl`, "
        "then restart this app."
    )
    st.stop()


# ── Search UI ─────────────────────────────────────────────────────────────────

results = pd.DataFrame()
query_used = ""

if search_mode == "By storyline":
    st.markdown("<div class='section-label'>Describe a plot</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        user_input = st.text_area(
            label="storyline_input",
            label_visibility="collapsed",
            placeholder=(
                "e.g. A young orphan discovers he has magical powers and is "
                "invited to a school for wizards, where he makes friends and "
                "must face a dark enemy from his past…"
            ),
            height=130,
        )
        st.markdown(
            "<div class='input-hint'>Tip: The more detail you add, the better the matches.</div>",
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        find_btn = st.button("Find movies", key="storyline_btn")

    if find_btn:
        if not user_input.strip():
            st.warning("Please enter a storyline first.")
        else:
            with st.spinner("Searching for similar movies…"):
                results = model.recommend(user_input)
                query_used = user_input

else:
    st.markdown("<div class='section-label'>Search by movie name</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        title_input = st.text_input(
            label="title_input",
            label_visibility="collapsed",
            placeholder="e.g. Dune, Oppenheimer, Inside Out…",
        )
    with col2:
        find_btn2 = st.button("Find similar", key="title_btn")

    if find_btn2:
        if not title_input.strip():
            st.warning("Please enter a movie title.")
        else:
            with st.spinner("Finding similar movies…"):
                results = model.recommend_by_title(title_input)
                query_used = title_input

    # Show available titles as reference
    if model.df is not None:
        with st.expander("Browse all indexed movies"):
            search_filter = st.text_input("Filter titles", placeholder="Type to search…")
            titles = model.df["Movie_Name"].sort_values().tolist()
            if search_filter:
                titles = [t for t in titles if search_filter.lower() in t.lower()]
            for t in titles[:60]:
                st.markdown(f"- {t}")
            if len(titles) > 60:
                st.caption(f"…and {len(titles) - 60} more")


# ── Results ───────────────────────────────────────────────────────────────────

st.markdown("<div class='gold-divider'></div>", unsafe_allow_html=True)

if results.empty and query_used == "":
    st.markdown("""
    <div class="empty-state">
        <div class="icon">🎞️</div>
        <p>Enter a storyline or movie title above to get recommendations.</p>
    </div>
    """, unsafe_allow_html=True)

elif results.empty:
    st.markdown("""
    <div class="empty-state">
        <div class="icon">🔍</div>
        <p>No matches found. Try a different title or a longer storyline.</p>
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown(
        f"<div class='section-label'>Top {len(results)} recommendations</div>",
        unsafe_allow_html=True,
    )

    for _, row in results.iterrows():
        score_pct = int(row["Similarity_Score"] * 100)
        storyline = str(row.get("Storyline", "")).strip()
        preview   = (storyline[:220] + "…") if len(storyline) > 220 else storyline

        st.markdown(f"""
        <div class="movie-card">
            <div class="card-rank">#{int(row['Rank'])}</div>
            <div class="card-title">{row['Movie_Name']}</div>
            <div class="score-badge">Match score: {score_pct}%</div>
            <div class="card-storyline">{preview}</div>
        </div>
        """, unsafe_allow_html=True)

    # Download results
    st.markdown("<br>", unsafe_allow_html=True)
    csv = results[["Rank", "Movie_Name", "Similarity_Score", "Storyline"]].to_csv(index=False)
    st.download_button(
        label="Download results as CSV",
        data=csv,
        file_name="recommendations.csv",
        mime="text/csv",
    )
