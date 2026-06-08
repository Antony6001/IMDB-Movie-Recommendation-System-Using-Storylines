# 🎬 CineMatch — IMDb Movie Recommendation System

A content-based movie recommendation system that scrapes IMDb 2024 movies, processes their storylines using NLP, and recommends similar movies using TF-IDF and Cosine Similarity — served through an interactive Streamlit web app.

---

## 📌 Project Overview

This project builds an end-to-end movie recommendation engine:

1. **Scrapes** movie names and plot summaries from IMDb (2024) using Selenium
2. **Cleans and preprocesses** storylines using NLP techniques (NLTK)
3. **Vectorizes** text using TF-IDF and computes Cosine Similarity
4. **Recommends** the top 5 most similar movies for any input storyline
5. **Serves** results through an interactive Streamlit UI

---

## 🗂️ Project Structure

```
imdb-movie-recommender/
│
├── scraper.py               # Phase 1 — Selenium scraping from IMDb
├── preprocessor.py          # Phase 2 — NLP text cleaning pipeline
├── recommender.py           # Phase 3 — TF-IDF vectorization + cosine similarity
├── app.py                   # Phase 5 — Streamlit web application
│
├── movies.csv               # Raw scraped data (auto-generated)
├── preprocessed_movies.csv  # Cleaned storylines (auto-generated)
├── recommender.pkl          # Trained model (auto-generated)
│
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

---

## 🔧 Tech Stack

| Category | Tools / Libraries |
|---|---|
| Web Scraping | Selenium, webdriver-manager |
| Data Handling | Pandas |
| NLP | NLTK (tokenization, stopwords, lemmatization) |
| ML / Vectorization | Scikit-learn (TF-IDF, Cosine Similarity) |
| Web Framework | Streamlit |
| Language | Python 3.10+ |

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/imdb-movie-recommender.git
cd imdb-movie-recommender
```

### 2. Create a virtual environment (recommended)


venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```
pip install -r requirements.txt
```

---

## 🚀 How to Run

Execute the scripts in order:

### Step 1 — Scrape IMDb data

```
python scraper.py
```

Scrapes movie names and storylines from IMDb 2024 and saves them to `movies.csv`.

> **Note:** Scraping ~250 movies takes 5–10 minutes. Make sure Google Chrome is installed.

---

### Step 2 — Preprocess the data

```bash
python preprocessor.py
```

Cleans and tokenizes the storylines using NLP. Saves the result to `preprocessed_movies.csv`.

**NLP pipeline applied:**
- Lowercase conversion
- HTML tag and URL removal
- Punctuation removal
- Tokenization
- Stopword removal
- Lemmatization
- Short word removal

---

### Step 3 — Build the recommendation model

```bash
python recommender.py
```

Fits TF-IDF vectorizer on cleaned storylines, computes the similarity matrix, and saves the trained model to `recommender.pkl`.

**Sample console output:**
```
Fitting TF-IDF on 241 movies…
  Vocabulary size : 4,872
  Matrix shape    : (241, 5000)
Model saved to 'recommender.pkl'
```

---

### Step 4 — Launch the Streamlit app

```bash
streamlit run app.py
```

Opens in your browser at `http://localhost:8501`

---

## 🎯 How to Use the App

**Search by storyline:**
1. Select "By storyline" in the sidebar
2. Type or paste a plot description in the text area
3. Click **Find movies**
4. View the top 5 recommended movies with match scores and plot previews

**Search by title:**
1. Select "By movie title" in the sidebar
2. Type a movie name (e.g. `Dune`, `Inside Out`)
3. Click **Find similar**
4. Get movies with similar storylines from the dataset

**Download results:**
Click the **Download results as CSV** button to save recommendations.

---

## 🖥️ App Screenshot

> *![alt text](<Screenshot 2026-06-08 153041.png>)*

---

## 📊 How It Works

```
User Input (storyline)
        │
        ▼
  Text Cleaning (NLP)
        │
        ▼
  TF-IDF Vectorization   ←── Trained on 241 movie storylines
        │
        ▼
  Cosine Similarity Score
        │
        ▼
  Top 5 Movies Ranked by Score
```

**Cosine Similarity** measures the angle between two story vectors in a 5000-dimensional word space. A score of `1.0` means identical storylines; `0.0` means completely unrelated.

---

## 📁 Dataset

| Column | Description |
|---|---|
| `Movie_Name` | Title of the movie |
| `Storyline` | Raw plot summary scraped from IMDb |
| `Cleaned_Storyline` | Preprocessed text (used for vectorization) |

- **Source:** IMDb 2024 movies page
- **Size:** ~250 movies
- **Format:** CSV

---

## 📦 requirements.txt

```
selenium
webdriver-manager
pandas
nltk
scikit-learn
streamlit
```

Generate it yourself with:
```
pip freeze > requirements.txt
```

---

## 🔍 Example

**Input storyline:**
> "A young orphan discovers he has magical powers and is invited to attend a school for wizards, where he makes loyal friends and must face a dark enemy from his past."

**Output — Top 5 Recommendations:**

| Rank | Movie Name | Match Score |
|---|---|---|
| #1 | The Wizard's Journey | 78% |
| #2 | The Magic Academy | 65% |
| #3 | The Dark Sorcerer | 61% |
| #4 | Legends of the Lost Kingdom | 54% |
| #5 | The Enchanted Forest | 49% |

---

## 🛠️ Troubleshooting

| Issue | Fix |
|---|---|
| `chromedriver` error | Install Google Chrome and run `pip install webdriver-manager` |
| `recommender.pkl not found` | Run `python recommender.py` before launching the app |
| IMDb blocks scraper | Add longer delays in `scraper.py` or use a VPN |
| NLTK resource error | Run `python -c "import nltk; nltk.download('all')"` |
| Streamlit not found | Run `pip install streamlit` |

---

## 📈 Future Improvements

- Add movie posters using the IMDb / TMDB API
- Include genre, director, and cast as additional features
- Deploy on Streamlit Cloud or Hugging Face Spaces
- Add user ratings and collaborative filtering
- Support multilingual storylines

---

## 👤 Author

**ANTHONY R**
- GitHub: [@Antony6001](https://github.com/Antony6001)
- LinkedIn: [ANTHONY R](www.linkedin.com/in/anthony-r-183296144)

---

## 📄 License

This project is for educational purposes only. Movie data is sourced from IMDb and is subject to IMDb's [Terms of Service](https://www.imdb.com/conditions).

---

## 🙏 Acknowledgements

- [IMDb](https://www.imdb.com) for movie data
- [Streamlit](https://streamlit.io) for the web framework
- [Scikit-learn](https://scikit-learn.org) for TF-IDF and cosine similarity
- [NLTK](https://www.nltk.org) for NLP preprocessing


[def]: <Screenshot 2026-06-08 153041.png>