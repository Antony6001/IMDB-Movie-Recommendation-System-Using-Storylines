"""
Phase 1 — IMDb 2024 Movie Scraper
Scrapes movie names and storylines from IMDb and saves to movies.csv

Requirements:
"""

import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager


# ── Configuration ─────────────────────────────────────────────────────────────

BASE_URL = (
    "https://www.imdb.com/search/title/"
    "?title_type=feature&release_date=2024-01-01,2024-12-31"
    "&sort=num_votes,desc&count=50"
)
MAX_PAGES   = 5          # each page has 50 movies → ~250 movies total
OUTPUT_FILE = "movies.csv"
SCROLL_PAUSE = 2         # seconds to wait after scroll / page load


# ── Driver setup ──────────────────────────────────────────────────────────────

def get_driver() -> webdriver.Chrome:
    """Return a headless Chrome WebDriver."""
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
    service = Service(ChromeDriverManager().install())
    driver  = webdriver.Chrome(service=service, options=options)
    driver.implicitly_wait(10)
    return driver


# ── Scraping helpers ──────────────────────────────────────────────────────────

def get_movie_links(driver: webdriver.Chrome, page_url: str) -> list[str]:
    """Return all movie detail-page URLs found on a search-results page."""
    driver.get(page_url)
    time.sleep(SCROLL_PAUSE)

    # Scroll to bottom so lazy-loaded items appear
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(SCROLL_PAUSE)

    anchors = driver.find_elements(
        By.CSS_SELECTOR,
        "a.ipc-title-link-wrapper"   # title links on the search results page
    )
    links = []
    for a in anchors:
        href = a.get_attribute("href")
        if href and "/title/tt" in href:
            # Normalise to bare title URL (strip query params)
            clean = href.split("?")[0]
            if clean not in links:
                links.append(clean)
    return links


def scrape_movie_detail(driver: webdriver.Chrome, url: str) -> dict | None:
    """
    Visit a single IMDb movie page and extract:
      - Movie name
      - Storyline / plot summary
    Returns a dict or None if extraction fails.
    """
    try:
        driver.get(url)
        time.sleep(1.5)

        # ── Movie name ──────────────────────────────────────────────────────
        name = ""
        try:
            name_el = WebDriverWait(driver, 8).until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, "h1[data-testid='hero__pageTitle'] span")
                )
            )
            name = name_el.text.strip()
        except Exception:
            # Fallback selector
            try:
                name = driver.find_element(
                    By.CSS_SELECTOR, "h1.sc-ec65ba05-0"
                ).text.strip()
            except Exception:
                pass

        if not name:
            print(f"  [skip] Could not find title on {url}")
            return None

        # ── Storyline ───────────────────────────────────────────────────────
        storyline = ""
        selectors = [
            "span[data-testid='plot-xl']",          # full plot on detail page
            "span[data-testid='plot-l']",
            "span.sc-2d37a7c7-2",                   # alternate selector
            "div[data-testid='storyline-plot-summary'] span",
        ]
        for sel in selectors:
            try:
                el = driver.find_element(By.CSS_SELECTOR, sel)
                text = el.text.strip()
                if text:
                    storyline = text
                    break
            except Exception:
                continue

        if not storyline:
            print(f"  [warn] No storyline found for '{name}'")

        return {"Movie_Name": name, "Storyline": storyline}

    except Exception as e:
        print(f"  [error] {url} → {e}")
        return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Starting IMDb 2024 movie scraper…\n")
    driver  = get_driver()
    records = []

    try:
        for page_num in range(MAX_PAGES):
            # IMDb paginates via &start= offset (1-based, step of 50)
            start     = page_num * 50 + 1
            page_url  = f"{BASE_URL}&start={start}"
            print(f"── Page {page_num + 1} ({page_url}) ──")

            links = get_movie_links(driver, page_url)
            print(f"   Found {len(links)} movie links")

            for idx, link in enumerate(links, 1):
                print(f"   [{idx}/{len(links)}] {link}")
                data = scrape_movie_detail(driver, link)
                if data and data["Storyline"]:
                    records.append(data)
                time.sleep(1)  # polite delay between requests

    finally:
        driver.quit()

    # ── Save ─────────────────────────────────────────────────────────────────
    if not records:
        print("\nNo records collected. Check your selectors or connection.")
        return

    df = pd.DataFrame(records).drop_duplicates(subset="Movie_Name")
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    print(f"\nDone! {len(df)} movies saved to '{OUTPUT_FILE}'")
    print(df.head())


if __name__ == "__main__":
    main()
