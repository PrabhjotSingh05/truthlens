# TrustScanner AI 🔍

A college-level AI project that analyzes websites and product pages to determine trustworthiness. It scrapes reviews, performs sentiment analysis, and uses Google Gemini AI (optional) to detect fake reviews and generate a 0-100 trust score.

![TrustScanner AI](https://via.placeholder.com/800x400.png?text=TrustScanner+AI+Preview)

## 🌟 Features

*   **Intelligent Web Scraping:** Uses BeautifulSoup to extract page metadata and user reviews using multiple extraction strategies (CSS matching, JSON-LD, paragraph heuristics).
*   **Basic Analysis (Always On):**
    *   Keyword-based Sentiment Analysis (Positive/Neutral/Negative)
    *   Duplicate Review Detection
    *   Average Rating Calculation
    *   Word Frequency Extraction
*   **AI-Powered Analysis (Optional):**
    *   Uses Google Gemini 2.0 API.
    *   Detects potentially fake or AI-generated reviews.
    *   Calculates a robust 0-100 Trust Score.
    *   Provides natural language explanations for its findings.
*   **Graceful Fallback:** The project **will not crash** if an API key is missing. It smoothly falls back to basic analysis, making it extremely beginner-friendly to test.
*   **Beautiful UI:** Modern dark theme with glassmorphism, responsive grids, and interactive SVG gauge animations.

## 🛠️ Tech Stack

*   **Backend:** Python 3, Flask
*   **Web Scraping:** `requests`, `beautifulsoup4`, `lxml`
*   **AI Integration:** `google-generativeai`
*   **Frontend:** HTML5, Vanilla CSS3 (Custom Design System), Vanilla JS

---

## 🚀 Setup Instructions (Step-by-Step)

Follow these instructions to run the project on your local machine.

### Prerequisites
*   Python 3.8 or higher installed on your computer.
*   (Optional) A free Google Gemini API Key.

### 1. Clone or Download the Project
Ensure you have the project folder `TrustScanner` on your computer. Open your terminal (or Command Prompt) and navigate into the folder:
```bash
cd path/to/TrustScanner
```

### 2. Create a Virtual Environment (Recommended)
It's best practice to install Python packages in an isolated environment.
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
Install all required libraries using the `requirements.txt` file.
```bash
pip install -r requirements.txt
```

### 4. Setup Environment Variables (Optional but recommended)
To unlock the AI features, you need to provide an API key.
1. Copy the `.env.example` file and rename the copy to exactly `.env`.
2. Open `.env` in a text editor.
3. Replace `your_gemini_api_key_here` with your actual Gemini API key (Get one free at [Google AI Studio](https://aistudio.google.com/app/apikey)).

*Note: If you skip this step, the app will still work, but AI features will be disabled.*

### 5. Run the Application
Start the Flask development server.
```bash
python app.py
```

### 6. View in Browser
Open your web browser and go to:
[http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 📂 Code Structure

*   `app.py` — The main Flask server and route definitions.
*   `config.py` — Handles environment variables and central settings safely.
*   `scraper.py` — The engine that fetches URLs and parses HTML to find reviews.
*   `analyzer.py` — Standard NLP logic (sentiment, duplicates, ratings) that runs without an API.
*   `ai_analyzer.py` — The bridge to Google Gemini for advanced trust scoring.
*   `utils.py` — Shared helper functions (URL validation, text cleaning).
*   `templates/` — Contains HTML files (`index.html`, `results.html`, `about.html`, `base.html`).
*   `static/` — Contains CSS (`style.css`) and JavaScript (`main.js`).

## 🎓 Academic Integrity Note
This project was designed for educational purposes. Web scraping should be performed responsibly and in accordance with the target website's `robots.txt` and Terms of Service.
