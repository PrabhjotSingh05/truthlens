"""
app.py - Flask Main Application
================================
TruthLens AI — News Credibility Analyzer
Powered by Groq Cloud API (Llama 3) for fast, free analysis.
Works locally AND on Google Colab.
"""

import os
import sys

# Fix Unicode output on Windows (emoji in print statements) without breaking live logs
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

from flask import Flask, render_template, request, jsonify
from config import Config
from scraper import scrape_url
from analyzer import generate_basic_report
from ai_analyzer import generate_ai_report, fetch_related_news
from utils import is_valid_url, check_suspicious_patterns
import base64
from database import check_and_increment_usage
import json

# Removed Firebase Admin SDK

# Admin passcode — set ADMIN_PASSCODE env var in Render, never hardcode
ADMIN_PASSCODE = os.getenv("ADMIN_PASSCODE", "")

app = Flask(__name__)
app.config["SECRET_KEY"] = Config.SECRET_KEY


# ==========================================
#  ROUTES
# ==========================================

@app.route("/")
def index():
    """Home page."""
    return render_template("index.html")


@app.route("/about")
def about():
    """About page."""
    return render_template("about.html")

@app.route("/image_check")
def image_check():
    """Image Fact-Check Home Page."""
    return render_template("image_check.html")




@app.route("/scan", methods=["POST"])
def scan():
    """
    Main scan endpoint.
    Scrapes URL, runs basic + AI analysis, renders results.
    """
    url = request.form.get("url", "").strip()

    if not url:
        return render_template("index.html", error="Please enter a URL to scan.")

    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    if not is_valid_url(url):
        return render_template("index.html", error="Invalid URL. Please enter a valid website address.")

    # Check Usage Limits
    device_id = request.form.get("deviceId", "").strip()
    
    # Admin Override (passcode set via ADMIN_PASSCODE env var on Render)
    if ADMIN_PASSCODE and device_id == ADMIN_PASSCODE:
        pass # Allow unlimited
    else:
        allowed, reason = check_and_increment_usage(device_id=device_id, scan_type="news")
        if not allowed:
            return render_template("index.html", error=reason)

    # Step 1: Scrape
    scraped_data = scrape_url(url)
    if not scraped_data["success"]:
        return render_template("index.html", error=f"Could not scrape the website: {scraped_data['error']}")

    # Step 2: Basic analysis
    report = generate_basic_report(scraped_data)

    # Step 3: AI analysis (Groq Cloud)
    if Config.is_ai_enabled():
        report = generate_ai_report(scraped_data, report)
        
        if not report.get("is_news_article", True):
            return render_template("index.html", error="Error: This does not appear to be a news article or informational story. Please submit a specific article link.")

    # Step 4: Suspicious pattern warnings
    warnings = check_suspicious_patterns(url, scraped_data.get("raw_text", ""))

    # Related news comes from multi-model pipeline (already fetched)
    related_news = report.pop("_related_news", [])

    # Step 6: Render
    return render_template("results.html",
        page_info=scraped_data["page_info"],
        reviews=scraped_data["reviews"],
        report=report,
        warnings=warnings,
        related_news=related_news,
        url=url)


@app.route("/api/scan", methods=["POST"])
def api_scan():
    """JSON API endpoint for scanning."""
    data = request.get_json()
    url = data.get("url", "").strip() if data else ""

    if not url:
        return jsonify({"error": "No URL provided"}), 400
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    if not is_valid_url(url):
        return jsonify({"error": "Invalid URL"}), 400

    scraped_data = scrape_url(url)
    if not scraped_data["success"]:
        return jsonify({"error": scraped_data["error"]}), 500

    report = generate_basic_report(scraped_data)
    if Config.is_ai_enabled():
        report = generate_ai_report(scraped_data, report)

    warnings = check_suspicious_patterns(url, scraped_data.get("raw_text", ""))

    return jsonify({
        "page_info": scraped_data["page_info"],
        "report": report,
        "warnings": warnings,
        "review_count": len(scraped_data["reviews"]),
    })


@app.route("/scan_image", methods=["POST"])
def scan_image():
    """Image analysis endpoint."""
    if 'image' not in request.files:
        return render_template("image_check.html", error="No image uploaded.")
        
    file = request.files['image']
    if file.filename == '':
        return render_template("image_check.html", error="No selected file.")
        
    # Check Usage Limits
    device_id = request.form.get("deviceId", "").strip()
    
    # Admin Override (passcode set via ADMIN_PASSCODE env var on Render)
    if ADMIN_PASSCODE and device_id == ADMIN_PASSCODE:
        pass # Allow unlimited
    else:
        allowed, reason = check_and_increment_usage(device_id=device_id, scan_type="image")
        if not allowed:
            return render_template("image_check.html", error=reason)
        
    try:
        # Read file and encode to base64
        image_bytes = file.read()
        base64_img = base64.b64encode(image_bytes).decode('utf-8')
        
        # Determine mime type
        mime_type = file.mimetype if file.mimetype else 'image/jpeg'
        if mime_type not in ['image/jpeg', 'image/png', 'image/webp', 'image/gif']:
            return render_template("image_check.html", error="Unsupported image format. Please upload JPG, PNG, or WEBP.")
            
        data_uri = f"data:{mime_type};base64,{base64_img}"
        
        # Multi-model image intelligence pipeline
        from image_analyzer import generate_image_report
        report = generate_image_report(data_uri)
        
        if not report:
            return render_template("image_check.html", error="AI analysis failed or returned empty data.")
            
        return render_template("image_results.html", report=report, image_data_uri=data_uri)
        
    except Exception as e:
        return render_template("image_check.html", error=f"Error processing image: {str(e)}")

# ==========================================
#  ERROR HANDLERS
# ==========================================

@app.errorhandler(404)
def page_not_found(e):
    return render_template("index.html", error="Page not found."), 404


@app.errorhandler(500)
def internal_error(e):
    return render_template("index.html", error="An internal error occurred. Please try again."), 500


# ==========================================
#  RUN
# ==========================================

if __name__ == "__main__":
    print("=" * 50)
    print("  TruthLens AI -- Starting Up")
    print("=" * 50)
    print(f"  {Config.get_status_message()}")
    print(f"  Server: http://127.0.0.1:5000")
    print("=" * 50)
    app.run(debug=Config.DEBUG, host="127.0.0.1", port=5000)
