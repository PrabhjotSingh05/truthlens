"""
scraper.py - Web Scraping Module
=================================
Uses BeautifulSoup to parse HTML and extract page info and reviews.
"""

import re
import json
import requests
from bs4 import BeautifulSoup
from config import Config
from utils import clean_text, extract_domain


def fetch_page(url):
    """Fetch HTML content of a webpage using a smart 4-tier cascading fallback strategy."""
    
    # STRATEGY 1: Standard Modern Browser
    try:
        headers = {
            "User-Agent": Config.USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }
        response = requests.get(url, headers=headers, timeout=Config.REQUEST_TIMEOUT)
        response.raise_for_status()
        return True, response.text
    except Exception as e1:
        status1 = e1.response.status_code if hasattr(e1, 'response') and e1.response is not None else 0
        
        if status1 in (404, 400, 500, 502, 503, 504):
            return False, f"Website returned a standard HTTP {status1} error. Please check the URL."
            
        # STRATEGY 2: Minimalist Headers (Bypasses WAFs that flag header mismatches, e.g. NDTV)
        print(f"[TrustScanner] S1 Blocked (HTTP {status1}). Trying Strategy 2: Minimalist Headers...")
        try:
            minimal_headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            }
            fb_resp = requests.get(url, headers=minimal_headers, timeout=Config.REQUEST_TIMEOUT)
            fb_resp.raise_for_status()
            return True, fb_resp.text
        except Exception as e2:
            status2 = e2.response.status_code if hasattr(e2, 'response') and e2.response is not None else 0
            
            # STRATEGY 3: Googlebot Spoof (Bypasses basic paywalls, but blocked by strict Cloudflare)
            print(f"[TrustScanner] S2 Blocked (HTTP {status2}). Trying Strategy 3: Googlebot Spoof...")
            try:
                bot_headers = {
                    "User-Agent": "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)",
                }
                fb_resp2 = requests.get(url, headers=bot_headers, timeout=Config.REQUEST_TIMEOUT)
                fb_resp2.raise_for_status()
                return True, fb_resp2.text
            except Exception as e3:
                status3 = e3.response.status_code if hasattr(e3, 'response') and e3.response is not None else 0
                
                # STRATEGY 4: Deep Urllib Bypass (Switches underlying TLS library from requests to urllib)
                print(f"[TrustScanner] S3 Blocked (HTTP {status3}). Trying Strategy 4: Deep Urllib Engine Switch...")
                try:
                    import urllib.request
                    req = urllib.request.Request(
                        url,
                        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
                    )
                    with urllib.request.urlopen(req, timeout=Config.REQUEST_TIMEOUT + 5) as final_resp:
                        html = final_resp.read().decode('utf-8', errors='ignore')
                        return True, html
                except Exception as e4:
                    return False, f"Error: Could not access this specific website. It is aggressively blocking scrapers across all 4 bypass layers (Final HTTP Status: {status3})."


def extract_page_info(soup, url):
    """Extract title, description, domain, link/image counts."""
    title_tag = soup.find("title")
    title = clean_text(title_tag.get_text()) if title_tag else "No title found"
    
    # MSN dynamically loads everything, leaving the title as just "MSN"
    if "msn" in title.lower() and "msn.com" in url:
        match = re.search(r'msn\.com/.*?/([^/]+)/ar-', url)
        if match:
            title = match.group(1).replace('-', ' ').title()

    description = ""
    for meta in [
        soup.find("meta", attrs={"name": "description"}),
        soup.find("meta", attrs={"property": "og:description"}),
    ]:
        if meta:
            description = clean_text(meta.get("content", ""))
            if description:
                break
    if not description:
        first_p = soup.find("p")
        if first_p:
            description = clean_text(first_p.get_text())[:300]

    # Count links on the page and analyze them for trust signals
    all_links = soup.find_all("a", href=True)
    
    # Productive Trust Signals
    has_privacy_policy = False
    has_terms = False
    social_links = {"facebook": None, "twitter": None, "instagram": None, "linkedin": None, "youtube": None, "github": None}
    has_contact_info = False

    for link in all_links:
        href_lower = link.get('href', '').lower()
        href_raw = link.get('href', '')
        text = link.get_text().lower()
        
        if 'privacy' in href_lower or 'privacy' in text:
            has_privacy_policy = True
        if 'terms' in href_lower or 'terms' in text or 'tos' in href_lower:
            has_terms = True
        if 'mailto:' in href_lower or 'tel:' in href_lower or 'contact' in href_lower or 'contact' in text:
            has_contact_info = True
            
        if 'facebook.com' in href_lower and not social_links['facebook']:
            social_links['facebook'] = href_raw
        if ('twitter.com' in href_lower or 'x.com' in href_lower) and not social_links['twitter']:
            social_links['twitter'] = href_raw
        if 'instagram.com' in href_lower and not social_links['instagram']:
            social_links['instagram'] = href_raw
        if 'linkedin.com' in href_lower and not social_links['linkedin']:
            social_links['linkedin'] = href_raw
        if 'youtube.com' in href_lower and not social_links['youtube']:
            social_links['youtube'] = href_raw
        if 'github.com' in href_lower and not social_links['github']:
            social_links['github'] = href_raw

    # Count images
    all_images = soup.find_all("img")
    image_count = len(all_images)

    return {
        "title": title,
        "description": description or "No description available",
        "domain": extract_domain(url),
        "url": url,
        "link_count": len(all_links),
        "image_count": image_count,
        "trust_signals": {
            "has_privacy_policy": has_privacy_policy,
            "has_terms": has_terms,
            "has_contact_info": has_contact_info,
            "social_links": social_links
        }
    }


def extract_rating_from_element(element):
    """Try to find a numerical rating inside an element."""
    # Check aria-label
    rated = element.find(attrs={"aria-label": True})
    if rated:
        match = re.search(r'(\d+\.?\d*)\s*(?:out of|\/)\s*(\d+)', rated.get("aria-label", ""))
        if match:
            try:
                return round((float(match.group(1)) / float(match.group(2))) * 5, 1)
            except ValueError:
                pass
    # Check text for "X/5" pattern
    match = re.search(r'(\d+\.?\d*)\s*(?:out of|\/)\s*5', element.get_text())
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    return None


def extract_author_from_element(element):
    """Try to find reviewer name inside an element."""
    author_el = element.find(class_=lambda x: x and any(
        k in str(x).lower() for k in ["author", "reviewer", "user-name", "username"]
    ))
    if author_el:
        name = clean_text(author_el.get_text())
        if 2 < len(name) < 50:
            return name
    return "Anonymous"


def extract_reviews(soup):
    """Extract review-like content using multiple strategies."""
    reviews = []

    # Strategy 1: Look for review-related CSS classes
    containers = soup.find_all(
        ["div", "section", "article", "li", "p"],
        class_=lambda x: x and any(
            k in str(x).lower() for k in ["review", "comment", "testimonial", "feedback"]
        )
    )
    for c in containers[:Config.MAX_REVIEWS]:
        text = clean_text(c.get_text())
        if len(text) < 20:
            continue
        if len(text) > 2000:
            text = text[:2000] + "..."
        reviews.append({
            "text": text,
            "rating": extract_rating_from_element(c),
            "author": extract_author_from_element(c),
        })

    # Strategy 2: JSON-LD structured data
    if not reviews:
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                data = json.loads(script.string)
                items = data if isinstance(data, list) else [data]
                for item in items:
                    if item.get("@type") == "Review":
                        reviews.append({
                            "text": item.get("reviewBody", item.get("description", "")),
                            "rating": item.get("reviewRating", {}).get("ratingValue"),
                            "author": item.get("author", {}).get("name", "Anonymous") if isinstance(item.get("author"), dict) else "Anonymous",
                        })
                    for r in (item.get("review", []) if isinstance(item.get("review"), list) else []):
                        if r.get("@type") == "Review":
                            reviews.append({
                                "text": r.get("reviewBody", ""),
                                "rating": r.get("reviewRating", {}).get("ratingValue"),
                                "author": r.get("author", {}).get("name", "Anonymous") if isinstance(r.get("author"), dict) else "Anonymous",
                            })
            except (json.JSONDecodeError, TypeError, AttributeError):
                continue

    # Strategy 3: Fallback to paragraphs
    if not reviews:
        for p in soup.find_all("p")[:20]:
            text = clean_text(p.get_text())
            if 30 < len(text) < 1000:
                reviews.append({"text": text, "rating": None, "author": "Unknown"})

    # Deduplicate
    seen = set()
    unique = []
    for r in reviews:
        if r["text"] not in seen:
            seen.add(r["text"])
            unique.append(r)
    return unique[:Config.MAX_REVIEWS]


def scrape_url(url):
    """
    Main entry point. Fetches URL, parses HTML, extracts info + reviews.
    Returns dict with success, error, page_info, reviews, raw_text.
    """
    success, result = fetch_page(url)
    if not success:
        return {"success": False, "error": result, "page_info": None, "reviews": [], "raw_text": ""}

    # Parse HTML using the built-in html.parser (no extra C++ compilers required)
    soup = BeautifulSoup(result, "html.parser")

    page_info = extract_page_info(soup, url)
    reviews = extract_reviews(soup)

    # Get clean page text for AI analysis
    for tag in soup(["script", "style", "nav", "footer", "header", "svg", "button", "form", "aside", "iframe"]):
        tag.decompose()
    raw_text = clean_text(soup.get_text())[:5000]

    return {"success": True, "error": None, "page_info": page_info, "reviews": reviews, "raw_text": raw_text}
