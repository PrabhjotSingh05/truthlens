"""
utils.py - Shared Utility Functions
====================================
Helper functions used across multiple modules in TrustScanner AI.
These are small, reusable functions that keep other files clean.
"""

import re
from urllib.parse import urlparse
from collections import Counter


def is_valid_url(url):
    """
    Check if a string is a valid URL.

    Args:
        url (str): The URL string to validate

    Returns:
        bool: True if the URL is valid, False otherwise

    Example:
        is_valid_url("https://www.amazon.com/product/123")  → True
        is_valid_url("not a url")                           → False
    """
    try:
        result = urlparse(url)
        # A valid URL must have both a scheme (http/https) and a network location (domain)
        return all([result.scheme in ("http", "https"), result.netloc])
    except Exception:
        return False


def clean_text(text):
    """
    Clean and normalize text by removing extra whitespace and special characters.

    Args:
        text (str): Raw text to clean

    Returns:
        str: Cleaned text
    """
    if not text:
        return ""
    # Remove extra whitespace (multiple spaces, tabs, newlines → single space)
    text = re.sub(r'\s+', ' ', text)
    # Remove leading/trailing whitespace
    text = text.strip()
    return text


def extract_domain(url):
    """
    Extract the domain name from a URL.

    Args:
        url (str): Full URL

    Returns:
        str: Domain name (e.g., "www.amazon.com")
    """
    try:
        parsed = urlparse(url)
        return parsed.netloc
    except Exception:
        return "Unknown"


def is_https(url):
    """
    Check if a URL uses HTTPS (secure connection).

    Args:
        url (str): URL to check

    Returns:
        bool: True if HTTPS, False otherwise
    """
    try:
        return urlparse(url).scheme == "https"
    except Exception:
        return False


def get_word_frequency(texts, top_n=20):
    """
    Get the most common words from a list of texts.
    Useful for generating word cloud data.

    Args:
        texts (list): List of text strings
        top_n (int): Number of top words to return

    Returns:
        list: List of (word, count) tuples
    """
    # Common words to ignore (stop words)
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
        'for', 'of', 'with', 'by', 'is', 'it', 'this', 'that', 'was',
        'are', 'were', 'be', 'been', 'has', 'had', 'have', 'do', 'does',
        'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can',
        'not', 'no', 'so', 'if', 'as', 'just', 'than', 'then', 'too',
        'very', 'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'she',
        'they', 'them', 'its', 'his', 'her', 'their', 'what', 'which',
        'who', 'when', 'where', 'how', 'all', 'each', 'every', 'both',
        'few', 'more', 'most', 'other', 'some', 'such', 'only', 'own',
        'same', 'from', 'about', 'up', 'out', 'into', 'over', 'after',
    }

    all_words = []
    for text in texts:
        # Split text into words, convert to lowercase
        words = re.findall(r'[a-zA-Z]+', text.lower())
        # Filter out stop words and very short words
        words = [w for w in words if w not in stop_words and len(w) > 2]
        all_words.extend(words)

    # Count word frequency and return top N
    return Counter(all_words).most_common(top_n)


def truncate_text(text, max_length=200):
    """
    Truncate text to a maximum length, adding "..." if truncated.

    Args:
        text (str): Text to truncate
        max_length (int): Maximum character count

    Returns:
        str: Truncated text
    """
    if not text or len(text) <= max_length:
        return text
    return text[:max_length].rsplit(' ', 1)[0] + "..."


def check_suspicious_patterns(url, page_text=""):
    """
    Check for suspicious patterns in a URL or page content.
    This is a basic heuristic check — not foolproof.

    Args:
        url (str): The URL to check
        page_text (str): The page content text

    Returns:
        list: List of warning messages (empty = no warnings)
    """
    warnings = []

    # Check URL patterns
    if not is_https(url):
        warnings.append("⚠️ Website does not use HTTPS (not secure)")

    domain = extract_domain(url)

    # Check for IP-based URLs (suspicious)
    if re.match(r'\d+\.\d+\.\d+\.\d+', domain):
        warnings.append("⚠️ URL uses an IP address instead of a domain name")

    # Check for very long URLs (sometimes used in phishing)
    if len(url) > 200:
        warnings.append("⚠️ Unusually long URL detected")

    # Check for suspicious keywords in page content
    suspicious_keywords = [
        'act now', 'limited time', 'guaranteed winner',
        'click here immediately', 'too good to be true',
        'wire transfer', 'nigerian prince', 'congratulations you won'
    ]

    page_lower = page_text.lower()
    for keyword in suspicious_keywords:
        if keyword in page_lower:
            warnings.append(f"⚠️ Suspicious phrase detected: \"{keyword}\"")

    return warnings
