"""
analyzer.py - Basic Analysis Module (No API Key Required)
=========================================================
Performs sentiment analysis, duplicate detection, and rating
calculations using simple Python logic — no external AI needed.
"""

import re
from collections import Counter
from utils import get_word_frequency


# --- Sentiment Word Lists ---
# These are simple keyword lists for basic sentiment analysis
POSITIVE_WORDS = {
    'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
    'awesome', 'love', 'perfect', 'best', 'happy', 'pleased', 'satisfied',
    'recommend', 'impressive', 'outstanding', 'superb', 'brilliant',
    'beautiful', 'quality', 'reliable', 'fast', 'easy', 'helpful',
    'friendly', 'comfortable', 'enjoy', 'worth', 'nice', 'top',
    'favorite', 'genuine', 'trustworthy', 'authentic', 'legitimate',
    'professional', 'efficient', 'smooth', 'solid', 'durable',
}

NEGATIVE_WORDS = {
    'bad', 'terrible', 'horrible', 'awful', 'worst', 'hate', 'poor',
    'disappointed', 'disappointing', 'waste', 'useless', 'broken',
    'slow', 'expensive', 'cheap', 'fake', 'scam', 'fraud', 'ripoff',
    'avoid', 'never', 'return', 'refund', 'complain', 'complaint',
    'damage', 'defective', 'misleading', 'overpriced', 'unreliable',
    'suspicious', 'dishonest', 'deceptive', 'spam', 'junk', 'rubbish',
    'problem', 'issue', 'error', 'fail', 'failed', 'failure',
}


def analyze_sentiment(reviews):
    """
    Analyze the sentiment of each review using keyword matching.

    Args:
        reviews (list): List of review dicts with "text" key

    Returns:
        dict: {
            "positive": int,
            "negative": int,
            "neutral": int,
            "details": list of {text, sentiment, score}
        }
    """
    results = {"positive": 0, "negative": 0, "neutral": 0, "details": []}

    for review in reviews:
        text = review.get("text", "").lower()
        words = set(re.findall(r'[a-zA-Z]+', text))

        pos_count = len(words & POSITIVE_WORDS)
        neg_count = len(words & NEGATIVE_WORDS)

        if pos_count > neg_count:
            sentiment = "positive"
            results["positive"] += 1
        elif neg_count > pos_count:
            sentiment = "negative"
            results["negative"] += 1
        else:
            sentiment = "neutral"
            results["neutral"] += 1

        score = pos_count - neg_count
        results["details"].append({
            "text": review["text"][:150] + ("..." if len(review["text"]) > 150 else ""),
            "sentiment": sentiment,
            "score": score,
            "author": review.get("author", "Anonymous"),
        })

    return results


def detect_duplicates(reviews):
    """
    Detect repeated or near-duplicate reviews.

    Args:
        reviews (list): List of review dicts

    Returns:
        dict: {
            "duplicate_count": int,
            "duplicate_groups": list of lists,
            "has_duplicates": bool
        }
    """
    # Normalize texts for comparison
    normalized = {}
    for i, review in enumerate(reviews):
        # Simple normalization: lowercase, remove extra spaces
        norm = re.sub(r'\s+', ' ', review["text"].lower().strip())
        # Also create a "fingerprint" by removing punctuation
        fingerprint = re.sub(r'[^a-z0-9 ]', '', norm)

        if fingerprint in normalized:
            normalized[fingerprint].append(i)
        else:
            normalized[fingerprint] = [i]

    # Find groups with more than one review
    duplicate_groups = []
    duplicate_count = 0
    for fingerprint, indices in normalized.items():
        if len(indices) > 1:
            group = [reviews[i]["text"][:100] + "..." for i in indices]
            duplicate_groups.append(group)
            duplicate_count += len(indices) - 1  # Extra copies

    return {
        "duplicate_count": duplicate_count,
        "duplicate_groups": duplicate_groups,
        "has_duplicates": duplicate_count > 0,
    }


def calculate_avg_rating(reviews):
    """
    Calculate the average star rating from reviews.

    Args:
        reviews (list): List of review dicts with optional "rating" key

    Returns:
        dict: {
            "average": float or None,
            "count": int (reviews with ratings),
            "distribution": dict (star → count)
        }
    """
    ratings = []
    distribution = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

    for review in reviews:
        rating = review.get("rating")
        if rating is not None:
            try:
                r = float(rating)
                if 0 <= r <= 5:
                    ratings.append(r)
                    # Round to nearest integer for distribution
                    star = max(1, min(5, round(r)))
                    distribution[star] += 1
            except (ValueError, TypeError):
                continue

    if ratings:
        avg = round(sum(ratings) / len(ratings), 1)
    else:
        avg = None

    return {
        "average": avg,
        "count": len(ratings),
        "distribution": distribution,
    }


def generate_basic_report(scraped_data):
    """
    Generate a complete basic analysis report.
    This is the main function called when no API key is available.

    Args:
        scraped_data (dict): Output from scraper.scrape_url()

    Returns:
        dict: Complete analysis report
    """
    reviews = scraped_data.get("reviews", [])

    # Run all analyses
    sentiment = analyze_sentiment(reviews)
    duplicates = detect_duplicates(reviews)
    ratings = calculate_avg_rating(reviews)
    word_freq = get_word_frequency([r["text"] for r in reviews], top_n=15)

    # Calculate a basic trust indicator (not a full AI trust score)
    basic_trust = calculate_basic_trust(sentiment, duplicates, ratings, len(reviews))

    return {
        "analysis_type": "basic",
        "ai_enabled": False,
        "review_count": len(reviews),
        "sentiment": sentiment,
        "duplicates": duplicates,
        "ratings": ratings,
        "word_frequency": word_freq,
        "basic_trust": basic_trust,
        "trust_score": None,  # Only available with AI
        "ai_explanation": None,
        "fake_reviews": None,
    }


def calculate_basic_trust(sentiment, duplicates, ratings, review_count):
    """
    Calculate a basic trust indicator based on available data.
    This is NOT the AI trust score — just a simple heuristic.

    Returns:
        dict: {
            "level": "high"/"medium"/"low"/"insufficient_data",
            "reasons": list of strings,
            "score_estimate": int (rough 0-100 estimate)
        }
    """
    reasons = []
    score = 50  # Start neutral

    if review_count == 0:
        return {
            "level": "insufficient_data",
            "reasons": ["No reviews found to analyze"],
            "score_estimate": None,
        }

    total = sentiment["positive"] + sentiment["negative"] + sentiment["neutral"]

    # Factor 1: Sentiment balance
    if total > 0:
        pos_ratio = sentiment["positive"] / total
        neg_ratio = sentiment["negative"] / total
        if pos_ratio > 0.7:
            score += 15
            reasons.append("Mostly positive sentiment detected")
        elif neg_ratio > 0.5:
            score -= 20
            reasons.append("High proportion of negative reviews")
        else:
            reasons.append("Mixed sentiment in reviews")

    # Factor 2: Duplicates
    if duplicates["has_duplicates"]:
        penalty = min(30, duplicates["duplicate_count"] * 10)
        score -= penalty
        reasons.append(f"Found {duplicates['duplicate_count']} duplicate review(s) — suspicious")
    else:
        score += 10
        reasons.append("No duplicate reviews detected")

    # Factor 3: Ratings
    if ratings["average"] is not None:
        if ratings["average"] >= 4.0:
            score += 10
            reasons.append(f"Good average rating: {ratings['average']}/5")
        elif ratings["average"] < 2.5:
            score -= 15
            reasons.append(f"Low average rating: {ratings['average']}/5")

    # Factor 4: Review count
    if review_count >= 10:
        score += 5
        reasons.append(f"Reasonable number of reviews ({review_count})")
    elif review_count < 3:
        reasons.append("Very few reviews — limited data")

    # Clamp score
    score = max(0, min(100, score))

    # Determine level
    if score >= 70:
        level = "high"
    elif score >= 40:
        level = "medium"
    else:
        level = "low"

    return {"level": level, "reasons": reasons, "score_estimate": score}
