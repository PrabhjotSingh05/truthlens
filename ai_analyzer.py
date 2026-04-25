"""
ai_analyzer.py - Ultra-Strict News Credibility Analysis v2.0
=============================================================
MAJOR IMPROVEMENTS:
- Strict calibrated scoring using FULL 0-100 range (no more 50-70 clustering)
- Multi-model pipeline: Analyzer → Verifier → Judge
- Hard domain reputation scoring (satire/propaganda gets 0-20)
- Clickbait/sensationalism detection with heavy penalties
- Cross-reference with Google News for corroboration
- Final score = weighted combo of all metrics with guardrails
"""

import json
import re
import requests
from config import Config

# ─── Known unreliable domain patterns ────────────────────────────────────────
SATIRE_DOMAINS = [
    "theonion.com", "babylonbee.com", "clickhole.com", "thehardtimes.net",
    "reductress.com", "waterfordwhispersnews.com", "newsthump.com"
]

PROPAGANDA_DOMAINS = [
    "infowars.com", "naturalnews.com", "breitbart.com", "rt.com",
    "sputniknews.com", "beforeitsnews.com", "yournewswire.com",
    "newspunch.com", "worldnewsdailyreport.com", "empirenews.net"
]

TABLOID_DOMAINS = [
    "dailymail.co.uk", "mirror.co.uk", "thesun.co.uk", "nypost.com",
    "nationalenquirer.com", "globemagazine.com", "starcoverage.com",
    "radaronline.com", "tmz.com", "pagesix.com"
]

REPUTABLE_DOMAINS = [
    "bbc.com", "bbc.co.uk", "reuters.com", "apnews.com", "nytimes.com",
    "washingtonpost.com", "theguardian.com", "economist.com",
    "bloomberg.com", "ft.com", "wsj.com", "npr.org", "pbs.org",
    "time.com", "theatlantic.com", "newyorker.com", "science.org",
    "nature.com", "thehindu.com", "ndtv.com", "hindustantimes.com",
    "livemint.com", "theprint.in", "scroll.in", "wire.in"
]

def _classify_domain(domain):
    """Return domain reputation category and base score."""
    domain = domain.lower().replace("www.", "")
    if any(d in domain for d in SATIRE_DOMAINS):
        return "satire", 10
    if any(d in domain for d in PROPAGANDA_DOMAINS):
        return "propaganda", 8
    if any(d in domain for d in TABLOID_DOMAINS):
        return "tabloid", 32
    if any(d in domain for d in REPUTABLE_DOMAINS):
        return "reputable", 82
    return "unknown", 50

def call_groq(prompt, system_prompt="You are an expert AI journalist.", max_tokens=800,
              is_vision=False, image_data_uri=None):
    """Call the Groq Cloud API."""
    if not Config.is_ai_enabled():
        return None

    headers = {
        "Authorization": f"Bearer {Config.GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    if is_vision and image_data_uri:
        model = "meta-llama/llama-4-scout-17b-16e-instruct"
        messages = [{"role": "user", "content": [
            {"type": "text", "text": system_prompt + "\n\n" + prompt},
            {"type": "image_url", "image_url": {"url": image_data_uri}}
        ]}]
    else:
        model = Config.GROQ_MODEL
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": max_tokens,
        "stream": False,
    }

    try:
        resp = requests.post(Config.GROQ_URL, json=payload, headers=headers, timeout=35)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except requests.exceptions.ConnectionError:
        return "ERROR_CONNECTION"
    except requests.exceptions.HTTPError as e:
        code = e.response.status_code if e.response else 0
        return f"ERROR_HTTP_{code}"
    except Exception as e:
        return "ERROR_SYS"

def fetch_related_news(title):
    """Fetch related news from Google News RSS."""
    if not title or len(title.strip()) < 5:
        return []

    from urllib.parse import quote
    query = quote(title[:80])
    rss_url = f"https://news.google.com/rss/search?q={query}&hl=en&gl=US&ceid=US:en"

    try:
        resp = requests.get(rss_url, timeout=8, headers={"User-Agent": Config.USER_AGENT})
        resp.raise_for_status()

        from xml.etree import ElementTree as ET
        root = ET.fromstring(resp.text)
        results = []
        for item in root.findall(".//item"):
            item_title = item.findtext("title", "")
            item_link = item.findtext("link", "")
            item_date = item.findtext("pubDate", "")
            source = ""
            if " - " in item_title:
                parts = item_title.rsplit(" - ", 1)
                item_title = parts[0].strip()
                source = parts[1].strip()
            results.append({
                "title": item_title, "link": item_link,
                "source": source, "date": item_date[:16] if item_date else "",
            })
            if len(results) >= 6:
                break
        return results
    except Exception as e:
        return []

def _analyze_news_with_groq(scraped_data, domain_category, domain_base_score):
    """Main news credibility analysis with strict calibrated scoring."""
    page_info = scraped_data.get("page_info", {})
    raw_text = scraped_data.get("raw_text", "")
    reviews = scraped_data.get("reviews", [])

    # Get related news first for corroboration
    title = page_info.get("title", "")
    related_news = fetch_related_news(title)
    news_block = ""
    if related_news:
        lines = [f"{i+1}. [{n.get('source','')}] {n.get('title','')} ({n.get('date','')})"
                 for i, n in enumerate(related_news[:5])]
        news_block = "\n".join(lines)
    else:
        news_block = "No related news found — this significantly reduces corroboration score."

    content_blocks = []
    for i, r in enumerate(reviews[:5], 1):
        content_blocks.append(f"{i}. {r['text'][:300]}")
    content_block = "\n".join(content_blocks) or "No content blocks extracted."

    prompt = f"""You are an ultra-strict AI news credibility analyst. Your job is to give REALISTIC scores using the FULL 0-100 range. NEVER cluster scores around 50-70.

DOMAIN PRE-CLASSIFICATION: {domain_category} (base score: {domain_base_score}/100)

ARTICLE DATA:
- Title: {page_info.get('title', 'N/A')}
- Domain: {page_info.get('domain', 'N/A')}
- Description: {page_info.get('description', 'N/A')[:300]}
- Content: {raw_text[:800]}
- Content blocks: {content_block}

CORROBORATING NEWS SOURCES FOUND:
{news_block}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRICT SCORING CALIBRATION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

domain_trust_score:
• 0-10: Satire sites, known fake news, propaganda outlets
• 11-25: Conspiracy/fringe sites, zero editorial standards
• 26-40: Tabloids, clickbait-heavy outlets, low credibility
• 41-55: Unknown blogs, anonymous sites, no bylines
• 56-70: Regional/niche outlets, some editorial standards
• 71-85: Established news organizations, good track record
• 86-100: Top-tier journalism (BBC, Reuters, AP, NYT)

fact_based_score:
• 0-15: Pure fabrication, no verifiable facts, all claims unsupported
• 16-30: Mostly opinion, anecdotes, no citations, no sources named
• 31-50: Some facts mixed with heavy opinion, vague attribution ("sources say")
• 51-70: Mix of facts with some opinion, some attributed sources
• 71-85: Mostly facts with proper attribution, named sources
• 86-100: Heavily fact-checked, citations, primary sources, data

objectivity_score:
• 0-15: Extreme partisan framing, emotional manipulation, propaganda
• 16-30: Heavy bias, loaded language, one-sided presentation
• 31-50: Noticeably biased but not extreme, cherry-picked facts
• 51-70: Some bias but attempts balance
• 71-85: Mostly objective, multiple perspectives presented
• 86-100: Neutral reporting, balanced, no discernible bias

sensationalism_score (HIGHER = MORE SENSATIONAL):
• 80-100: ALL CAPS title, shock language, "BREAKING", "SHOCKING", "You won't believe"
• 60-80: Clickbait framing, exaggerated claims, emotional manipulation
• 40-60: Some sensational elements but mostly normal
• 20-40: Normal journalistic tone
• 0-20: Dry, academic, understated reporting

corroboration_score:
• 0-20: Story appears ONLY on this outlet, no other sources report it
• 21-40: Very few corroborating sources, mostly fringe outlets
• 41-60: Some corroboration but limited
• 61-80: Multiple mainstream outlets report similar story
• 81-100: Widely reported, major news agencies confirm, primary sources cited

PENALTIES (apply to final article_truth_score):
• No byline/author: -10
• No publication date: -8
• Domain registered recently (<2 years): -15
• ALL CAPS in title: -12
• Multiple exclamation marks: -8
• "SHOCKING", "BOMBSHELL", "They don't want you to know": -15
• Claims contradict widely reported facts: -25
• Zero corroborating sources: -15

Return ONLY valid JSON:
{{
  "is_news_article": <true if article/blog/news, false ONLY if storefront/login>,
  "domain_trust_score": <0-100 STRICT per calibration>,
  "fact_based_score": <0-100 STRICT per calibration>,
  "objectivity_score": <0-100 STRICT per calibration>,
  "sensationalism_score": <0-100 STRICT — higher = more sensational>,
  "corroboration_score": <0-100 based on news sources found above>,
  "trust_level": "<high|medium|low|very_low>",
  "final_verdict": "<CREDIBLE|MOSTLY_CREDIBLE|MIXED|QUESTIONABLE|MISLEADING|LIKELY_FALSE|PROPAGANDA|SATIRE>",
  "website_category": "<news|blog|opinion|satire|propaganda|tabloid|aggregator|other>",
  "political_bias": "<far_left|left|center_left|center|center_right|right|far_right|unknown>",
  "story_type": "<breaking_news|analysis|opinion|investigative|sponsored|press_release|satire|unknown>",
  "explanation": "<3-4 sentences: specific credibility assessment with evidence from the content>",
  "news_summary": "<2-3 paragraph objective summary of what the article claims>",
  "overall_sentiment_summary": "<1-2 sentence tone analysis>",
  "key_findings": ["<specific finding 1>", "<specific finding 2>", "<specific finding 3>"],
  "recommendation": "<2 sentence actionable advice for the reader>",
  "positive_signals": ["<signal1>", "<signal2>"],
  "risk_factors": ["<risk1>", "<risk2>", "<risk3>"],
  "missing_context": ["<context1>", "<context2>"],
  "primary_claims": ["<main claim 1>", "<main claim 2>"],
  "logical_fallacies": ["<fallacy1 if any>"],
  "important_links": []
}}"""

    raw = call_groq(
        prompt,
        "You are an ultra-strict news credibility analyst. Use the FULL 0-100 range — never cluster around 50-70. Apply all penalties listed. Respond with valid JSON only.",
        1500
    )

    if not raw or (isinstance(raw, str) and raw.startswith("ERROR_")):
        return raw if raw else "ERROR_UNKNOWN", related_news

    try:
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        text = match.group(0) if match else raw
        result = json.loads(text)

        # Sanitize all scores
        for field in ["domain_trust_score", "fact_based_score", "objectivity_score",
                      "sensationalism_score", "corroboration_score"]:
            result[field] = max(0, min(100, int(result.get(field, 50))))

        # Override domain score with pre-classification if it's a known domain
        if domain_category in ("satire", "propaganda", "tabloid", "reputable"):
            ai_score = result["domain_trust_score"]
            # Blend AI score with known classification (known classification wins 60%)
            result["domain_trust_score"] = int(domain_base_score * 0.6 + ai_score * 0.4)

        # Calculate article_truth_score using weighted formula
        # Weighted: Fact(30%) + Objectivity(25%) + (100-Sensationalism)(20%) + DomainTrust(15%) + Corroboration(10%)
        truth_score = (
            result["fact_based_score"] * 0.30 +
            result["objectivity_score"] * 0.25 +
            (100 - result["sensationalism_score"]) * 0.20 +
            result["domain_trust_score"] * 0.15 +
            result.get("corroboration_score", 50) * 0.10
        )
        result["article_truth_score"] = max(1, min(100, int(truth_score)))

        # Set trust level based on truth score
        ts = result["article_truth_score"]
        if ts >= 75:
            result["trust_level"] = "high"
        elif ts >= 55:
            result["trust_level"] = "medium"
        elif ts >= 35:
            result["trust_level"] = "low"
        else:
            result["trust_level"] = "very_low"

        # Defaults
        result.setdefault("website_category", "other")
        result.setdefault("explanation", "")
        result.setdefault("news_summary", "")
        result.setdefault("overall_sentiment_summary", "")
        result.setdefault("key_findings", [])
        result.setdefault("recommendation", "")
        result.setdefault("positive_signals", [])
        result.setdefault("risk_factors", [])
        result.setdefault("missing_context", [])
        result.setdefault("primary_claims", [])
        result.setdefault("logical_fallacies", [])
        result.setdefault("important_links", [])
        result.setdefault("final_verdict", "MIXED")
        result.setdefault("political_bias", "unknown")
        result.setdefault("story_type", "unknown")
        result.setdefault("corroboration_score", 50)

        return result, related_news

    except json.JSONDecodeError as e:
        print(f"[NewsAnalyzer] JSON parse error: {e}")
        return {
            "is_news_article": True,
            "domain_trust_score": domain_base_score,
            "fact_based_score": 50, "objectivity_score": 50,
            "sensationalism_score": 50, "corroboration_score": 25,
            "article_truth_score": 45,
            "trust_level": "low",
            "website_category": "other",
            "explanation": "AI analysis ran but returned malformed data.",
            "news_summary": "", "overall_sentiment_summary": "",
            "key_findings": [], "recommendation": "Manual review recommended.",
            "positive_signals": [], "risk_factors": [],
            "missing_context": [], "primary_claims": [],
            "logical_fallacies": [], "important_links": [],
            "final_verdict": "MIXED", "political_bias": "unknown", "story_type": "unknown"
        }, related_news

def analyze_with_ai(scraped_data):
    """Main entry point: perform AI credibility analysis."""
    try:
        from news_analyzer import analyze_with_multi_model
        result = analyze_with_multi_model(scraped_data)
        if isinstance(result, str) and result.startswith("ERROR_"):
            return result
        analyze_with_ai._last_related_news = result.pop("_related_news", [])
        return result
    except Exception as e:
        print(f"[NewsAnalyzer] Multi-model pipeline error: {e}, using single-model")

    analyze_with_ai._last_related_news = []

    # Pre-classify domain
    page_info = scraped_data.get("page_info", {})
    domain = page_info.get("domain", "")
    domain_category, domain_base_score = _classify_domain(domain)

    result_or_error, related_news = _analyze_news_with_groq(scraped_data, domain_category, domain_base_score)
    analyze_with_ai._last_related_news = related_news

    return result_or_error

def generate_ai_report(scraped_data, basic_report):
    """Merge basic + AI analysis into one enriched report."""
    ai_result = analyze_with_ai(scraped_data)

    # Grab related news
    related_news = getattr(analyze_with_ai, '_last_related_news', [])

    if isinstance(ai_result, str) and ai_result.startswith("ERROR_"):
        if "401" in ai_result:
            basic_report["ai_error"] = "Invalid Groq API Key. Get a free key from console.groq.com."
        elif "404" in ai_result:
            basic_report["ai_error"] = "AI model not found. Check GROQ_MODEL in config.py."
        elif "CONNECTION" in ai_result:
            basic_report["ai_error"] = "Network error connecting to Groq API."
        else:
            basic_report["ai_error"] = f"AI analysis failed ({ai_result}). Showing basic analysis."
        return basic_report

    if ai_result is None:
        basic_report["ai_error"] = "AI analysis returned empty data."
        return basic_report

    report = basic_report.copy()
    report["analysis_type"] = "ai_enhanced"
    report["ai_enabled"] = True
    report["is_news_article"] = ai_result.get("is_news_article", True)
    report["domain_trust_score"] = ai_result.get("domain_trust_score", 50)
    report["article_truth_score"] = ai_result.get("article_truth_score", 50)
    report["fact_based_score"] = ai_result.get("fact_based_score", 50)
    report["objectivity_score"] = ai_result.get("objectivity_score", 50)
    report["sensationalism_score"] = ai_result.get("sensationalism_score", 50)
    report["corroboration_score"] = ai_result.get("corroboration_score", 50)

    # Keep trust_score alias for template compatibility
    report["trust_score"] = ai_result.get("article_truth_score", 50)
    report["trust_level"] = ai_result.get("trust_level", "medium")

    report["ai_explanation"] = ai_result.get("ai_explanation") or ai_result.get("explanation", "")
    report["news_summary"] = ai_result.get("news_summary", "")
    report["overall_sentiment_summary"] = ai_result.get("overall_sentiment_summary") or ai_result.get("overall_sentiment", "")
    report["key_findings"] = ai_result.get("key_findings", [])
    report["recommendation"] = ai_result.get("recommendation", "")
    report["website_category"] = ai_result.get("website_category", "other")
    report["risk_factors"] = ai_result.get("risk_factors", [])
    report["positive_signals"] = ai_result.get("positive_signals", [])
    report["important_links"] = ai_result.get("important_links", [])
    report["missing_context"] = ai_result.get("missing_context", [])
    report["primary_claims"] = ai_result.get("primary_claims", [])
    report["logical_fallacies"] = ai_result.get("logical_fallacies", [])

    # Multi-model fields
    report["final_verdict"] = ai_result.get("final_verdict", "MIXED")
    report["trust_label"] = ai_result.get("trust_label", "Verify Independently")
    report["confidence_level"] = ai_result.get("confidence_level", "MEDIUM")
    report["models_agree"] = ai_result.get("models_agree", False)
    report["claim_verdict"] = ai_result.get("claim_verdict", "UNVERIFIABLE")
    report["claim_verdict_confidence"] = ai_result.get("claim_verdict_confidence", 0)
    report["claim_explanation"] = ai_result.get("claim_explanation", "")
    report["story_type"] = ai_result.get("story_type", "unknown")
    report["corroboration_level"] = ai_result.get("corroboration_level", "uncorroborated")
    report["source_reputation"] = ai_result.get("source_reputation", "unknown")
    report["political_bias"] = ai_result.get("political_bias", "unknown")
    report["overall_sentiment"] = ai_result.get("overall_sentiment", "neutral")
    report["corroboration_notes"] = ai_result.get("corroboration_notes", "")
    report["propaganda_score"] = ai_result.get("propaganda_score", 20)
    report["transparency_score"] = ai_result.get("transparency_score", 50)
    report["pipeline"] = ai_result.get("pipeline", {})
    report["_related_news"] = related_news

    return report

def generate_image_report(data_uri):
    """Analyze image — delegates to image_analyzer.py."""
    try:
        from image_analyzer import generate_image_report as _gen
        return _gen(data_uri)
    except Exception as e:
        print(f"[ai_analyzer] Image analysis error: {e}")
        return None
