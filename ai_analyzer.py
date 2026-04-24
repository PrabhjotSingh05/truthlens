"""
ai_analyzer.py - Groq Cloud AI Analysis + Google News Fetching
===============================================================
Uses Groq Cloud API (free Llama 3) for credibility analysis.
Also fetches related news from Google News RSS for cross-referencing.
"""

import json
import re
import requests
from config import Config


def call_groq(prompt, system_prompt="You are an expert AI journalist.", max_tokens=800, is_vision=False, image_data_uri=None):
    """Call the Groq Cloud API. Returns response text or error string."""
    if not Config.is_ai_enabled():
        print("[TrustScanner] AI is disabled (no GROQ_API_KEY)")
        return None

    headers = {
        "Authorization": f"Bearer {Config.GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    
    if is_vision and image_data_uri:
        model = "meta-llama/llama-4-scout-17b-16e-instruct"
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": system_prompt + "\n\n" + prompt},
                {"type": "image_url", "image_url": {"url": image_data_uri}}
            ]}
        ]
    else:
        model = Config.GROQ_MODEL
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": max_tokens,
        "stream": False,
    }

    try:
        print(f"[TrustScanner] Calling Groq API ({Config.GROQ_MODEL})...")
        resp = requests.post(Config.GROQ_URL, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"].strip()
        print(f"[TrustScanner] Groq response received ({len(text)} chars)")
        return text
    except requests.exceptions.ConnectionError:
        print("[TrustScanner] Groq connection failed - check internet")
        return "ERROR_CONNECTION"
    except requests.exceptions.HTTPError as e:
        code = e.response.status_code if e.response is not None else 0
        body = e.response.text[:300] if e.response is not None else str(e)
        print(f"[TrustScanner] Groq HTTP {code}: {body}")
        return f"ERROR_HTTP_{code}"
    except Exception as e:
        print(f"[TrustScanner] Groq error: {e}")
        return "ERROR_SYS"


def fetch_related_news(title):
    """
    Fetch related news articles from Google News RSS feed.
    Returns a list of dicts: [{title, link, source, date}]
    """
    if not title or len(title.strip()) < 5:
        return []

    # Use Google News RSS search
    from urllib.parse import quote
    query = quote(title[:80])
    rss_url = f"https://news.google.com/rss/search?q={query}&hl=en&gl=US&ceid=US:en"

    try:
        print(f"[TrustScanner] Fetching related news for: {title[:50]}...")
        resp = requests.get(rss_url, timeout=8, headers={
            "User-Agent": Config.USER_AGENT
        })
        resp.raise_for_status()

        # Parse XML manually (no lxml needed)
        from xml.etree import ElementTree as ET
        root = ET.fromstring(resp.text)
        items = root.findall(".//item")

        results = []
        for item in items:
            item_title = item.findtext("title", "")
            item_link = item.findtext("link", "")
            item_date = item.findtext("pubDate", "")
            
            # Filter out fact checking sites as per user request
            lower_title = item_title.lower()
            if any(fc in lower_title for fc in ["fact check", "snopes", "politifact", "factcheck.org", "reuters fact check"]):
                continue
                
            source = ""
            if " - " in item_title:
                parts = item_title.rsplit(" - ", 1)
                item_title = parts[0].strip()
                source = parts[1].strip()
                
            results.append({
                "title": item_title,
                "link": item_link,
                "source": source,
                "date": item_date[:16] if item_date else "",
            })
            
            if len(results) >= 6:  # Keep top 6 standard news articles
                break
        return results

    except Exception as e:
        print(f"[TrustScanner] Google News fetch failed: {e}")
        return []


def analyze_with_ai(scraped_data):
    """Perform AI credibility analysis — routes to multi-model pipeline."""
    try:
        from news_analyzer import analyze_with_multi_model
        result = analyze_with_multi_model(scraped_data)
        if isinstance(result, str) and result.startswith("ERROR_"):
            return result
        # Store related news for app.py to pass to template
        analyze_with_ai._last_related_news = result.pop("_related_news", [])
        return result
    except Exception as e:
        print(f"[TrustScanner] Multi-model pipeline error: {e}, falling back to single-model")
        analyze_with_ai._last_related_news = []

    """[FALLBACK] Single-model analysis."""
    reviews = scraped_data.get("reviews", [])
    page_info = scraped_data.get("page_info", {})
    raw_text = scraped_data.get("raw_text", "")

    review_texts = []
    for i, r in enumerate(reviews[:5], 1):
        review_texts.append(f"{i}. [{r.get('author','Anon')}]: {r['text'][:250]}")
    reviews_block = "\n".join(review_texts) or "No content blocks."

    prompt = f"""You are a strict, unbiased AI news credibility analyst. Analyze this article and assign REALISTIC scores using the FULL 0-100 range.

SCORING RULES — READ CAREFULLY:
- Do NOT cluster scores around 50-70. Use the full range.
- Satire/parody/propaganda sites: domain_trust_score 0-20
- Tabloids/clickbait sites: domain_trust_score 20-45
- Average blogs/unknown sites: 40-60
- Established regional news: 60-75
- Major reputable outlets (BBC, Reuters, AP): 80-95
- fact_based_score: 0-20 if pure opinion/hearsay, 20-50 if mostly opinion with few facts, 50-75 if mixed, 75-100 if heavily cited/factual
- objectivity_score: 0-20 if clearly partisan/emotional, 20-45 if biased language, 45-70 if somewhat balanced, 70-100 if neutral reporting
- sensationalism_score: 80-100 if clickbait/ALL CAPS/shocking language, 50-80 if moderately sensational, 20-50 if normal reporting, 0-20 if very dry/academic

ARTICLE DATA:
- Title: {page_info.get('title', 'N/A')}
- Domain: {page_info.get('domain', 'N/A')}
- Description: {page_info.get('description', 'N/A')[:200]}
- Content: {raw_text[:600]}
- Blocks: {reviews_block}

Return ONLY valid JSON:
{{
  "is_news_article": <true if article/blog/news, false ONLY if storefront or login page>,
  "domain_trust_score": <0-100, use full range per calibration above>,
  "fact_based_score": <0-100, use full range per calibration above>,
  "objectivity_score": <0-100, use full range per calibration above>,
  "sensationalism_score": <0-100, use full range per calibration above>,
  "trust_level": "<high if combined>70 | medium if 40-70 | low if <40>",
  "website_category": "<news|blog|opinion|satire|propaganda|aggregator|other>",
  "explanation": "<3-4 sentence credibility analysis mentioning specific evidence>",
  "news_summary": "<2-3 paragraph summary of the news story and its factual basis>",
  "overall_sentiment_summary": "<1-2 sentence tone analysis>",
  "key_findings": ["<finding1>","<finding2>","<finding3>"],
  "recommendation": "<1-2 sentence actionable advice for the reader>",
  "positive_signals": ["<signal1>","<signal2>"],
  "risk_factors": ["<risk1>","<risk2>"],
  "important_links": []
}}"""

    raw = call_groq(prompt, "You are a strict news credibility analyst. Use the full 0-100 scoring range. Never default to middle scores. Respond with valid JSON only.", 1200)
    if not raw or (isinstance(raw, str) and raw.startswith("ERROR_")):
        print(f"[TrustScanner] AI call returned: {raw}")
        return raw if raw else "ERROR_UNKNOWN"

    try:
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        text = match.group(0) if match else raw
        result = json.loads(text)

        # Sanitize
        result["is_news_article"] = bool(result.get("is_news_article", True))
        result["domain_trust_score"] = max(0, min(100, int(result.get("domain_trust_score", 50))))
        result["fact_based_score"] = max(0, min(100, int(result.get("fact_based_score", 50))))
        result["objectivity_score"] = max(0, min(100, int(result.get("objectivity_score", 50))))
        result["sensationalism_score"] = max(0, min(100, int(result.get("sensationalism_score", 50))))
        
        # Calculate a combined Truth Score based on the new metrics
        # Formula: (Fact + Objectivity + (100-Sensationalism)) / 3
        combined_truth = (result["fact_based_score"] + result["objectivity_score"] + (100 - result["sensationalism_score"])) / 3
        result["article_truth_score"] = int(combined_truth)
        
        result.setdefault("trust_level", "medium")
        result.setdefault("website_category", "other")
        result.setdefault("explanation", "")
        result.setdefault("news_summary", "")
        result.setdefault("overall_sentiment_summary", "")
        result.setdefault("key_findings", [])
        result.setdefault("recommendation", "")
        result.setdefault("positive_signals", [])
        result.setdefault("risk_factors", [])
        result.setdefault("important_links", [])
        return result

    except json.JSONDecodeError as e:
        print(f"[TrustScanner] Bad JSON from AI: {e}")
        print(f"[TrustScanner] Raw: {raw[:300]}")
        return {
            "is_news_article": True,
            "domain_trust_score": 50, "fact_based_score": 50, "objectivity_score": 50, "sensationalism_score": 50, "article_truth_score": 50,
            "trust_level": "medium", "website_category": "other",
            "explanation": "AI analysis ran but returned malformed data.",
            "news_summary": "", "overall_sentiment_summary": "",
            "key_findings": [], "recommendation": "Manual review recommended.",
            "positive_signals": [], "risk_factors": [],
            "important_links": []
        }


def generate_ai_report(scraped_data, basic_report):
    """Merge basic + multi-model AI analysis into one enriched report."""
    ai_result = analyze_with_ai(scraped_data)
    # Grab related news stored by the pipeline
    related_news = getattr(analyze_with_ai, '_last_related_news', [])

    if isinstance(ai_result, str) and ai_result.startswith("ERROR_"):
        if "401" in ai_result:
            basic_report["ai_error"] = "Invalid Groq API Key! Please get a valid free key from console.groq.com and update your .env or config.py."
        elif "decommissioned" in ai_result.lower():
            basic_report["ai_error"] = "The configured AI model has been decommissioned by Groq. Please update GROQ_MODEL in config.py to 'llama-3.3-70b-versatile'."
        elif "404" in ai_result:
            basic_report["ai_error"] = "AI Model not found on Groq. Please check GROQ_MODEL in config.py."
        elif "CONNECTION" in ai_result:
            basic_report["ai_error"] = "Network error. Could not connect to the Groq API."
        else:
            basic_report["ai_error"] = f"AI analysis failed ({ai_result}). The system has defaulted to basic heuristic scanning."
        return basic_report

    if ai_result is None:
        basic_report["ai_error"] = "AI analysis returned empty data."
        return basic_report

    report = basic_report.copy()
    report["analysis_type"] = "ai_enhanced"
    report["ai_enabled"] = True
    report["is_news_article"] = ai_result.get("is_news_article", True)
    report["domain_trust_score"] = ai_result["domain_trust_score"]
    report["article_truth_score"] = ai_result["article_truth_score"]
    report["fact_based_score"] = ai_result["fact_based_score"]
    report["objectivity_score"] = ai_result["objectivity_score"]
    report["sensationalism_score"] = ai_result["sensationalism_score"]
    
    # Keep legacy trust_score for compatibility if needed elsewhere
    report["trust_score"] = ai_result["article_truth_score"]
    
    report["trust_level"] = ai_result["trust_level"]
    # ai_explanation may come as 'ai_explanation' (multi-model) or 'explanation' (legacy)
    report["ai_explanation"]            = ai_result.get("ai_explanation") or ai_result.get("explanation", "")
    report["news_summary"]              = ai_result.get("news_summary", "")
    report["overall_sentiment_summary"] = ai_result.get("overall_sentiment_summary") or ai_result.get("overall_sentiment", "")
    report["key_findings"]              = ai_result.get("key_findings", [])
    report["recommendation"]            = ai_result.get("recommendation", "")
    report["website_category"]          = ai_result.get("website_category", "other")
    report["risk_factors"]              = ai_result.get("risk_factors", [])
    report["positive_signals"]          = ai_result.get("positive_signals", [])
    report["important_links"]           = ai_result.get("important_links", [])
    # New multi-model fields
    report["final_verdict"]           = ai_result.get("final_verdict", "MIXED")
    report["trust_label"]             = ai_result.get("trust_label", "Verify Independently")
    report["confidence_level"]        = ai_result.get("confidence_level", "MEDIUM")
    report["models_agree"]            = ai_result.get("models_agree", False)
    report["claim_verdict"]           = ai_result.get("claim_verdict", "UNVERIFIABLE")
    report["claim_verdict_confidence"]= ai_result.get("claim_verdict_confidence", 0)
    report["claim_explanation"]       = ai_result.get("claim_explanation", "")
    report["story_type"]              = ai_result.get("story_type", "unknown")
    report["corroboration_level"]     = ai_result.get("corroboration_level", "uncorroborated")
    report["source_reputation"]       = ai_result.get("source_reputation", "unknown")
    report["political_bias"]          = ai_result.get("political_bias", "unknown")
    report["overall_sentiment"]       = ai_result.get("overall_sentiment", "neutral")
    report["primary_claims"]          = ai_result.get("primary_claims", [])
    report["logical_fallacies"]       = ai_result.get("logical_fallacies", [])
    report["missing_context"]         = ai_result.get("missing_context", [])
    report["narrative_techniques"]    = ai_result.get("narrative_techniques", [])
    report["corroboration_notes"]     = ai_result.get("corroboration_notes", "")
    report["propaganda_score"]        = ai_result.get("propaganda_score", 20)
    report["transparency_score"]      = ai_result.get("transparency_score", 50)
    report["pipeline"]                = ai_result.get("pipeline", {})
    report["_related_news"]           = related_news
    return report

def generate_image_report(data_uri):
    """Analyze an image using Groq's Vision model to detect AI generation, manipulation, and misinformation."""

    prompt = """You are an expert AI image forensics analyst. Your job is to determine if this image is AI-generated, manipulated, or authentic. Be STRICT and PRECISE — do NOT default to middle scores.

STEP 1 — Identify what this image is:
- A real photograph taken by a camera
- An AI-generated image (Midjourney, DALL-E, Stable Diffusion, etc.)
- A digitally manipulated/photoshopped real photo
- A screenshot (social media, WhatsApp, news)
- A meme or graphic with text
- A digital illustration or artwork
- A document or infographic

STEP 2 — Check for AI generation artifacts:
- HANDS: Extra fingers, merged fingers, impossible hand positions, wrong number of fingers
- FACES: Asymmetrical features, glassy/soulless eyes, blurry ear details, skin too smooth/perfect
- TEXT IN IMAGE: Garbled, misspelled, distorted, or nonsensical text (major AI giveaway)
- BACKGROUNDS: Repeating patterns, objects that defy physics, blurry/melting edges
- LIGHTING: Inconsistent light sources, impossible shadows, overly dramatic lighting
- FINE DETAILS: Jewelry that merges with skin, clothing logos that are distorted, buttons misaligned
- OVERALL FEEL: Too perfect, hyperrealistic yet uncanny, dream-like quality

STEP 3 — Check for Photoshop/manipulation:
- Inconsistent pixel sharpness (part sharp, part blurry)
- Copy-paste artifacts, cloned regions, suspicious edge halos
- Color grading mismatches between subjects and backgrounds
- Text overlays with suspicious fonts or placements
- Metadata-style visual inconsistencies

STEP 4 — Check for misinformation/out-of-context use:
- Does the image show something extreme/shocking that seems staged?
- Is there text in the image making a claim? Is it plausible?
- Does the context feel misleading?

SCORING CALIBRATION (use the FULL range):
- authenticity_score 0-15: Obvious AI-generated (clear artifacts visible)
- authenticity_score 16-35: Likely AI-generated or heavily manipulated
- authenticity_score 36-55: Uncertain — some suspicious elements
- authenticity_score 56-75: Likely real photo but may be edited/out-of-context
- authenticity_score 76-90: Appears to be a real, unmanipulated photograph
- authenticity_score 91-100: Clearly authentic, professional or citizen photography

Return ONLY valid JSON:
{
  "is_authentic": <true ONLY if authenticity_score >= 60 AND no major manipulation found>,
  "authenticity_score": <0-100 per calibration above — do NOT default to 50>,
  "image_type": "<photograph|ai_generated|manipulated_photo|screenshot|meme|digital_art|document|unknown>",
  "ai_confidence": "<definitely_ai|likely_ai|possibly_ai|likely_real|definitely_real>",
  "verdict": "<AI GENERATED|MANIPULATED|OUT OF CONTEXT|LIKELY AUTHENTIC|AUTHENTIC>",
  "explanation": "<3-4 sentences describing what you see and specifically WHY you gave this score — cite actual observed artifacts>",
  "persons_identified": "<describe any people visible, e.g. 'A man in a suit', 'Appears to be a political figure', 'No persons visible'>",
  "ai_artifacts_found": ["<specific artifact 1>", "<specific artifact 2>"],
  "manipulation_signs": ["<sign1>", "<sign2>"],
  "key_findings": ["<finding1>", "<finding2>", "<finding3>"],
  "recommendation": "<1-2 sentence clear advice: should the viewer trust this image or not, and why>"
}"""

    raw = call_groq(
        prompt,
        "You are a strict AI image forensics expert. Analyze every pixel detail carefully. Use the full 0-100 authenticity range — never default to 50. Respond with valid JSON only.",
        1200,
        is_vision=True,
        image_data_uri=data_uri
    )

    if not raw or (isinstance(raw, str) and raw.startswith("ERROR_")):
        print(f"[TruthLens Vision] AI call returned: {raw}")
        return None

    try:
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        text = match.group(0) if match else raw
        result = json.loads(text)

        # Sanitize scores
        result["is_authentic"] = bool(result.get("is_authentic", False))
        score = max(0, min(100, int(result.get("authenticity_score", 50))))
        result["authenticity_score"] = score

        # Auto-set is_authentic based on score if not set correctly
        if score < 55 and result["is_authentic"]:
            result["is_authentic"] = False
        if score >= 70 and not result["is_authentic"]:
            result["is_authentic"] = True

        result.setdefault("image_type", "unknown")
        result.setdefault("ai_confidence", "possibly_ai")
        result.setdefault("verdict", "UNVERIFIED")
        result.setdefault("explanation", "Could not analyze the image.")
        result.setdefault("persons_identified", "No persons visible.")
        result.setdefault("ai_artifacts_found", [])
        result.setdefault("manipulation_signs", [])
        result.setdefault("key_findings", [])
        result.setdefault("recommendation", "Exercise caution when sharing this image.")

        return result

    except Exception as e:
        print(f"[TruthLens Vision] JSON parse failed: {e}")
        return None

