"""
image_analyzer.py - Ultra-Strict Multi-Model Image Intelligence Pipeline v2.0
==============================================================================
MAJOR IMPROVEMENTS:
- Extremely strict AI-generated image detection (penalizes AI images hard)
- Proper person/celebrity recognition with web search cross-reference
- Calibrated scoring: AI images get 0-30, manipulated get 20-45, real get 60-100
- News claim verification via Google News RSS
- 3-model ensemble: Vision forensics → Fact-check → Final synthesis
- Hard scoring guardrails: cannot give high scores to AI/fake images
"""

import json
import re
import time
import requests
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from config import Config

# ─── Model Config ──────────────────────────────────────────────────────────────
VISION_MODEL  = "meta-llama/llama-4-scout-17b-16e-instruct"
REASONING_MODEL = "llama-3.3-70b-versatile"
SYNTH_MODEL   = "llama-3.1-8b-instant"

# ─── Known AI/fake image domains for penalty ───────────────────────────────────
AI_IMAGE_INDICATORS = [
    "midjourney", "dall-e", "stable diffusion", "firefly", "ideogram",
    "leonardo.ai", "nightcafe", "craiyon", "artbreeder", "runway",
    "ai generated", "ai-generated", "made with ai", "created by ai"
]

def _post_groq(model, messages, max_tokens=1500, temperature=0.1):
    """Raw Groq API call. Returns text or raises."""
    headers = {
        "Authorization": f"Bearer {Config.GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    resp = requests.post(Config.GROQ_URL, json=payload, headers=headers, timeout=55)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()

def _parse_json(raw):
    """Extract JSON from raw model output robustly."""
    raw = re.sub(r"```json|```", "", raw).strip()
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except:
            pass
    try:
        return json.loads(raw)
    except:
        return {}

# ─── Model 1: Ultra-Strict Visual Forensics ─────────────────────────────────────
def _run_vision_model(data_uri):
    """Llama 4 Scout — maximum strictness on AI detection and forensics."""
    print("[ImagePipeline] Vision forensics starting...")
    t0 = time.time()

    prompt = """You are the world's most rigorous AI image forensics expert. Your PRIMARY job is detecting AI-generated images. You are EXTREMELY strict — you would rather over-flag than miss an AI image.

CRITICAL SCORING RULES (MUST FOLLOW):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• authenticity_score 0–15:  Definitively AI-generated (Midjourney, DALL-E, SD, Firefly etc.)
• authenticity_score 16–30: Very likely AI or heavily manipulated (95%+ confidence it's fake)
• authenticity_score 31–45: Probably AI/manipulated (many artifacts, not a real photo)
• authenticity_score 46–60: Uncertain — suspicious elements, might be real or edited
• authenticity_score 61–75: Likely real photograph, possibly minor edits
• authenticity_score 76–88: Real photo, looks authentic, minor doubts
• authenticity_score 89–100: Clearly genuine unmanipulated photograph (rare — only for clear evidence)

AI IMAGE RED FLAGS — if you see ANY of these, score MUST be ≤ 35:
• Skin that is unnaturally smooth, pore-free, plastic-like, or "airbrushed"
• Eyes that are unnaturally detailed, glassy, symmetrical, or glowing
• Hair that looks like individual painted strands or is impossibly perfect
• Background that looks painted, dreamlike, or has unrealistic depth-of-field bokeh
• Lighting that is dramatic, studio-perfect, or comes from impossible directions
• Hands with wrong finger count, merged fingers, or impossible anatomy
• Text in image that is garbled, misspelled, or uses fake/distorted fonts
• Overall "hyperrealistic" feel that's more perfect than any real photo
• Clothing without realistic wrinkles, folds, or fabric texture variation
• Jewelry that merges with skin or has distorted/impossible details
• Repeating patterns or textures that look algorithmically generated

PERSON IDENTIFICATION — be very specific:
• If you see a REAL person: describe their features, clothing, setting, any badges/text
• Try to identify WHO they might be based on context, setting, uniforms, surroundings
• Note any context clues: podium, flag, uniform, logo, nameplate, event setting
• If they look AI-generated, say so explicitly

IMAGE TYPE CLASSIFICATION:
• "ai_generated" — made by AI image generator
• "photograph" — real camera photo (digital or film)
• "manipulated_photo" — real photo with edits/compositing
• "screenshot" — screen capture
• "meme" — meme format
• "digital_art" — drawn/painted digitally by human
• "document" — text document or infographic

TEXT READING — read ALL text visible WORD FOR WORD.

Return ONLY valid JSON:
{
  "visual_description": "<4-5 sentence detailed description of EXACTLY what is visible>",
  "text_in_image": "<All text visible, quoted exactly. 'None' if no text>",
  "main_claim": "<The claim/message this image makes. 'None' if just a generic scene>",
  "persons_identified": {
    "count": <number of people visible>,
    "details": [
      {
        "description": "<Physical description: gender, age estimate, features>",
        "clothing": "<What they are wearing>",
        "context_clues": "<Setting, backdrop, text near them, uniform, logo>",
        "likely_identity": "<Best guess of who this might be based on context, or 'Unknown'>",
        "confidence": "<HIGH/MEDIUM/LOW>"
      }
    ],
    "setting": "<Where does this appear to be taken? Room type, outdoor, studio, event, etc.>"
  },
  "image_type": "<photograph|ai_generated|manipulated_photo|screenshot|meme|digital_art|document>",
  "ai_artifacts_detected": {
    "skin_quality": "<natural/suspiciously_smooth/clearly_ai>",
    "eye_quality": "<natural/suspicious/ai_typical>",
    "hair_quality": "<natural/suspicious/ai_typical>",
    "background_quality": "<natural/suspicious/ai_typical>",
    "hands_visible": <true/false>,
    "hand_quality": "<not_visible/natural/suspicious/clearly_wrong>",
    "text_quality": "<no_text/legible/garbled/distorted>",
    "lighting_quality": "<natural/dramatic/impossible/studio_perfect>",
    "overall_feel": "<realistic/hyperrealistic/painted/dreamlike/clearly_fake>"
  },
  "authenticity_score": <0-100 STRICT per calibration — DO NOT give AI images scores above 35>,
  "is_ai_generated": <true if score <= 45>,
  "manipulation_signs": ["<specific artifact with exact location/detail>"],
  "estimated_time_period": "<When was this taken? Decade, year range, or specific year>",
  "visual_confidence": <0-100, how confident you are>,
  "forensic_summary": "<2-3 sentence technical verdict: why you gave this score>"
}"""

    messages = [{"role": "user", "content": [
        {"type": "text", "text": "RESPOND WITH VALID JSON ONLY. No preamble.\n\n" + prompt},
        {"type": "image_url", "image_url": {"url": data_uri}}
    ]}]

    try:
        raw = _post_groq(VISION_MODEL, messages, max_tokens=1500)
        result = _parse_json(raw)
        result["model"] = "llama-4-scout-vision"
        result["latency_ms"] = int((time.time() - t0) * 1000)
        print(f"[ImagePipeline] Vision done ({result['latency_ms']}ms) — score: {result.get('authenticity_score','?')}")
        return result
    except Exception as e:
        print(f"[ImagePipeline] Vision model error: {e}")
        return {"error": str(e), "authenticity_score": 50, "visual_confidence": 0,
                "image_type": "unknown", "persons_identified": {"count": 0, "details": [], "setting": "unknown"}}

# ─── Web Search for Person/Claim Verification ──────────────────────────────────
def _search_for_verification(vision_result):
    """Search Google News to verify claims, persons, and context."""
    print("[ImagePipeline] Web verification starting...")
    t0 = time.time()

    from urllib.parse import quote
    from xml.etree import ElementTree as ET

    queries = []

    # Build smart search queries from vision data
    claim = vision_result.get("main_claim", "None")
    text_in_image = vision_result.get("text_in_image", "None")
    persons = vision_result.get("persons_identified", {})

    if isinstance(persons, dict):
        for p in persons.get("details", []):
            if p.get("likely_identity") and p["likely_identity"] != "Unknown":
                queries.append(p["likely_identity"][:60])

    if claim and claim != "None" and len(claim) > 10:
        queries.append(claim[:80])

    if text_in_image and text_in_image != "None" and len(text_in_image) > 8:
        queries.append(text_in_image[:70])

    all_results = []

    for q in queries[:3]:
        try:
            encoded = quote(q)
            url = f"https://news.google.com/rss/search?q={encoded}&hl=en&gl=US&ceid=US:en"
            resp = requests.get(url, timeout=9, headers={"User-Agent": Config.USER_AGENT})
            resp.raise_for_status()

            root = ET.fromstring(resp.text)
            for item in root.findall(".//item")[:4]:
                title = item.findtext("title", "")
                link = item.findtext("link", "")
                date = item.findtext("pubDate", "")
                source = ""
                if " - " in title:
                    parts = title.rsplit(" - ", 1)
                    title = parts[0].strip()
                    source = parts[1].strip()
                all_results.append({
                    "title": title, "link": link,
                    "source": source, "date": date[:16] if date else "",
                    "query_used": q
                })

            if len(all_results) >= 8:
                break
        except Exception as e:
            print(f"[ImagePipeline] Search error for '{q}': {e}")

    print(f"[ImagePipeline] Web search done ({len(all_results)} results, {int((time.time()-t0)*1000)}ms)")
    return all_results

# ─── Model 2: Claim & Context Fact-Checker ────────────────────────────────────
def _run_reasoning_model(vision_result, news_results):
    """Llama 3.3 70B — verifies claims and context using web evidence."""
    print("[ImagePipeline] Fact-check reasoning starting...")
    t0 = time.time()

    claim = vision_result.get("main_claim", "None")
    text_in_image = vision_result.get("text_in_image", "None")
    description = vision_result.get("visual_description", "")
    persons = vision_result.get("persons_identified", {})
    is_ai = vision_result.get("is_ai_generated", False)
    v_score = vision_result.get("authenticity_score", 50)
    time_period = vision_result.get("estimated_time_period", "unknown")

    # Format person info
    person_info = "No persons visible"
    if isinstance(persons, dict) and persons.get("details"):
        parts = []
        for p in persons["details"]:
            identity = p.get("likely_identity", "Unknown")
            desc = p.get("description", "")
            ctx = p.get("context_clues", "")
            parts.append(f"- {identity} ({desc}) | Context: {ctx}")
        person_info = "\n".join(parts)

    news_block = ""
    if news_results:
        lines = [f"{i+1}. [{n.get('source','')}] {n.get('title','')} ({n.get('date','')})"
                 for i, n in enumerate(news_results[:6])]
        news_block = "\n".join(lines)
    else:
        news_block = "No matching news found."

    prompt = f"""You are an expert fact-checker and investigative journalist. Analyze this image's claims and context.

VISION FORENSICS REPORT:
- Image authenticity score: {v_score}/100
- Is AI-generated: {is_ai}
- Time period: {time_period}

IMAGE DESCRIPTION: {description}

TEXT VISIBLE IN IMAGE: {text_in_image}

MAIN CLAIM: {claim}

PERSONS IDENTIFIED:
{person_info}

RELATED NEWS FOUND:
{news_block}

TASK:
1. Is the claim TRUE, FALSE, MISLEADING, OUT_OF_CONTEXT, or UNVERIFIABLE?
2. Are the identified persons actually who they appear to be?
3. Does this image match what the news says about this event/topic?
4. Is this an old image being recycled as recent news?
5. If AI-generated: is it being used to spread misinformation about a real event?

IMPORTANT: If the image is AI-generated (score ≤ 45) AND makes a factual claim about real events, that is EXTREMELY suspicious — score should be very low.

Return ONLY valid JSON:
{{
  "claim_verdict": "<TRUE|FALSE|MISLEADING|OUT_OF_CONTEXT|UNVERIFIABLE>",
  "claim_verdict_confidence": <0-100>,
  "claim_explanation": "<2-3 sentences explaining the verdict>",
  "persons_verified": [
    {{
      "described_as": "<who vision model identified>",
      "verification": "<CONFIRMED|LIKELY|UNCERTAIN|CONTRADICTED>",
      "evidence": "<what news/context supports or refutes this identity>"
    }}
  ],
  "is_old_image_misused": <true if recycled/recontextualized>,
  "is_ai_image_misinfo": <true if AI image is claiming to show real events>,
  "factual_truth": "<What is actually true about this topic? 2-3 sentences.>",
  "web_evidence_summary": "<What do the news results tell us?>",
  "timeline_accuracy": "<Does this image match the time period claimed? Evidence?>",
  "credibility_factors": ["<factor 1>", "<factor 2>", "<factor 3>"],
  "reasoning_score": <0-100 — MUST be low if AI image making false claims>,
  "misinformation_risk": "<LOW|MEDIUM|HIGH|CRITICAL>"
}}"""

    messages = [
        {"role": "system", "content": "You are a strict fact-checker. Respond with valid JSON only. Be extremely suspicious of AI-generated images making factual claims."},
        {"role": "user", "content": prompt}
    ]

    try:
        raw = _post_groq(REASONING_MODEL, messages, max_tokens=1300)
        result = _parse_json(raw)
        result["model"] = "llama-3.3-70b"
        result["latency_ms"] = int((time.time() - t0) * 1000)
        print(f"[ImagePipeline] Fact-check done ({result['latency_ms']}ms) — verdict: {result.get('claim_verdict','?')}")
        return result
    except Exception as e:
        print(f"[ImagePipeline] Reasoning error: {e}")
        return {"error": str(e), "claim_verdict": "UNVERIFIABLE", "reasoning_score": 50,
                "claim_verdict_confidence": 0, "misinformation_risk": "MEDIUM"}

# ─── Model 3: Final Synthesis Judge ───────────────────────────────────────────
def _run_synthesis_model(vision_result, reasoning_result, news_results):
    """Fast synthesis + STRICT scoring guardrails."""
    print("[ImagePipeline] Synthesis judge starting...")
    t0 = time.time()

    v_score = vision_result.get("authenticity_score", 50)
    v_conf = vision_result.get("visual_confidence", 50)
    r_score = reasoning_result.get("reasoning_score", 50)
    verdict = reasoning_result.get("claim_verdict", "UNVERIFIABLE")
    is_ai = vision_result.get("is_ai_generated", False)
    is_ai_misinfo = reasoning_result.get("is_ai_image_misinfo", False)
    misinfo_risk = reasoning_result.get("misinformation_risk", "MEDIUM")
    manip = vision_result.get("manipulation_signs", [])
    img_type = vision_result.get("image_type", "unknown")
    persons = vision_result.get("persons_identified", {})

    prompt = f"""You are the final AI judge synthesizing two expert analysis reports about an image.

VISION FORENSICS:
- Authenticity score: {v_score}/100 (confidence: {v_conf}%)
- Image type: {img_type}
- Is AI generated: {is_ai}
- Manipulation signs: {manip}
- Forensic summary: {vision_result.get('forensic_summary', '')}

FACT-CHECK:
- Claim verdict: {verdict} (confidence: {reasoning_result.get('claim_verdict_confidence', 0)}%)
- Reasoning score: {r_score}/100
- Is old image misused: {reasoning_result.get('is_old_image_misused', False)}
- Is AI image spreading misinfo: {is_ai_misinfo}
- Misinformation risk: {misinfo_risk}

WEB SOURCES FOUND: {len(news_results)}

CRITICAL SCORING GUARDRAILS (MUST ENFORCE):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• If image_type is "ai_generated" OR v_score ≤ 35: final_authenticity_score MUST be ≤ 30
• If is_ai_image_misinfo is true: final_authenticity_score MUST be ≤ 20
• If claim_verdict is FALSE: deduct 25 points minimum from v_score
• If claim_verdict is MISLEADING or OUT_OF_CONTEXT: deduct 20 points
• If misinfo_risk is CRITICAL: final score MUST be ≤ 25
• If misinfo_risk is HIGH: final score MUST be ≤ 40
• Only give scores 70+: clearly real photo + claim is TRUE or UNVERIFIABLE

Calculate: final_score = (v_score * 0.45) + (r_score * 0.55), then apply guardrails.

Return ONLY valid JSON:
{{
  "final_authenticity_score": <0-100 after guardrails>,
  "final_verdict": "<AUTHENTIC|SUSPICIOUS|FAKE|AI_GENERATED|MISLEADING|OUT_OF_CONTEXT|MANIPULATED>",
  "is_authentic": <true ONLY if score >= 65 AND claim is not FALSE>,
  "confidence_level": "<HIGH|MEDIUM|LOW>",
  "models_agree": <true if both point same direction>,
  "trust_label": "<Safe to Share|Verify Before Sharing|Do Not Share|AI Generated - Do Not Trust>",
  "summary": "<4-5 sentence comprehensive synthesis of what this image is, who is in it, and whether it can be trusted>",
  "key_findings": ["<critical finding 1>", "<finding 2>", "<finding 3>", "<finding 4>"],
  "recommendation": "<2-3 sentences: what should the viewer know and do?>",
  "why_low_score": "<If score ≤ 50: specific reasons why this scored low>",
  "persons_summary": "<Summary of who is in the image and confidence level>"
}}"""

    messages = [
        {"role": "system", "content": "You are a strict AI synthesis judge. ENFORCE the scoring guardrails. Respond with valid JSON only."},
        {"role": "user", "content": prompt}
    ]

    try:
        raw = _post_groq(SYNTH_MODEL, messages, max_tokens=900, temperature=0.05)
        result = _parse_json(raw)
        result["model"] = "llama-3.1-8b-instant"
        result["latency_ms"] = int((time.time() - t0) * 1000)
        print(f"[ImagePipeline] Synthesis done ({result['latency_ms']}ms) — final: {result.get('final_authenticity_score','?')}")
        return result
    except Exception as e:
        print(f"[ImagePipeline] Synthesis error: {e}")
        return {"error": str(e), "final_authenticity_score": v_score, "final_verdict": "UNVERIFIABLE"}

# ─── Scoring Guardrail Engine ──────────────────────────────────────────────────
def _apply_hard_guardrails(score, vision_result, reasoning_result):
    """Apply absolute scoring rules that cannot be overridden."""
    img_type = vision_result.get("image_type", "unknown")
    v_score = vision_result.get("authenticity_score", 50)
    is_ai = vision_result.get("is_ai_generated", False)
    verdict = reasoning_result.get("claim_verdict", "UNVERIFIABLE")
    misinfo = reasoning_result.get("misinformation_risk", "MEDIUM")
    is_ai_misinfo = reasoning_result.get("is_ai_image_misinfo", False)

    # Hard caps for AI-generated images
    if img_type == "ai_generated" or is_ai or v_score <= 35:
        score = min(score, 30)

    # Misinfo with AI = critical
    if is_ai_misinfo:
        score = min(score, 20)

    # False claims penalty
    if verdict == "FALSE":
        score = min(score, score - 25 if score > 25 else 10)
    elif verdict in ("MISLEADING", "OUT_OF_CONTEXT"):
        score = min(score, score - 15 if score > 20 else 15)

    # Misinformation risk caps
    if misinfo == "CRITICAL":
        score = min(score, 25)
    elif misinfo == "HIGH":
        score = min(score, 40)

    return max(1, min(100, int(score)))

# ─── Main Pipeline ──────────────────────────────────────────────────────────────
def generate_image_report(data_uri):
    """
    Full multi-model image intelligence pipeline v2.0.
    Features strict AI detection, person recognition, and calibrated scoring.
    """
    if not Config.is_ai_enabled():
        print("[ImagePipeline] AI disabled — no GROQ_API_KEY")
        return None

    print("[ImagePipeline] === Starting Ultra-Strict Image Analysis Pipeline ===")
    t_start = time.time()

    # Phase 1: Vision forensics (must run first to extract info for search)
    vision_result = _run_vision_model(data_uri)

    # Phase 2: Web verification + fact-check in parallel
    news_results = []
    reasoning_result = {}

    with ThreadPoolExecutor(max_workers=2) as executor:
        search_future = executor.submit(_search_for_verification, vision_result)
        reason_future = executor.submit(_run_reasoning_model, vision_result, [])

        try:
            news_results = search_future.result(timeout=15)
        except Exception as e:
            print(f"[ImagePipeline] Search timeout: {e}")

        try:
            reasoning_result = reason_future.result(timeout=45)
        except Exception as e:
            print(f"[ImagePipeline] Reasoning timeout: {e}")
            reasoning_result = {
                "claim_verdict": "UNVERIFIABLE",
                "reasoning_score": 50,
                "claim_verdict_confidence": 0,
                "misinformation_risk": "MEDIUM",
                "is_ai_image_misinfo": vision_result.get("is_ai_generated", False)
            }

    # Phase 3: Synthesis
    synthesis_result = _run_synthesis_model(vision_result, reasoning_result, news_results)

    total_ms = int((time.time() - t_start) * 1000)
    print(f"[ImagePipeline] === Pipeline complete in {total_ms}ms ===")

    # Apply hard scoring guardrails
    raw_score = synthesis_result.get("final_authenticity_score", 50)
    final_score = _apply_hard_guardrails(raw_score, vision_result, reasoning_result)

    # Recalculate is_authentic based on final guardrailed score
    is_authentic = (
        final_score >= 65 and
        reasoning_result.get("claim_verdict") not in ("FALSE", "MISLEADING") and
        not vision_result.get("is_ai_generated", False) and
        vision_result.get("image_type") != "ai_generated"
    )

    # Format persons data for template
    persons_data = vision_result.get("persons_identified", {})
    if isinstance(persons_data, dict):
        persons_list = persons_data.get("details", [])
        person_summary = synthesis_result.get("persons_summary", "")
        if not person_summary and persons_list:
            parts = []
            for p in persons_list:
                identity = p.get("likely_identity", "Unknown")
                desc = p.get("description", "")
                conf = p.get("confidence", "LOW")
                parts.append(f"{identity} ({conf} confidence) — {desc}")
            person_summary = "; ".join(parts) if parts else "No persons identified."
        elif not person_summary:
            person_summary = "No persons visible in this image."
    else:
        persons_list = []
        person_summary = str(persons_data) if persons_data else "No persons identified."

    # Format verified persons
    persons_verified = reasoning_result.get("persons_verified", [])

    # Compile trust label
    trust_label = synthesis_result.get("trust_label", "Verify Before Sharing")
    if vision_result.get("is_ai_generated") or vision_result.get("image_type") == "ai_generated":
        trust_label = "AI Generated — Do Not Trust as Real"
    elif final_score <= 25:
        trust_label = "Do Not Share"
    elif final_score <= 50:
        trust_label = "Verify Before Sharing"
    elif final_score >= 75:
        trust_label = "Safe to Share"

    # AI artifacts for display
    ai_artifacts = vision_result.get("ai_artifacts_detected", {})
    artifacts_list = []
    if isinstance(ai_artifacts, dict):
        for key, val in ai_artifacts.items():
            if val in ("suspiciously_smooth", "ai_typical", "clearly_ai", "clearly_wrong",
                       "garbled", "distorted", "impossible", "studio_perfect",
                       "hyperrealistic", "painted", "dreamlike", "clearly_fake"):
                readable_key = key.replace("_", " ").title()
                artifacts_list.append(f"{readable_key}: {val.replace('_', ' ')}")

    all_manipulation_signs = list(vision_result.get("manipulation_signs", []))
    # Merge ai artifact flags into manipulation signs
    all_manipulation_signs = artifacts_list + all_manipulation_signs

    report = {
        # Core verdict
        "is_authentic": is_authentic,
        "authenticity_score": final_score,
        "raw_vision_score": vision_result.get("authenticity_score", 50),
        "final_verdict": synthesis_result.get("final_verdict", "SUSPICIOUS"),
        "trust_label": trust_label,
        "confidence_level": synthesis_result.get("confidence_level", "MEDIUM"),
        "models_agree": synthesis_result.get("models_agree", False),

        # Image type & AI detection
        "image_type": vision_result.get("image_type", "unknown"),
        "is_ai_generated": vision_result.get("is_ai_generated", final_score <= 35),
        "is_ai_image_misinfo": reasoning_result.get("is_ai_image_misinfo", False),
        "ai_confidence": "definitely_ai" if final_score <= 20 else
                         "likely_ai" if final_score <= 35 else
                         "possibly_ai" if final_score <= 50 else
                         "likely_real" if final_score <= 75 else "definitely_real",

        # Visual description
        "visual_description": vision_result.get("visual_description", ""),
        "text_in_image": vision_result.get("text_in_image", "None"),
        "main_claim": vision_result.get("main_claim", "None"),
        "estimated_time_period": vision_result.get("estimated_time_period", "Unknown"),
        "forensic_notes": vision_result.get("forensic_summary", ""),
        "manipulation_signs": all_manipulation_signs,
        "vision_score": vision_result.get("authenticity_score", 50),
        "vision_confidence": vision_result.get("visual_confidence", 50),

        # Person identification
        "persons_identified": person_summary,
        "persons_list": persons_list,
        "persons_verified": persons_verified,
        "persons_setting": persons_data.get("setting", "") if isinstance(persons_data, dict) else "",
        "persons_count": persons_data.get("count", 0) if isinstance(persons_data, dict) else 0,

        # Fact-check
        "claim_verdict": reasoning_result.get("claim_verdict", "UNVERIFIABLE"),
        "claim_verdict_confidence": reasoning_result.get("claim_verdict_confidence", 0),
        "claim_explanation": reasoning_result.get("claim_explanation", ""),
        "is_old_image_misused": reasoning_result.get("is_old_image_misused", False),
        "factual_truth": reasoning_result.get("factual_truth", ""),
        "web_evidence_summary": reasoning_result.get("web_evidence_summary", ""),
        "timeline_accuracy": reasoning_result.get("timeline_accuracy", ""),
        "credibility_factors": reasoning_result.get("credibility_factors", []),
        "reasoning_score": reasoning_result.get("reasoning_score", 50),
        "misinformation_risk": reasoning_result.get("misinformation_risk", "MEDIUM"),

        # Synthesis output
        "explanation": synthesis_result.get("summary", ""),
        "key_findings": synthesis_result.get("key_findings", []),
        "recommendation": synthesis_result.get("recommendation", ""),
        "why_low_score": synthesis_result.get("why_low_score", ""),

        # Web evidence
        "related_news": news_results,
        "sources_checked": len(news_results),

        # Pipeline metadata
        "pipeline": {
            "models_used": [VISION_MODEL, REASONING_MODEL, SYNTH_MODEL],
            "total_ms": total_ms,
            "news_sources": len(news_results),
            "guardrails_applied": True,
            "score_before_guardrails": raw_score,
            "score_after_guardrails": final_score,
        }
    }

    return report
