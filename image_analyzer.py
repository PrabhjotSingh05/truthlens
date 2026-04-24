"""
image_analyzer.py - Multi-Model Image Intelligence Pipeline
============================================================
Runs 3 AI models in parallel + web search to produce a high-accuracy,
fact-verified image analysis report.

Pipeline:
  Model 1 (Vision)   - Llama 4 Scout 17B   → Visual forensics & manipulation detection
  Model 2 (Reason)   - Llama 3.3 70B       → Claim extraction & text-based reasoning
  Model 3 (Synthesis)- Llama 3.1 8B Instant → Ensemble judge: weighs both, final verdict
  Web Search         - Google News RSS       → Verifies real-world claims from image
"""

import json
import re
import time
import requests
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from config import Config

# ─── Model Config ──────────────────────────────────────────────────────────────
VISION_MODEL    = "meta-llama/llama-4-scout-17b-16e-instruct"
REASONING_MODEL = "llama-3.3-70b-versatile"
SYNTH_MODEL     = "llama-3.1-8b-instant"


def _post_groq(model, messages, max_tokens=1200, temperature=0.15):
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
    resp = requests.post(Config.GROQ_URL, json=payload, headers=headers, timeout=45)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def _parse_json(raw):
    """Extract JSON from raw model output robustly."""
    # Strip markdown fences
    raw = re.sub(r"```json|```", "", raw).strip()
    # Find outermost { }
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    return json.loads(raw)


# ─── Model 1: Visual Forensics (Vision) ────────────────────────────────────────
def _run_vision_model(data_uri):
    """Llama 4 Scout with image — deep visual forensics."""
    print("[MultiModel] Vision model starting...")
    t0 = time.time()

    prompt = """You are an expert digital forensics investigator. Analyze this image with extreme care.

STEP 1 — DESCRIBE what you literally see in the image (objects, people, text, setting).
STEP 2 — READ any text visible in the image word-for-word (captions, headlines, watermarks, dates, labels).
STEP 3 — ASSESS authenticity: look for AI generation artifacts, Photoshop signs, inconsistent lighting, cloned areas, unnatural faces/hands, melting text, wrong physics.
STEP 4 — ESTIMATE the time period this image was taken (decade, year range, or specific year if visible).

Return ONLY valid JSON, no preamble:
{
  "visual_description": "<Detailed 3-4 sentence description of what is visible in the image>",
  "text_in_image": "<All text visible in the image, quoted exactly. 'None' if no text>",
  "main_claim": "<The core claim or message this image is asserting, if any. 'None' if just a photo>",
  "detected_persons": "<Names or descriptions of any people visible>",
  "estimated_time_period": "<When was this photo/image likely taken? e.g. '2015-2018', 'circa 2020', 'unknown'>",
  "authenticity_score": <0-100, 100=definitely real unedited photo, 0=definitely AI/fake>,
  "image_type": "<photograph|screenshot|meme|ai_generated|digital_art|document|infographic>",
  "manipulation_signs": ["<specific artifact 1>", "<specific artifact 2>"],
  "visual_confidence": <0-100, how confident you are in your visual assessment>,
  "forensic_notes": "<Technical notes on image quality, compression, lighting consistency, etc.>"
}"""

    messages = [{"role": "user", "content": [
        {"type": "text", "text": "You are an expert digital forensics AI. Respond with valid JSON only.\n\n" + prompt},
        {"type": "image_url", "image_url": {"url": data_uri}}
    ]}]

    try:
        raw = _post_groq(VISION_MODEL, messages, max_tokens=1000)
        result = _parse_json(raw)
        result["model"] = "llama-4-scout-vision"
        result["latency_ms"] = int((time.time() - t0) * 1000)
        print(f"[MultiModel] Vision model done ({result['latency_ms']}ms)")
        return result
    except Exception as e:
        print(f"[MultiModel] Vision model error: {e}")
        return {"error": str(e), "authenticity_score": 50, "visual_confidence": 0}


# ─── Model 2: Claim Reasoning (Text LLM) ───────────────────────────────────────
def _run_reasoning_model(vision_result, news_results):
    """Llama 3.3 70B — fact-checks claims using web evidence."""
    print("[MultiModel] Reasoning model starting...")
    t0 = time.time()

    claim = vision_result.get("main_claim", "None")
    text_in_image = vision_result.get("text_in_image", "None")
    description = vision_result.get("visual_description", "")
    time_period = vision_result.get("estimated_time_period", "unknown")

    news_block = ""
    if news_results:
        lines = []
        for i, n in enumerate(news_results[:5], 1):
            lines.append(f"{i}. [{n.get('source','')}] {n.get('title','')} ({n.get('date','')})")
        news_block = "\n".join(lines)
    else:
        news_block = "No related news found."

    prompt = f"""You are an expert fact-checker and investigative journalist. Based on the following evidence, determine whether the claim in this image is TRUE, FALSE, MISLEADING, UNVERIFIABLE, or OUT-OF-CONTEXT.

IMAGE DESCRIPTION (from Vision AI):
{description}

TEXT VISIBLE IN IMAGE:
{text_in_image}

MAIN CLAIM EXTRACTED:
{claim}

ESTIMATED TIME PERIOD OF IMAGE:
{time_period}

RELATED NEWS ARTICLES FOUND ON THE WEB:
{news_block}

Your task:
1. Is the main claim TRUE, FALSE, MISLEADING, OUT_OF_CONTEXT, or UNVERIFIABLE?
2. If the image is old, has the claimed event actually happened? Is it being presented misleadingly?
3. What does the web evidence say?
4. What is the factual truth behind this image?

Return ONLY valid JSON:
{{
  "claim_verdict": "<TRUE|FALSE|MISLEADING|OUT_OF_CONTEXT|UNVERIFIABLE>",
  "claim_verdict_confidence": <0-100>,
  "claim_explanation": "<2-3 sentences explaining the verdict with specific reasoning>",
  "is_old_image_misused": <true if this appears to be an old/archived image being presented as recent or used out of context>,
  "factual_truth": "<What is actually true about this topic, based on evidence? 2-3 sentences.>",
  "web_evidence_summary": "<What do the web results say? If no results, note that.>",
  "timeline_assessment": "<Was this image taken at the time it claims? Is it being recycled?>",
  "credibility_factors": ["<factor supporting or hurting credibility 1>", "<factor 2>", "<factor 3>"],
  "reasoning_score": <0-100, your overall trust score for this image's claim>,
  "sources_checked": {news_results | length if news_results else 0}
}}"""

    messages = [
        {"role": "system", "content": "You are an expert fact-checker. Respond with valid JSON only."},
        {"role": "user", "content": prompt}
    ]

    try:
        raw = _post_groq(REASONING_MODEL, messages, max_tokens=1200)
        result = _parse_json(raw)
        result["model"] = "llama-3.3-70b"
        result["latency_ms"] = int((time.time() - t0) * 1000)
        print(f"[MultiModel] Reasoning model done ({result['latency_ms']}ms)")
        return result
    except Exception as e:
        print(f"[MultiModel] Reasoning model error: {e}")
        return {"error": str(e), "claim_verdict": "UNVERIFIABLE", "reasoning_score": 50}


# ─── Web Search for Claims ──────────────────────────────────────────────────────
def _search_image_claims(claim, description, text_in_image):
    """Search Google News for the claim extracted from the image."""
    print("[MultiModel] Web search starting...")

    queries = []
    if claim and claim != "None" and len(claim) > 10:
        queries.append(claim[:80])
    if text_in_image and text_in_image != "None" and len(text_in_image) > 10:
        queries.append(text_in_image[:60])
    if not queries and description:
        queries.append(description[:70])

    all_results = []
    from urllib.parse import quote

    for q in queries[:2]:
        try:
            encoded = quote(q)
            url = f"https://news.google.com/rss/search?q={encoded}&hl=en&gl=US&ceid=US:en"
            resp = requests.get(url, timeout=8, headers={"User-Agent": Config.USER_AGENT})
            resp.raise_for_status()
            from xml.etree import ElementTree as ET
            root = ET.fromstring(resp.text)
            items = root.findall(".//item")
            for item in items[:4]:
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
                    "source": source, "date": date[:16] if date else ""
                })
                if len(all_results) >= 6:
                    break
        except Exception as e:
            print(f"[MultiModel] Search failed for '{q}': {e}")

    print(f"[MultiModel] Web search done ({len(all_results)} results)")
    return all_results


# ─── Model 3: Synthesis / Judge ────────────────────────────────────────────────
def _run_synthesis_model(vision_result, reasoning_result, news_results):
    """Llama 3.1 8B — fast synthesis of both model outputs into final verdict."""
    print("[MultiModel] Synthesis model starting...")
    t0 = time.time()

    v_score = vision_result.get("authenticity_score", 50)
    v_conf  = vision_result.get("visual_confidence", 50)
    r_score = reasoning_result.get("reasoning_score", 50)
    r_conf  = reasoning_result.get("claim_verdict_confidence", 50)
    verdict = reasoning_result.get("claim_verdict", "UNVERIFIABLE")
    manip   = vision_result.get("manipulation_signs", [])
    cred    = reasoning_result.get("credibility_factors", [])

    prompt = f"""You are an AI judge synthesizing two expert reports about an image.

VISION FORENSICS REPORT:
- Authenticity score: {v_score}/100 (confidence: {v_conf}%)
- Image type: {vision_result.get("image_type", "unknown")}
- Manipulation signs: {manip}
- Forensic notes: {vision_result.get("forensic_notes", "")}

FACT-CHECK REPORT:
- Claim verdict: {verdict} (confidence: {r_conf}%)
- Reasoning score: {r_score}/100
- Is old image misused: {reasoning_result.get("is_old_image_misused", False)}
- Credibility factors: {cred}
- Timeline assessment: {reasoning_result.get("timeline_assessment", "")}

NEWS SOURCES FOUND: {len(news_results)}

Synthesize these into a final authoritative verdict. Weight vision score 40%, fact-check 60% for the final score.

Return ONLY valid JSON:
{{
  "final_authenticity_score": <0-100 weighted final score>,
  "final_verdict": "<AUTHENTIC|SUSPICIOUS|FAKE|MISLEADING|OUT_OF_CONTEXT>",
  "is_authentic": <true only if final_authenticity_score >= 70 AND claim_verdict is TRUE or UNVERIFIABLE>,
  "confidence_level": "<HIGH|MEDIUM|LOW> based on how much evidence was available",
  "models_agree": <true if both vision and fact-check point in same direction>,
  "summary": "<3-4 sentence comprehensive synthesis of both models' findings>",
  "key_findings": ["<most important finding 1>", "<finding 2>", "<finding 3>", "<finding 4>"],
  "recommendation": "<2 sentences: what should the reader do with this image/claim?>",
  "trust_label": "<Safe to Share|Verify Before Sharing|Do Not Share|Needs More Context>"
}}"""

    messages = [
        {"role": "system", "content": "You are an AI synthesis judge. Respond with valid JSON only."},
        {"role": "user", "content": prompt}
    ]

    try:
        raw = _post_groq(SYNTH_MODEL, messages, max_tokens=800, temperature=0.1)
        result = _parse_json(raw)
        result["model"] = "llama-3.1-8b-instant"
        result["latency_ms"] = int((time.time() - t0) * 1000)
        print(f"[MultiModel] Synthesis model done ({result['latency_ms']}ms)")
        return result
    except Exception as e:
        print(f"[MultiModel] Synthesis model error: {e}")
        return {"error": str(e), "final_authenticity_score": 50, "final_verdict": "UNVERIFIABLE"}


# ─── Main Pipeline ──────────────────────────────────────────────────────────────
def generate_image_report(data_uri):
    """
    Full multi-model image intelligence pipeline.
    Runs vision + web search in parallel, then reasoning, then synthesis.
    Returns a rich compiled report dict.
    """
    if not Config.is_ai_enabled():
        print("[MultiModel] AI disabled — no GROQ_API_KEY")
        return None

    print("[MultiModel] === Starting Multi-Model Image Pipeline ===")
    t_start = time.time()

    # Phase 1: Run vision model (needs to be first to extract claim for search)
    vision_result = _run_vision_model(data_uri)

    # Phase 2: Run web search + reasoning model in parallel
    claim       = vision_result.get("main_claim", "None")
    description = vision_result.get("visual_description", "")
    text_img    = vision_result.get("text_in_image", "None")

    news_results = []
    reasoning_result = {}

    with ThreadPoolExecutor(max_workers=2) as executor:
        search_future   = executor.submit(_search_image_claims, claim, description, text_img)
        # Reasoning starts immediately with what we have; will get news after
        reason_future   = executor.submit(_run_reasoning_model, vision_result, [])

        # Get search results
        try:
            news_results = search_future.result(timeout=12)
        except Exception as e:
            print(f"[MultiModel] Search future error: {e}")

        # Get reasoning result
        try:
            reasoning_result = reason_future.result(timeout=40)
        except Exception as e:
            print(f"[MultiModel] Reasoning future error: {e}")
            reasoning_result = {"claim_verdict": "UNVERIFIABLE", "reasoning_score": 50,
                                "claim_verdict_confidence": 0, "claim_explanation": "Reasoning model timed out."}

    # Phase 3: Synthesis (fast, uses both results)
    synthesis_result = _run_synthesis_model(vision_result, reasoning_result, news_results)

    total_ms = int((time.time() - t_start) * 1000)
    print(f"[MultiModel] === Pipeline complete in {total_ms}ms ===")

    # ── Compile final report ──────────────────────────────────────────────────
    sc = synthesis_result.get("final_authenticity_score", 50)
    sc = max(0, min(100, int(sc)))

    report = {
        # Core verdict
        "is_authentic":            synthesis_result.get("is_authentic", sc >= 70),
        "authenticity_score":      sc,
        "final_verdict":           synthesis_result.get("final_verdict", "SUSPICIOUS"),
        "trust_label":             synthesis_result.get("trust_label", "Verify Before Sharing"),
        "confidence_level":        synthesis_result.get("confidence_level", "MEDIUM"),
        "models_agree":            synthesis_result.get("models_agree", False),

        # Image details (from vision)
        "image_type":              vision_result.get("image_type", "unknown"),
        "visual_description":      vision_result.get("visual_description", ""),
        "text_in_image":           vision_result.get("text_in_image", "None"),
        "main_claim":              vision_result.get("main_claim", "None"),
        "persons_identified":      vision_result.get("detected_persons", "None identified."),
        "estimated_time_period":   vision_result.get("estimated_time_period", "Unknown"),
        "forensic_notes":          vision_result.get("forensic_notes", ""),
        "manipulation_signs":      vision_result.get("manipulation_signs", []),
        "vision_score":            vision_result.get("authenticity_score", 50),
        "vision_confidence":       vision_result.get("visual_confidence", 50),

        # Fact-check (from reasoning)
        "claim_verdict":           reasoning_result.get("claim_verdict", "UNVERIFIABLE"),
        "claim_verdict_confidence":reasoning_result.get("claim_verdict_confidence", 0),
        "claim_explanation":       reasoning_result.get("claim_explanation", ""),
        "is_old_image_misused":    reasoning_result.get("is_old_image_misused", False),
        "factual_truth":           reasoning_result.get("factual_truth", ""),
        "web_evidence_summary":    reasoning_result.get("web_evidence_summary", ""),
        "timeline_assessment":     reasoning_result.get("timeline_assessment", ""),
        "credibility_factors":     reasoning_result.get("credibility_factors", []),
        "reasoning_score":         reasoning_result.get("reasoning_score", 50),

        # Synthesis
        "explanation":             synthesis_result.get("summary", ""),
        "key_findings":            synthesis_result.get("key_findings", []),
        "recommendation":          synthesis_result.get("recommendation", ""),

        # Web evidence
        "related_news":            news_results,
        "sources_checked":         len(news_results),

        # Pipeline metadata
        "pipeline": {
            "models_used": [VISION_MODEL, REASONING_MODEL, SYNTH_MODEL],
            "total_ms": total_ms,
            "news_sources": len(news_results),
        }
    }

    return report
