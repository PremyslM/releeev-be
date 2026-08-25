"""
Claude Vision extraction + Voyage AI embeddings.

Public API:
    extractAttributes(image_base64, mime_type) -> ExtractedAttributes
    getEmbedding(text)                         -> list[float]  (1024-dim)
"""
from __future__ import annotations

import json
import logging
import re
import time
from functools import lru_cache

import anthropic
import voyageai

from src.config import settings
from src.models import CaptureType, ExtractedAttributes

logger = logging.getLogger(__name__)

# ── Clients ───────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _claude() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=settings.anthropic_api_key)


@lru_cache(maxsize=1)
def _voyage() -> voyageai.Client:
    # Interactive path (search rerank). max_retries=0: on the free tier's 3 RPM
    # limit, fail fast (~100ms) and degrade gracefully instead of blocking search.
    key = settings.voyage_api_key or settings.anthropic_api_key
    return voyageai.Client(api_key=key, max_retries=0)


@lru_cache(maxsize=1)
def _voyage_durable() -> voyageai.Client:
    # Embedding path (ingest / reembed). A missing embedding is a broken capture
    # (zero vector ⇒ no links, no semantic search), so here we let the SDK back
    # off and retry through a rate-limit spike rather than fail fast.
    key = settings.voyage_api_key or settings.anthropic_api_key
    return voyageai.Client(api_key=key, max_retries=3)


def _embed_with_retry(texts: list[str], attempts: int = 3, pause: float = 6.0):
    """Embed on the durable client, retrying a rate-limit spike with a bounded
    wait. Raises on final failure so the caller can flag the capture for the
    background drain instead of silently storing a zero vector."""
    import time
    last: Exception | None = None
    for i in range(attempts):
        try:
            return _voyage_durable().embed(texts=texts, model="voyage-3", input_type="document")
        except Exception as exc:  # noqa: BLE001
            last = exc
            msg = str(exc).lower()
            transient = "rate" in msg or "429" in msg or "timeout" in msg or "connection" in msg
            if not transient or i == attempts - 1:
                raise
            logger.warning("Voyage embed rate-limited, retry %d/%d in %.0fs",
                           i + 1, attempts, pause)
            time.sleep(pause)
    raise last  # unreachable, keeps type checkers happy


# ── Extraction prompt ─────────────────────────────────────────────────────────

_SYSTEM = """You are an intelligent capture analyser for a second-brain app.
Look at the image and return a strict JSON object with these fields (all required):

{
  "title":        "<short, specific title of WHAT this is>",
  "summary":      "<the takeaway — see rules>",
  "raw_text":     "<readable text transcribed verbatim, or empty string>",
  "capture_type": "<one of: screenshot|photo|document|unknown>",
  "topics":       ["<keyword1>", ...],
  "category":     "<exactly ONE broad domain category from the allowed list>",
  "people":       ["<full name if visible>", ...],
  "places":       ["<city / building / URL if visible>", ...],
  "language":     "<ISO-639-1 code, e.g. en>",
  "sentiment":    "<positive|negative|neutral>",
  "confidence":   <float 0.0-1.0 representing your extraction confidence>
}

SUMMARY RULES — this is the important part:
- Do NOT describe what is visibly in the picture; the user already saw it.
  Never write "a photo of…", "the image shows…", "a screenshot of…".
- Instead, infer and state what it MEANS or what it's FOR: the point, the
  decision, the fact worth remembering, the conclusion. Be concise (1–3 sentences).
- If it identifies a real thing (a product, place, dish, document), name it and
  give the useful gist (e.g. what it is, key spec/price/where), not its looks.
- For a document/screenshot with text, summarise the substance, not the layout.

Respond with ONLY valid JSON — no markdown fences, no prose.
"""


# ISO-639-1 → language name, so the model gets a named target (which it follows
# far better than a bare code) rather than just "cs".
_LANG_NAMES = {
    "cs": "Czech (čeština)", "sk": "Slovak (slovenčina)", "en": "English",
    "de": "German (Deutsch)", "pl": "Polish (polski)", "es": "Spanish (español)",
    "fr": "French (français)", "it": "Italian (italiano)", "pt": "Portuguese (português)",
    "ru": "Russian (русский)", "uk": "Ukrainian (українська)", "nl": "Dutch (Nederlands)",
}


# One log should join a couple of WIDE shelves, not spawn a private taxonomy.
# Topics are the user-visible "categories"/strings, so many narrow ones fragment
# the brain into single-member buckets; the title and summary already carry the
# specifics.
_TOPIC_RULES = (
    'Topics: name the BROADEST shelf this log belongs on — the general kind of '
    'thing it is, not its details. Give 1-3 topics, and only more than one if it '
    'genuinely spans separate areas. Prefer wide, reusable English nouns that '
    'many future logs could share ("coffee" — not "v60 pour-over technique"; '
    '"cars" — not "nissan pathfinder offroad suspension"; "ai" — not "cursor vs '
    'copilot autocomplete"). Never add a topic just to be thorough: a narrower '
    'topic is only worth it if you would expect a dozen other logs under it. Do '
    'NOT put the category word itself into topics.'
)


# Languages written in Latin script — for these, any Cyrillic/Greek character in
# the output is the model bleeding the source language through, never legitimate.
_LATIN_SCRIPT_LANGS = {
    "cs", "sk", "en", "de", "pl", "es", "fr", "it", "pt", "nl", "da", "sv",
    "no", "fi", "hu", "ro", "hr", "sl", "lt", "lv", "et", "tr",
}
_FOREIGN_SCRIPT = re.compile(r"[Ѐ-ӿͰ-Ͽ]")   # Cyrillic, Greek


def _script_leaked(text: str, lang: str) -> bool:
    """True when the output contains a script the target language never uses.

    The fast model reliably follows the "write in Czech" instruction for whole
    sentences but still drops in stray source-language words (e.g. "в равném
    objemu", "pиву"). A prompt can't fully stop it, so we detect and redo."""
    if (lang or "").strip().lower() not in _LATIN_SCRIPT_LANGS:
        return False
    return bool(_FOREIGN_SCRIPT.search(text or ""))


def _lang_instruction(lang: str) -> str:
    """Ask the model to write the human-facing fields in the device language,
    while topics stay canonical English (so related captures share keywords)."""
    lang = (lang or "").strip()
    if not lang:
        return ""
    name = _LANG_NAMES.get(lang.lower(), f'the language with ISO code "{lang}"')
    return (f' CRITICAL LANGUAGE RULE: write "title" and "summary" ENTIRELY in '
            f'{name}. TRANSLATE everything into {name} — the summary must contain '
            f'NO words from the source language or any other language. Never copy '
            f'or transliterate source words (e.g. a product description); render '
            f'them in {name}. Every sentence, section label (translate '
            f'"Ingredients:"/"Steps:"/"Note:" too) and list item must be in '
            f'{name}. Keep only "topics" in canonical English.')


def _parse_json_reply(resp) -> dict:
    """Best-effort JSON out of a model reply.

    Models occasionally wrap the object in prose/fences, or emit an unescaped
    quote inside a value (a poster reading `"středo/věk"` is enough). A failed
    parse used to silently yield {} — an untitled, unsummarised, unlinkable
    capture — so tolerate what we safely can and log the rest legibly."""
    raw = next((b.text for b in resp.content if b.type == "text"), "").strip()
    if not raw:
        return {}
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\s*", "", raw)
        raw = re.sub(r"```\s*$", "", raw).strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end > start:
        raw = raw[start : end + 1]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    try:                                     # drop trailing commas: {"a": 1,}
        return json.loads(re.sub(r",\s*([}\]])", r"\1", raw))
    except json.JSONDecodeError as exc:
        # \n-escape so the whole payload lands on one log line, not just "{".
        logger.error("Attribute JSON parse failed: %s | Raw: %s",
                     exc, raw[:600].replace("\n", "\\n"))
        return {}


def _complete_json(
    *,
    messages: list,
    lang: str,
    max_tokens: int,
    system: str | None = None,
    model: str | None = None,
) -> dict:
    """Extract on `model` (the cheap one by default), retrying once on the
    stronger one if the reply was unusable (unparseable / empty) or leaked
    foreign script into localized text. Both are rare, so cost stays close to
    single-pass. Vision passes the strong model as primary — images need it."""
    client = _claude()
    primary = model or settings.claude_model_fast

    def run(model: str, msgs: list) -> dict:
        kwargs = {"model": model, "max_tokens": max_tokens, "messages": msgs}
        if system:
            kwargs["system"] = system
        return _parse_json_reply(client.messages.create(**kwargs))

    data = run(primary, messages)
    blob = f"{data.get('title', '')} {data.get('summary', '')}"
    broken = not (data.get("title") or data.get("summary"))
    leaked = _script_leaked(blob, lang)
    if not broken and not leaked:
        return data

    reason = "unparseable/empty reply" if broken else f"foreign script in {lang} output"
    logger.warning("Extraction retry on %s — %s", settings.claude_model, reason)
    nudge = (
        "\n\nYour previous attempt was not valid JSON, or was empty. Return ONLY "
        "the JSON object. Escape every quote inside a string value as \\\", and "
        "never use a raw newline inside a string."
        if broken else
        "\n\nYour previous attempt leaked words in a foreign script (e.g. "
        "Cyrillic) into the text. Rewrite it so EVERY word is in the target "
        "language and script. Translate any source-language term; never copy it."
    )
    retry_msgs = [dict(m) for m in messages]
    last = retry_msgs[-1]
    if isinstance(last.get("content"), str):
        last["content"] = last["content"] + nudge
    elif isinstance(last.get("content"), list):
        last["content"] = last["content"] + [{"type": "text", "text": nudge}]

    strong = run(settings.claude_model, retry_msgs)
    if not (strong.get("title") or strong.get("summary")):
        return data                                    # retry no better — keep first
    if _script_leaked(f"{strong.get('title', '')} {strong.get('summary', '')}", lang):
        return data if not broken else strong
    return strong


def extractTopicsOnly(title: str, summary: str, raw_text: str = "") -> dict:
    """Re-derive ONLY the shelves (category/topics/people/places) from a capture's
    stored text, using the broad-topic rule. Keeps the existing title/summary.

    The single source of truth for "what shelves does this go on", shared by the
    per-capture reanalyze and the whole-brain backfill — so neither drifts back to
    the old narrow-topic behaviour."""
    from src.link_engine import DOMAIN_TOPICS

    allowed = ", ".join(sorted(DOMAIN_TOPICS))
    prompt = f"""Re-derive the metadata for this saved note. Return ONLY valid JSON.

Title: {title}
Summary: {summary}
Raw text: {raw_text[:800]}

{{
  "category": "<exactly ONE of: {allowed}>",
  "topics":   ["<1-3 broad English topic keywords — see Topics rule>"],
  "people":   ["<real person names mentioned, or []>"],
  "places":   ["<real place names / brands mentioned, or []>"]
}}

{_TOPIC_RULES}"""

    resp = _claude().messages.create(
        model=settings.claude_model_fast,
        max_tokens=250,
        messages=[{"role": "user", "content": prompt}],
    )
    data = _parse_json_reply(resp)   # no title/summary in the reply → parse directly

    topics = [t for t in data.get("topics", []) if isinstance(t, str)][:3]
    category = (data.get("category") or "").strip().lower()
    if category in DOMAIN_TOPICS:
        if category not in [t.lower() for t in topics]:
            topics.append(category)
    else:
        category = ""
    return {
        "category": category,
        "topics": topics,
        "people": [p for p in data.get("people", []) if isinstance(p, str)][:10],
        "places": [pl for pl in data.get("places", []) if isinstance(pl, str)][:10],
    }


# ── Main functions ────────────────────────────────────────────────────────────

def extractAttributes(
    image_base64: str,
    mime_type: str = "image/jpeg",
    lang: str = "",
) -> ExtractedAttributes:
    """
    Send an image to Claude (settings.claude_model) via the Vision API and parse
    the structured JSON response into an ExtractedAttributes object.
    """
    from src.link_engine import DOMAIN_TOPICS  # local import: avoids any cycle

    t0 = time.perf_counter()
    allowed = ", ".join(sorted(DOMAIN_TOPICS))

    data = _complete_json(
        system=_SYSTEM,
        lang=lang,
        max_tokens=2048,
        model=settings.claude_model,   # vision quality matters; keep the strong model
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": mime_type,
                            "data": image_base64,
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            "Analyse this image and return the JSON object. "
                            f'The "category" must be exactly ONE of: {allowed}. '
                            "Pick the closest one even for photos (e.g. a car photo → "
                            "automotive). "
                            + _TOPIC_RULES
                            + _lang_instruction(lang)
                        ),
                    },
                ],
            }
        ],
    )

    elapsed = (time.perf_counter() - t0) * 1000
    logger.info("extractAttributes completed in %.0f ms", elapsed)

    # Map capture_type to enum safely
    ct_raw = data.get("capture_type", "unknown")
    try:
        ct = CaptureType(ct_raw)
    except ValueError:
        ct = CaptureType.unknown

    # Ensure the broad domain category is one of the topics, so photos link to
    # documents of the same domain (e.g. a car photo ↔ a car listing via
    # "automotive"). Mirrors the URL/text extraction path.
    # Hard guard: keep the shelf list short even if the model over-produces.
    topics = [t for t in data.get("topics", []) if isinstance(t, str)][:3]
    category = (data.get("category") or "").strip().lower()
    if category in DOMAIN_TOPICS and category not in [t.lower() for t in topics]:
        topics.append(category)

    return ExtractedAttributes(
        title=data.get("title", ""),
        summary=data.get("summary", ""),
        raw_text=data.get("raw_text", ""),
        capture_type=ct,
        topics=topics,
        people=data.get("people", []),
        places=data.get("places", []),
        language=data.get("language", "en"),
        sentiment=data.get("sentiment", "neutral"),
        confidence=float(data.get("confidence", 1.0)),
    )


def extractAttributesFromText(
    text: str, url: str = "", signals: dict | None = None, lang: str = ""
) -> ExtractedAttributes:
    """
    Build structured attributes from a web page's text (instead of an image).

    Used by POST /analyze-url. Mirrors extractAttributes() but the input is the
    page's text + HTML signals (title, OG tags) rather than pixels.
    """
    from src.link_engine import DOMAIN_TOPICS  # local import: avoids any cycle

    signals = signals or {}
    sig_lines = "\n".join(f"{k}: {v}" for k, v in signals.items() if v)
    allowed = ", ".join(sorted(DOMAIN_TOPICS))

    prompt = f"""Analyse this web page / video and return ONLY valid JSON, no prose.

URL: {url}
{sig_lines}

Content (truncated):
{text[:6000]}

Return JSON:
{{
  "title":    "<concise, specific title of what this is>",
  "summary":  "<a genuinely useful synthesis — see rules>",
  "topics":   ["<1-3 broad English topic keywords — see Topics rule>"],
  "category": "<exactly ONE of: {allowed}>",
  "people":   ["<real person names, or []>"],
  "places":   ["<real place names / brands, or []>"],
  "language": "<ISO-639-1 code>",
  "sentiment":"<positive|negative|neutral>"
}}

SUMMARY RULES — extract what's worth keeping, don't just describe the page:
- Open with 1-2 plain sentences of gist (no heading), then structure the rest.
- Adapt the shape to the content:
  • Recipe → an ingredients bulleted list where EACH bullet includes the
    amount/quantity from the page (e.g. "- 500 g potatoes", "- 2 eggs"). If the
    page truly gives no amount for an item, list it without one — never invent
    quantities. Then a numbered steps list.
  • How-to / tutorial → the key steps or takeaways as a short list.
  • Product → what it is + key specs / price as a few bullets.
  • Article / video → 3-6 bullet key points or conclusions.
- FORMATTING (Markdown, this is rendered — get it right):
  • Section labels MUST be Markdown headings: "## Ingredients", "## Steps",
    "## Specs", "## Key points". NEVER write a bare "Label:" line — it renders
    as flat body text. Translate the heading text into the target language.
  • Bullets are "- " lines; ordered steps are "1." lines. One item per line,
    short. Use **bold** only for a single key value, sparingly.
- Draw conclusions; be concrete and skimmable. Keep it tight — no fluff, no
  "this page shows…".

{_TOPIC_RULES}{_lang_instruction(lang) or ' Write title and summary in the content language.'}"""

    data = _complete_json(messages=[{"role": "user", "content": prompt}],
                          lang=lang, max_tokens=1400)

    # Hard guard: keep the shelf list short even if the model over-produces.
    topics = [t for t in data.get("topics", []) if isinstance(t, str)][:3]
    category = (data.get("category") or "").strip().lower()
    if category in DOMAIN_TOPICS and category not in [t.lower() for t in topics]:
        topics.append(category)

    return ExtractedAttributes(
        # A web page's own value lives in the synthesized summary + the reopenable
        # link — not the raw scrape, which is site chrome + boilerplate. Storing it
        # as raw_text just surfaces a confusing "Notes" dump, so we leave it empty.
        title=data.get("title", "") or (signals.get("og:title") or signals.get("page title") or url),
        summary=data.get("summary", ""),
        raw_text="",
        capture_type=CaptureType.document,
        topics=topics,
        people=[p for p in data.get("people", []) if isinstance(p, str)][:10],
        places=[pl for pl in data.get("places", []) if isinstance(pl, str)][:10],
        language=data.get("language", "en"),
        sentiment=data.get("sentiment", "neutral"),
        confidence=0.85,
    )


def extractAttributesFromNote(text: str, lang: str = "") -> ExtractedAttributes:
    """
    Attributes for the user's OWN note/idea (typed or dictated). Unlike a web
    page, we stay faithful — extract a title + light summary + topics, never
    rewrite the thought. The note itself is kept verbatim in `raw_text`.
    """
    from src.link_engine import DOMAIN_TOPICS  # local import: avoids any cycle

    client = _claude()
    allowed = ", ".join(sorted(DOMAIN_TOPICS))
    note = text.strip()

    prompt = f"""This is the user's OWN short note saved to their second brain.
Return ONLY valid JSON, no prose. Stay faithful — do NOT rewrite or invent.

Note:
\"\"\"{note[:4000]}\"\"\"

Return JSON:
{{
  "title":    "<3-6 word title capturing the note>",
  "summary":  "<see SUMMARY RULES>",
  "kind":     "<one of: quote|idea|task|fact|question|list|note>",
  "author":   "<if kind=quote and an author is stated/known, their name; else empty>",
  "topics":   ["<1-3 broad topic keywords — see Topics rule>"],
  "category": "<exactly ONE of: {allowed}>",
  "people":   ["<names mentioned, or []>"],
  "places":   ["<places mentioned, or []>"],
  "language": "<ISO-639-1 code>",
  "sentiment":"<positive|negative|neutral>"
}}

SUMMARY RULES — match the shape to the content, but stay faithful (never invent):
- If the note enumerates things (films, books, places, shopping, songs, ideas)
  → format the summary as a clean Markdown list ("- item"), one per line, tidied
  (dedupe, fix casing). You MAY add a 2-4 word tag per item ONLY if it's clearly
  implied by the note; otherwise just the item.
- If it's a to-do list → "- " checklist of the tasks.
- If it's steps/a plan → numbered "1." list.
- Otherwise → a faithful 1-2 sentence gist.
- Use **bold** sparingly for the key thing. Don't invent facts (the verbatim
  note is preserved separately, so the summary may be phrased in another language).

kind: "list" when the note is mainly an enumeration; "quote" only for a
quotation/aphorism; "task" for a to-do; "question" for an open question; "fact"
for a factual snippet; else "idea" or "note".
{_TOPIC_RULES}{_lang_instruction(lang)}"""

    data = _complete_json(messages=[{"role": "user", "content": prompt}],
                          lang=lang, max_tokens=600)

    # Hard guard: keep the shelf list short even if the model over-produces.
    topics = [t for t in data.get("topics", []) if isinstance(t, str)][:3]
    category = (data.get("category") or "").strip().lower()
    if category in DOMAIN_TOPICS and category not in [t.lower() for t in topics]:
        topics.append(category)

    kind = (data.get("kind") or "note").strip().lower()
    if kind not in {"quote", "idea", "task", "fact", "question", "list", "note"}:
        kind = "note"

    return ExtractedAttributes(
        title=data.get("title", "") or (note[:40] + ("…" if len(note) > 40 else "")),
        summary=data.get("summary", "") or note,
        raw_text=note,                       # keep the user's words verbatim
        capture_type=CaptureType.document,
        topics=topics,
        people=[p for p in data.get("people", []) if isinstance(p, str)][:10],
        places=[pl for pl in data.get("places", []) if isinstance(pl, str)][:10],
        language=data.get("language", "en"),
        sentiment=data.get("sentiment", "neutral"),
        kind=kind,
        author=(data.get("author") or "").strip() if kind == "quote" else "",
        confidence=0.9,
    )


def getEmbedding(text: str) -> list[float]:
    """
    Generate a 1024-dimensional embedding vector using Voyage AI
    (voyage-3 model — Anthropic's first-party embedding service).

    Falls back to a zero-vector on error so downstream code stays functional.
    """
    if not text.strip():
        return [0.0] * 1024

    try:
        return _embed_with_retry([text]).embeddings[0]
    except Exception as exc:  # noqa: BLE001
        # Zero vector keeps downstream code alive; the capture is flagged
        # needs_embedding on persist and healed by the background drain.
        logger.error("getEmbedding failed after retries: %s", exc)
        return [0.0] * 1024


def getEmbeddingsBatch(texts: list[str]) -> list[list[float]]:
    """
    Embed many texts in a SINGLE Voyage API request.

    The Voyage free tier is limited to 3 requests/minute, so embedding captures
    one-by-one quickly hits the rate limit. Batching all texts into one call
    avoids that entirely (one request, regardless of how many texts).

    Returns a list of 1024-dim vectors aligned with the input order.
    Empty/blank inputs get a zero-vector; on total failure all are zero-vectors.
    """
    if not texts:
        return []

    # Replace blank texts with a single space (Voyage rejects empty strings)
    safe = [t if t.strip() else " " for t in texts]

    try:
        result = _embed_with_retry(safe)
        out: list[list[float]] = []
        for i, emb in enumerate(result.embeddings):
            out.append([0.0] * 1024 if not texts[i].strip() else emb)
        return out
    except Exception as exc:  # noqa: BLE001
        logger.error("getEmbeddingsBatch failed (%d texts): %s", len(texts), exc)
        return [[0.0] * 1024 for _ in texts]
