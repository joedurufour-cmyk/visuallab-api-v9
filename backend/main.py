from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Literal
import os
import httpx
import json
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Visual Lab API v9 · Motion Matrix", version="9.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
API_SECRET     = os.getenv("API_SECRET", "default-secret-change-me")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o")
OPENAI_BASE    = "https://api.openai.com/v1"

# ============================================================
# Auth
# ============================================================
async def verify_api_key(x_api_key: Optional[str] = Header(None)):
    if not x_api_key or x_api_key != API_SECRET:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key

# ============================================================
# Models
# ============================================================
class TranslateRequest(BaseModel):
    text: str
    mode: Literal["native", "brutal", "anime"] = "native"
    ar: Optional[str] = "9:20"
    chaos: Optional[int] = 15
    stylize: Optional[int] = 250

class EnhanceRequest(BaseModel):
    prompt: str
    instruction: str = "enhance realism and muscular definition hierarchy"
    ar: Optional[str] = "9:20"
    chaos: Optional[int] = 15
    stylize: Optional[int] = 250

class VariationRequest(BaseModel):
    base_prompt: str
    actresses: List[str] = []
    physiques: List[str] = ["Brutal defined"]
    clothing: List[str] = []
    settings: List[str] = []
    lighting: List[str] = []
    count: int = 4
    ar: Optional[str] = "9:20"
    chaos: Optional[int] = 15
    stylize: Optional[int] = 250

class PromptResponse(BaseModel):
    prompt: str
    tokens: int
    meta: dict

class VariationsResponse(BaseModel):
    variations: List[dict]

# ============================================================
# System prompts
# ============================================================
TRANSLATE_SYSTEM = """You are the Visual Lab Native Translator — the world's best Midjourney V8.1 prompt engineer.

RULES (non-negotiable):
1. Output ONLY the Midjourney prompt. No markdown, no code blocks, no explanations.
2. ALWAYS include "female" explicitly in the first 10 tokens when the subject is a woman.
3. Physique vocabulary MUST appear early: abs, rectus abdominis, serratus, oblique, vascular, pores, sweat beads, skin texture.
4. Light must be SPECIFIC: "hard 45° sun", "rim light red", "neon cold from left". Never generic "dramatic lighting".
5. Camera must be specific: "Shot on ARRI Alexa 65 with Cooke S7i 50mm, f/2.8"
6. Use --raw for realism. Stylize 250 = cinema sweet spot. 240 = raw documentary. 280 = balanced.
7. AR 9:20 for phone wallpapers. 9:16 only if explicitly requested.
8. Chaos 15 = increased real density and muscle definition (user's sweet spot).
9. Keep active tokens under 40. Short beats long every time.
10. If the input mentions an actress by name, include full name + brief descriptor.

HIERARCHY (this order matters for MJ token priority):
1. Subject + gender + physique descriptor
2. Clothing + material state
3. Background + environment
4. Light source + camera
5. Style flags + parameters

NEGATIVES (auto-inject if realism mode): --no anime, cartoon, smoothing, plastic, doll, perfect skin, generic

Output format:
<Prompt text> --ar <ar> --raw --stylize <s> --c <c> --v 8.1
"""

ENHANCE_SYSTEM = """You are a Midjourney V8.1 prompt refiner. Your job: take a user's prompt and make it technically perfect without changing their intent.

CHECKLIST:
- Is "female" in the first 10 tokens? If subject is woman, add it.
- Are physique terms (abs, rectus, serratus, oblique) in first 15 tokens? Move them up if not.
- Is lighting specific (angle, source, color)? Replace generic terms.
- Is camera specified (lens, aperture, body)? Add if missing.
- Remove fluff: "very", "extremely", "ultra" — MJ ignores these. Replace with concrete descriptors.
- Ensure --raw is present for realism.
- Keep token count under 45. Cut weak words, never cut anatomy terms.

Output ONLY the refined prompt. No commentary.
"""

VARIATIONS_SYSTEM = """You are a Midjourney prompt variation generator. You receive a base prompt and lists of variables.

Generate exactly N variations by swapping ONE variable at a time while keeping the core composition locked.

Rules:
- Each variation must change only 1-2 elements from base
- Keep physique description locked unless "physique" is the variable being tested
- Maintain AR, stylize, chaos, version across all
- Return as JSON array: [{"changed": "What changed", "prompt": "full prompt"}]

Output valid JSON only. No markdown.
"""

# ============================================================
# Helper — OpenAI chat
# ============================================================
async def openai_chat(messages: list, temperature: float = 0.3, max_tokens: int = 800) -> str:
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{OPENAI_BASE}/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": OPENAI_MODEL,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"OpenAI API error: {resp.status_code} {resp.text}")

    data = resp.json()
    return data["choices"][0]["message"]["content"]


def estimate_tokens(text: str) -> int:
    return len(text.split())


def strip_markdown(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    return t

# ============================================================
# Endpoints
# ============================================================
@app.get("/")
async def root():
    return {"status": "Visual Lab API v9 · Motion Matrix", "version": "9.0.0", "llm": "OpenAI"}

@app.get("/health")
async def health():
    return {"ok": True, "openai_configured": bool(OPENAI_API_KEY), "model": OPENAI_MODEL, "version": "9.0.0"}

@app.post("/api/translate", response_model=PromptResponse)
async def translate(req: TranslateRequest, api_key: str = Depends(verify_api_key)):
    user_msg = f'''Translate this intention into a perfect Midjourney V8.1 prompt.

Input: "{req.text}"

Parameters: --ar {req.ar} --raw --stylize {req.stylize} --c {req.chaos} --v 8.1
Mode: {req.mode}

Output only the prompt.'''

    content = await openai_chat([
        {"role": "system", "content": TRANSLATE_SYSTEM},
        {"role": "user",   "content": user_msg}
    ])

    prompt = strip_markdown(content)
    return PromptResponse(
        prompt=prompt,
        tokens=estimate_tokens(prompt),
        meta={"mode": req.mode, "ar": req.ar, "chaos": req.chaos, "stylize": req.stylize}
    )


@app.post("/api/enhance", response_model=PromptResponse)
async def enhance(req: EnhanceRequest, api_key: str = Depends(verify_api_key)):
    user_msg = f"""Refine this Midjourney prompt:

{req.prompt}

Instruction: {req.instruction}
Keep AR {req.ar}, stylize {req.stylize}, chaos {req.chaos}, version 8.1.

Output only the refined prompt."""

    content = await openai_chat([
        {"role": "system", "content": ENHANCE_SYSTEM},
        {"role": "user",   "content": user_msg}
    ])

    prompt = strip_markdown(content)
    return PromptResponse(
        prompt=prompt,
        tokens=estimate_tokens(prompt),
        meta={"instruction": req.instruction, "ar": req.ar}
    )


@app.post("/api/variations", response_model=VariationsResponse)
async def variations(req: VariationRequest, api_key: str = Depends(verify_api_key)):
    vars_desc = {
        "actresses": req.actresses,
        "physiques":  req.physiques,
        "clothing":   req.clothing,
        "settings":   req.settings,
        "lighting":   req.lighting,
        "count":      req.count,
    }

    user_msg = f"""Base prompt:
{req.base_prompt}

Variables (pick from these to create {req.count} variations, changing 1-2 elements each):
{json.dumps(vars_desc, indent=2)}

Fixed parameters: --ar {req.ar} --raw --stylize {req.stylize} --c {req.chaos} --v 8.1

Return ONLY a JSON array. Example:
[{{"changed": "Anya Taylor-Joy | Tactical Bikini", "prompt": "..."}}]"""

    content = await openai_chat(
        [
            {"role": "system", "content": VARIATIONS_SYSTEM},
            {"role": "user",   "content": user_msg}
        ],
        temperature=0.5,
        max_tokens=2000
    )

    cleaned = strip_markdown(content)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        import re
        match = re.search(r'\[.*\]', cleaned, re.DOTALL)
        if match:
            parsed = json.loads(match.group(0))
        else:
            raise HTTPException(status_code=500, detail="Failed to parse variations JSON")

    return VariationsResponse(variations=parsed)


# ============================================================
# NEXUS NORMALIZER — /api/normalize
# GPT-4o semantic inference on MJ prompt redundancy
# ============================================================
class NormalizeRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 40

class NormalizeResponse(BaseModel):
    original: str
    normalized: str
    token_count_before: int
    token_count_after: int
    changes: List[str]
    layers: dict

NORMALIZE_SYSTEM = """You are NEXUS VISUAL — a Midjourney V8.1 prompt normalizer with deep knowledge of latent space coherence.

Your job: receive a raw MJ prompt (assembled from multiple sources — may have redundancies, conflicts, duplicates) and return a clean, optimized version.

NORMALIZATION RULES:

1. SEMANTIC DEDUPLICATION (most important):
   - Detect semantically equivalent tokens: "ascending flight" + "hover" → keep only "ascending flight" (more specific)
   - "hyper-exaggerated musculature" + "brutal defined abs" → merge into single strongest descriptor
   - "void black background" + "pure black studio" → keep one
   - "hard 45° sun" + "hard studio light 45" → keep one

2. ACTRESS DEDUPLICATION:
   - If more than one actress name appears → keep ONLY the first
   - Remove all subsequent actress names and their descriptors

3. PARAMETER DEDUPLICATION:
   - One value per flag: --ar, --stylize/--s, --c/--chaos, --v, --style
   - If conflict: Motion params take priority over Vars params
   - --raw stays if present
   - --no flags: merge all values into one --no

4. TOKEN HIERARCHY (correct order for MJ attention):
   L1: subject + gender (first 5 tokens — NEVER move these)
   L2: physique descriptors (tokens 6-15)
   L3: motion + expression (tokens 16-22)
   L4: clothing + artifacts (tokens 23-28)
   L5: environment + lighting (tokens 29-34)
   L6: camera (tokens 35-38)
   L7: parameters (always at end)

5. RELEVANCE FILTER (Nexus principle):
   - Remove generic filler: "beautiful", "stunning", "perfect", "amazing", "high quality", "8k", "masterpiece"
   - These carry near-zero weight in MJ V8.1 latent space
   - Keep concrete specifics: measurements, materials, anatomy terms, brand names

6. CONFLICT RESOLUTION:
   - If motion token conflicts with pose token → motion takes priority
   - If two lighting tokens conflict → keep the one with higher specificity
   - If chaos > 40 AND action is "static pose" → flag contradiction, remove static

7. TOKEN LIMIT:
   - Maximum 38 active descriptive tokens (not counting params)
   - If over limit: remove lowest specificity tokens first

OUTPUT FORMAT (JSON only, no markdown):
{
  "normalized": "clean prompt here --params here",
  "changes": ["what was removed/merged and why"],
  "layers": {
    "subject": "tokens here",
    "physique": "tokens here", 
    "motion": "tokens here",
    "clothing": "tokens here",
    "environment": "tokens here",
    "camera": "tokens here",
    "params": "params here"
  },
  "token_count": 32
}"""

@app.post("/api/normalize", response_model=NormalizeResponse)
async def normalize(req: NormalizeRequest, api_key: str = Depends(verify_api_key)):
    token_count_before = len([t for t in req.prompt.split(',') if t.strip() and not t.strip().startswith('--')])

    user_msg = f"""Normalize this Midjourney V8.1 prompt. Apply all rules strictly.

RAW PROMPT:
{req.prompt}

Return ONLY valid JSON. No markdown, no explanation outside the JSON."""

    content = await openai_chat([
        {"role": "system", "content": NORMALIZE_SYSTEM},
        {"role": "user",   "content": user_msg}
    ], temperature=0.1, max_tokens=1200)

    cleaned = strip_markdown(content)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        import re
        match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if match:
            parsed = json.loads(match.group(0))
        else:
            raise HTTPException(status_code=500, detail="Failed to parse normalize response")

    normalized = parsed.get('normalized', req.prompt)
    changes = parsed.get('changes', [])
    layers = parsed.get('layers', {})
    token_count_after = parsed.get('token_count', len([t for t in normalized.split(',') if t.strip() and not t.strip().startswith('--')]))

    return NormalizeResponse(
        original=req.prompt,
        normalized=normalized,
        token_count_before=token_count_before,
        token_count_after=token_count_after,
        changes=changes,
        layers=layers
    )
