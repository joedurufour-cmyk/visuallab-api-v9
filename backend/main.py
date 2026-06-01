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

# CORS — allow any frontend (tighten to your domain in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
API_SECRET   = os.getenv("API_SECRET", "default-secret-change-me")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_BASE    = "https://api.groq.com/openai/v1"

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
# Helpers
# ============================================================
async def groq_chat(messages: list, temperature: float = 0.3, max_tokens: int = 800) -> str:
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured")

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{GROQ_BASE}/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": GROQ_MODEL,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Groq API error: {resp.status_code} {resp.text}")

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
    return {"status": "Visual Lab API v9 · Motion Matrix", "version": "9.0.0"}

@app.get("/health")
async def health():
    return {"ok": True, "groq_configured": bool(GROQ_API_KEY), "version": "9.0.0"}

@app.post("/api/translate", response_model=PromptResponse)
async def translate(req: TranslateRequest, api_key: str = Depends(verify_api_key)):
    user_msg = f'''Translate this intention into a perfect Midjourney V8.1 prompt.

Input: "{req.text}"

Parameters: --ar {req.ar} --raw --stylize {req.stylize} --c {req.chaos} --v 8.1
Mode: {req.mode}

Output only the prompt.'''

    content = await groq_chat([
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

    content = await groq_chat([
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

    content = await groq_chat(
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
