# visuallab-api-v9

Backend API for **Visual Lab PRO v9 · Motion Matrix**.  
Identical logic to v8 — new repo name for clean Render deploy.

## Stack
- FastAPI + Uvicorn
- Groq API (llama-3.3-70b-versatile)
- Python 3.11

## Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/` | — | Status |
| GET | `/health` | — | Health + Groq status |
| POST | `/api/translate` | ✅ | Native → MJ prompt |
| POST | `/api/enhance` | ✅ | Refine existing prompt |
| POST | `/api/variations` | ✅ | Generate N variations |

## Deploy to Render

1. Create new repo on GitHub: `visuallab-api-v9`
2. Push this folder
3. On Render: **New Web Service** → connect repo
4. Render auto-detects `render.yaml`
5. Add env vars in Render dashboard:
   - `GROQ_API_KEY` → your key from console.groq.com
   - `API_SECRET` → any strong secret (copy to frontend Config tab)

## Local dev

```bash
cd backend
cp .env.example .env
# fill in your keys
pip install -r requirements.txt
uvicorn main:app --reload
```

## Connect frontend

In Visual Lab PRO v9 → **Config** tab:
- Backend URL: `https://visuallab-api-v9.onrender.com`
- API Secret: same value as `API_SECRET` env var
- Enable Cloud Mode ✅
