# KazaAPI v4 — Uri, the backend-centric AI study assistant

Backend-first FastAPI service powering "Uri," an AI study assistant. All
heavy lifting — LLM calls, web scraping, form submission, plotting, code
execution, memory — happens server-side. `index.html` is a thin, standalone
client that only sends requests and renders SSE streams.

```
kazaapi-v4/
├── main.py                 # FastAPI app, CORS, cron/proactive engine
├── database.py              # async Supabase client wrapper
├── schema.sql                # Supabase DDL (run this first)
├── requirements.txt
├── index.html                 # standalone frontend (host separately)
├── routers/
│   ├── chat.py               # /api/v1/chat (SSE), /history, /reset
│   ├── tools.py               # /api/v1/tools/* (kaza-only)
│   └── config.py               # /api/v1/config (kaza-only)
├── services/
│   ├── llm.py                # Gemini -> Groq -> OpenRouter failover + persona
│   ├── search.py               # free DuckDuckGo search for fact-checking
│   ├── agent.py                # Jina Reader scraping + form-fill agent
│   ├── tools.py                 # plot / Piston / truth-table / SM-2 / Keep mock
│   └── memory.py                # idle compaction + 7-day purge
├── models/
│   └── schemas.py                # all Pydantic request/response models
└── utils/
    ├── crypto.py                  # Fernet encryption for stored API keys
    └── auth.py                     # kaza-token RBAC dependency
```

## 1. Supabase setup (free tier)

1. Create a project at [supabase.com](https://supabase.com) (free tier: 500MB storage).
2. Open **SQL Editor → New query**, paste the contents of `schema.sql`, run it.
   - If your project doesn't have the `pg_cron` extension available, skip the
     final `select cron.schedule(...)` line — the Python-side purge
     (`/api/v1/cron`, pinged externally) covers the same 7-day retention rule.
3. Grab your **Project URL** and **anon/service key** from
   Project Settings → API — you'll need them below.

## 2. Backend deployment (Render free tier)

1. Push this repo to GitHub.
2. On [Render](https://render.com), **New → Web Service**, connect the repo.
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Instance type: **Free** (512MB RAM — the app is written to stay within this:
   Matplotlib uses the headless Agg backend and never writes to disk, scrape
   caching is in-memory and TTL-bounded, and SSE streaming avoids buffering
   full responses).
6. Set environment variables (Render → Environment):

| Variable | Required | Notes |
|---|---|---|
| `SUPABASE_URL` | yes | from Supabase Project Settings → API |
| `SUPABASE_KEY` | yes | anon or service_role key |
| `MASTER_KEY` | yes | Fernet key: `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"` |
| `KAZA_ACCESS_TOKEN` | yes | any long random string — this is *your* password for `role: kaza` access. Generate with `python -c "import secrets; print(secrets.token_urlsafe(32))"` |
| `GEMINI_KEY` | recommended | free key from [Google AI Studio](https://aistudio.google.com/) |
| `GROQ_KEY` | recommended | free key from [console.groq.com](https://console.groq.com/) |
| `OPENROUTER_KEY` | recommended | free key from [openrouter.ai](https://openrouter.ai/) (use their `:free` models) |
| `ALLOWED_ORIGINS` | recommended | comma-separated frontend origin(s), e.g. `https://kazaapi.vercel.app`. Defaults to `*` (fine for testing, loosen later). |

At least one of `GEMINI_KEY` / `GROQ_KEY` / `OPENROUTER_KEY` must be set or
chat will have nothing to fail over between.

7. Deploy. Your API will be live at `https://<your-service>.onrender.com`.
8. **Keep it awake**: Render free instances sleep after 15 minutes of
   inactivity. Point a free [UptimeRobot](https://uptimerobot.com) monitor at
   `https://<your-service>.onrender.com/api/v1/cron` on a ~10-minute interval
   — this both keeps the instance warm and drives the idle/proactive engine
   and the 7-day memory purge.

## 3. Frontend deployment (Vercel / GitHub Pages)

`index.html` is fully standalone — no build step.

- **Vercel**: `vercel deploy` in a folder containing just `index.html`, or
  drag-and-drop the file into a new Vercel project.
- **GitHub Pages**: push `index.html` to a repo, enable Pages on that branch.

On first load, the frontend will ask for your **backend URL** (e.g.
`https://kazaapi-v4.onrender.com`) and store it in memory for the session.
If you want `role: kaza` access, enter your `KAZA_ACCESS_TOKEN` in the
settings panel — it's sent as the `X-Kaza-Token` header, never stored server-side
in plaintext beyond what you configure.

## 4. RBAC — how roles actually work

The client can *claim* `role: kaza` in a chat request, but the backend only
honors that claim if the request also carries a correct `X-Kaza-Token`
header (checked with a constant-time comparison against `KAZA_ACCESS_TOKEN`).
Without it, every request is treated as `role: stranger`: casual
conversation only, no tool execution (`/api/v1/tools/*` 403s outright), no
config access, and nothing gets written to long-term memory.

## 5. Local development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export SUPABASE_URL=... SUPABASE_KEY=... MASTER_KEY=... KAZA_ACCESS_TOKEN=... GROQ_KEY=...
uvicorn main:app --reload --port 8000
```

Then open `index.html` directly in a browser (or serve it with
`python -m http.server 5500`) and point it at `http://localhost:8000`.

## 6. Notes on "100% free" services and their real-world limits

- **Jina Reader** (`r.jina.ai`) — free, no key, but is a shared public
  service; heavy use may get rate-limited. Scrape results are cached
  in-memory for 5 minutes per URL to reduce repeat calls.
- **DuckDuckGo HTML search** — used for fact-checking and as an image-search
  approximation, since there's no official free DuckDuckGo JSON API. This is
  inherently best-effort: DuckDuckGo can change markup or block scraping
  without notice. If that happens, swap `services/search.py` for a proper
  free-tier key (e.g. Jina's `s.jina.ai` search endpoint, also free).
- **Piston API** (`emkc.org`) — free public code execution, no auth, but
  shared and rate-limited; don't expose `/api/v1/tools/run_code` to
  untrusted strangers (it already requires a kaza token by default).
- **Google Keep** has no free public API, so `/api/v1/tools/keep/*` is an
  in-memory mock (`services/tools.py: KeepMock`). Swap it for the free
  **Google Tasks API** (OAuth-based) for a real integration.

## 7. Every route

| Method | Path | Auth |
|---|---|---|
| POST | `/api/v1/chat` | open (role auto-downgraded without token) |
| GET | `/api/v1/chat/history` | open |
| POST | `/api/v1/chat/reset` | open |
| POST | `/api/v1/tools/plot` | kaza |
| POST | `/api/v1/tools/run_code` | kaza |
| GET | `/api/v1/tools/run_code/runtimes` | kaza |
| GET | `/api/v1/tools/logic/truth_table` | kaza |
| POST | `/api/v1/tools/scrape` | kaza |
| POST | `/api/v1/tools/submit_form` | kaza |
| GET | `/api/v1/tools/search_images` | kaza |
| CRUD | `/api/v1/tools/flashcard` | kaza |
| POST | `/api/v1/tools/flashcard/review` | kaza |
| CRUD | `/api/v1/tools/keep/notes` | kaza |
| GET/PUT | `/api/v1/config` | kaza |
| GET | `/api/v1/cron` | open (meant for an external pinger) |
| GET | `/healthz` | open |
