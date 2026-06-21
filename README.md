---
title: AD
emoji: 💙
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Safe Space — Mental Health Support App

A private, RAG-powered mental health companion for people in India living with ADHD and limited financial resources. Built with FastAPI + ChromaDB + Claude.

**Completely private. No login. No data stored permanently.**

---

## Setup (5 steps)

### 1. Install dependencies
```bash
cd mental_health_rag
pip install -r requirements.txt
```
> First install downloads the sentence-transformers model (~90MB). Do this on WiFi.

### 2. Add your API key
```bash
cp .env.example .env
```
Edit `.env` and replace `your_api_key_here` with your Anthropic API key.
Get one at: https://console.anthropic.com

### 3. Build the knowledge base
```bash
python ingest.py
```
This reads all files in `docs/`, chunks them, embeds them, and stores them in `chroma_db/`. Run this once (or whenever you add new documents to `docs/`).

### 4. Start the server
```bash
uvicorn main:app --reload
```

### 5. Open the app
```
http://localhost:8000
```

---

## Adding Your Own Documents

Drop any `.txt`, `.md`, or `.pdf` files into the `docs/` folder, then re-run:
```bash
python ingest.py
```

---

## Project Structure

```
mental_health_rag/
├── main.py              # FastAPI app (API endpoints, Claude streaming)
├── rag.py               # RAG retrieval (ChromaDB queries)
├── ingest.py            # Document ingestion script
├── requirements.txt
├── .env                 # Your API key (never commit this)
├── .env.example
├── .gitignore
├── docs/
│   ├── ADHD_overview.txt
│   ├── coping_strategies.txt
│   ├── india_resources.txt
│   ├── emotional_support.txt
│   ├── relationships_and_adhd.txt
│   └── financial_stress_mental_health.txt
├── static/
│   └── index.html       # Frontend (single file, no framework)
└── chroma_db/           # Created by ingest.py (gitignored)
```

---

## Crisis Helplines (always shown in the app)

| Helpline | Number | Hours |
|---|---|---|
| iCall (TISS) | 9152987821 | Mon-Sat, 8am-10pm, Free |
| Vandrevala Foundation | 1860-2662-345 | 24/7, Free |
| Aasra | 9820466627 | 24/7 |
| iCall Chat | icallhelpline.org | — |

---

## Privacy

- No user accounts. No login required.
- Conversation history is stored **only in your browser session** (RAM).
- When you close the tab, history is gone.
- No data is written to disk except the ChromaDB knowledge base (which contains only the documents you put in `docs/`).

---

## Accessing From Your Phone (Same WiFi)

If your laptop and phone are on the same WiFi:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
Find your laptop's local IP (Settings → WiFi → IP address, e.g. `192.168.1.5`) and open:
```
http://192.168.1.5:8000
```

---

## Remote Access (Laptop Closed / Different Network)

Use VS Code Tunnels to access from anywhere:
```bash
code tunnel
```
Follow the link it provides. Full VS Code + terminal in your browser. Your laptop does all the work.

---

## Troubleshooting

**"API key error"** — Check that `.env` exists and has your `ANTHROPIC_API_KEY`.

**"ChromaDB not ready"** — Run `python ingest.py` first.

**Slow first response** — The embedding model loads on first request (~5-10 seconds). Subsequent requests are fast.

**Port already in use** — `uvicorn main:app --port 8001 --reload`
