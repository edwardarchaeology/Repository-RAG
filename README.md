# Local RAG MVP (Cosmos DB Emulator + Ollama)

A minimal Retrieval-Augmented Generation setup that:

- Ingests local text/code into **Azure Cosmos DB Emulator** (Docker)
- Uses **Ollama** for embeddings and chat
- Supports **incremental ingest**: unchanged files are skipped (quick/safe checks)
- Ships with simple Git helpers to scrape, rank, and clone repos to ingest

> Core CLI lives in `rag_mvp.py` (ingest, search, ask, counts).

---

## Why this Exists

[Robin Cole](https://github.com/robmarkcole) is one of the luminaries of satellite imagery analysis and has a [repository](https://github.com/satellite-image-deep-learning/techniques) collating over 1700 other repositories in this field. I am a single developer with a day job so going through the entire list would be not only temporally impossible but attempting to do so might make my brain melt out of my ears. To avert this brain liquefaction disaster and to show recruiters that what little is left of said grey matter might be worth employing, like all good venture capitalists, I decided AI was the solution. 

A RAG would be able to ingest all of the documentation and code for Robin’s repositories and be able to direct me to the ones I needed whenever I hit a roadblock in my own geospatial analysis workflows. [Abhishek Gupta](https://devblogs.microsoft.com/cosmosdb/build-a-rag-application-with-langchain-and-local-llms-powered-by-ollama/) wrote this fantastic quick start article for getting a simple RAG working based off of local documents. I took a “if it ain’t broke” mindset and used this as a blueprint to set up the basic framework for this project. This article can walk you through setting up docker + the cosmos db emulator + Ollama if you don’t have those yet. 

I want to say thank you again to to Robin and Abhishek as without them I wouldn't have had a problem and a quick solution to said problem. 

## Features

- **Incremental ingest**  
  Quick-skip (`size-mtime`) and strong-skip (`size-mtime-sha1`) so we never re-embed identical files. Meta docs track file signatures.

- **Cosmos DB Emulator-friendly TLS**  
  Auto-fetches `emulator.pem` and prefers Gateway mode so the SDK stays on `https://localhost:8081`. A `--no-verify` escape hatch exists for dev.

- **RAG ask / vector search**  
  Embeds the query with Ollama, runs a vector distance sort in Cosmos, then answers with an LLM on the retrieved chunks.

- **Git data helpers (optional)**  
  - `git_scraper.py`: harvests repos linked from a GitHub README  
  - `git_ranker.py`: scores/filters repos via GitHub API; optional shallow clones  
  - `git_cloner.py`: clones from a CSV of ranked repos  
  Use these to quickly build a local corpus to ingest.

---

## Repo structure

```
.
├─ rag_mvp.py                # CLI: ingest / search / ask / counts
├─ requirements.txt          # minimal deps
├─ git_scraper.py            # collect repo links from a README (HTML crawl)
├─ git_ranker.py             # score/filter repos (GitHub API), optional clone
├─ git_cloner.py             # clone from repos_ranked.csv
├─ data/
│  └─ repos/                 # (optional) cloned repos live here
└─ emulator.pem              # (generated) Cosmos Emulator CA cert (don't commit)
```

---

## Prereqs

- Python 3.10+ (virtualenv recommended)
- Docker Desktop
- **Ollama** installed and running (for embeddings + chat)
  - Models used by default: `mxbai-embed-large`, `gemma2:2b` (pull them once)
- (Optional) GitHub token for better API limits when ranking repos

---

## Quickstart

### 1) Install Python deps

```bash
python -m venv .venv
# Windows PowerShell
. .venv/Scripts/Activate.ps1
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

### 2) Start Cosmos DB Emulator (Docker)

```powershell
docker rm -f cosmos-emulator 2>$null
docker run --name cosmos-emulator `
  -p 8081:8081 -p 10250-10255:10250-10255 `
  -e AZURE_COSMOS_EMULATOR_ENABLE_DATA_PERSISTENCE=false `
  -d mcr.microsoft.com/cosmosdb/linux/azure-cosmos-emulator:latest

# Wait for logs to show "Started 11/11 partitions"
docker logs -f cosmos-emulator
```

Fetch a valid cert **from inside the container** (most reliable):

```powershell
docker exec cosmos-emulator sh -lc "curl -ks https://localhost:8081/_explorer/emulator.pem" > emulator.pem
$env:COSMOS_DB_URL = "https://localhost:8081/"
$env:REQUESTS_CA_BUNDLE = (Resolve-Path .\emulator.pem)
```

> The code also tries to fetch the PEM automatically, but the command above avoids the classic 404 JSON trap. The client uses Gateway mode so it stays on `https://localhost:8081`.

### 3) Prep some data

Make a tiny test file or clone a few repos into `data/repos`:

```powershell
mkdir -Force data | Out-Null
"Hello RAG!" | Out-File data\hello.txt -Encoding utf8
```

(Optionally) build a corpus from a README:

```bash
# Scrape linked repos from a source README (prints & writes referenced_repos.txt)
python git_scraper.py --repo https://github.com/<owner>/<repo> --out-list referenced_repos.txt
# Rank/filter and write repos_ranked.csv (can clone top repos too)
python git_ranker.py --in referenced_repos.txt --out repos_ranked.csv --min-stars 10 --top-n 100 --only-python
# Shallow-clone the ranked repos into data/repos
python git_cloner.py
```

These scripts include concurrency/rate-limit handling and shallow `--depth 1` clones.

### 4) Ingest

```bash
# first run writes; re-runs quick-skip unchanged files
python rag_mvp.py --ingest
# just the first repo under data/ with target files
python rag_mvp.py --ingest --first-repo
# only files modified in last 24h
python rag_mvp.py --ingest --since-hours 24
# count chunk docs
python rag_mvp.py --count
```

Ingest filters out noisy files/dirs and over-long/short chunks. It deletes old chunks for changed files, writes new ones, and upserts meta (signatures & stats).

### 5) Ask / Search

```bash
# vector search
python rag_mvp.py --search "what is GROIE in these configs"

# RAG answer (LLM over retrieved chunks)
python rag_mvp.py --ask "Explain GROIE as used in this repo"
```

The pipeline is: embed query → Cosmos vector distance sort → LLM answers with retrieved context.

---

## Configuration

Environment variables (all optional; see inline defaults):

- `USE_EMULATOR=true`  
- `COSMOS_DB_URL=https://localhost:8081/`  
- `DATABASE_NAME=rag_mvp_db`, `CONTAINER_NAME=docs`  
- `DOCS_DIR=./data`  
- `EMBEDDINGS_MODEL=mxbai-embed-large`, `CHAT_MODEL=gemma2:2b`  
- `CHUNK_SIZE=1200`, `CHUNK_OVERLAP=200`, `TOP_K=5`  
- Write behavior: `WRITE_THROTTLE`, `PAUSE_EVERY`, `PAUSE_SEC`  
- Batching: `EMBED_BATCH`, retries/backoff knobs

CLI flags:

- `--ingest`, `--force`, `--since-hours N`, `--first-repo`  
- `--search "..."`, `--ask "..."`, `--count`  
- `--seed-meta` (backfill meta for existing chunks), `--ping`, `--rebuild-pem`  
- `--no-verify` (emulator/dev only) disables TLS verification for the SDK if needed

All of these are implemented in `rag_mvp.py`.

---

## Known pitfalls & fixes (Cosmos Emulator)

- **Wrong cert file**: `/ _explorer / emulator.pem` can briefly return 404 JSON → not a PEM. Re-fetch until the file contains `-----BEGIN CERTIFICATE-----` or use the `docker exec` approach above. The script has a retry loop too.
- **Hostname mismatch**: Use `https://localhost:8081/` consistently (avoid `127.0.0.1`).
- **Direct-mode timeouts (172.*)**: Client uses **Gateway** mode to avoid dialing container private IPs.
- **“Nothing’s happening” during ingest**: That’s RU throttling (429) and the resilient writer backing off—count may dip while a file is being re-written (delete→insert). Re-runs will skip quickly.

---

## Development notes

- Filters skip big files (>2MB), certain dirs (`cache`, `logs`, `build`, …), and “noise” filenames (metrics/preds/etc.). Chunks with too much punctuation/digits or very short text are dropped.
- Partition key is `"/pk"`; default uses top-level folder under `DOCS_DIR` (or `root`).
- Simple vector search uses `VectorDistance` ordering; a future enhancement is enabling vector indexes when supported by your emulator/SDK build.

---

## Security

- Do **not** commit `emulator.pem`, `.env`, or cloned corpora. See `.gitignore` below.
- The Cosmos **emulator key** is a fixed public dev key; never use this pattern for production.

---

## License

MIT (example). Add your name/year as needed.

---

## Suggested .gitignore

```
# Python
__pycache__/
*.pyc
.venv/

# Local data & artifacts
data/
emulator.pem
*.csv
referenced_repos.txt

# OS / editor
.DS_Store
Thumbs.db
.vscode/
.idea/
```

---

## .env.example (optional)

```
# Emulator & TLS
USE_EMULATOR=true
COSMOS_DB_URL=https://localhost:8081/
EMULATOR_PEM_PATH=./emulator.pem

# Models
EMBEDDINGS_MODEL=mxbai-embed-large
CHAT_MODEL=gemma2:2b

# Ingest
DOCS_DIR=./data
CHUNK_SIZE=1200
CHUNK_OVERLAP=200
TOP_K=5
EMBED_BATCH=48
WRITE_THROTTLE=0.01
PAUSE_EVERY=200
PAUSE_SEC=1.0
```

---

# Example of a Successful Implementation

## Search

<img width="973" height="628" alt="Screenshot 2025-08-25 175233" src="https://github.com/user-attachments/assets/3b61ceb9-a942-4dc5-b581-b1c463147619" />

## Ask

<img width="866" height="165" alt="Screenshot 2025-08-25 175247" src="https://github.com/user-attachments/assets/9169fb20-8217-460a-964f-24878fb3f7a0" />

## Proof that the AI aren't coming for us yet

<img width="934" height="167" alt="image" src="https://github.com/user-attachments/assets/1e6bfb44-0878-459f-a762-23997ae6fc54" />

