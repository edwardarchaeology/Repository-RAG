# rag_mvp.py
# RAG MVP w/ Cosmos Emulator + Ollama + incremental ingest (skip unchanged files, quick-skip)
# -----------------------------------------------------------------------------
# This script ingests plain-text/code files into Azure Cosmos DB Emulator,
# storing chunk text and vector embeddings for similarity search. It supports:
# - Quick-skip: skip files if size/mtime haven't changed (no disk read)
# - Strong-skip: skip files if full content hash matches prior ingest
# - Resilient Cosmos writes with backoff/retry
# - Vector search via Cosmos's VectorDistance (emulator build supporting it)
# - Basic RAG "ask" that retrieves top-k chunks and answers with an LLM
# -----------------------------------------------------------------------------

# =======================
# Imports
# =======================
import os
import re
import glob
import time
import random
import socket
import hashlib
import http.client
from typing import List, Iterable, Tuple, Optional

from dotenv import load_dotenv
import requests

from azure.cosmos import CosmosClient, PartitionKey
from azure.cosmos.exceptions import (
    CosmosHttpResponseError,
    CosmosResourceNotFoundError,
)
from azure.core.exceptions import ServiceRequestError, ServiceResponseError

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.schema import SystemMessage, HumanMessage

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)  # avoid noisy emulator TLS warnings


# =======================
# Config / Environment
# =======================
load_dotenv()  # Load .env variables into the process environment

CLI_NO_VERIFY = False  # toggled by --no-verify; disables TLS verification in Cosmos SDK

# Cosmos Emulator-only configuration
USE_EMULATOR   = os.getenv("USE_EMULATOR", "true").lower() == "true"
COSMOS_DB_URL  = os.getenv("COSMOS_DB_URL", "https://localhost:8081/")
DATABASE_NAME  = os.getenv("DATABASE_NAME", "rag_mvp_db")
CONTAINER_NAME = os.getenv("CONTAINER_NAME", "docs")

# Local Ollama model names (ensure these are available in your Ollama)
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "mxbai-embed-large")
CHAT_MODEL       = os.getenv("CHAT_MODEL", "gemma2:2b")

# Ingestion & retrieval settings
DOCS_DIR      = os.getenv("DOCS_DIR", "./data")  # root folder to scan
CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K         = int(os.getenv("TOP_K", "5"))

# Ingest tunables: file patterns and code chunking (often needs bigger chunks)
DOC_EXTS        = ["*.md", "*.txt", "*.rst"]
CODE_EXTS       = ["*.py", "*.ipynb"]
CODE_CHUNK_SIZE = 2000
CODE_OVERLAP    = 200

# Embedding batching & retry
EMBED_BATCH     = int(os.getenv("EMBED_BATCH", "48"))
EMBED_RETRIES   = 3
EMBED_BACKOFF   = 0.25  # base backoff seconds, exponential

# Write pacing to reduce emulator thrash
WRITE_THROTTLE  = float(os.getenv("WRITE_THROTTLE", "0.01"))  # sleep after each write
PAUSE_EVERY     = int(os.getenv("PAUSE_EVERY", "200"))        # micro-pause after N writes
PAUSE_SEC       = float(os.getenv("PAUSE_SEC", "1.0"))

# Retry/backoff settings for transient errors
TRANSIENT_STATUSES = {408, 429, 503}
MAX_TRIES       = 12
BASE_SLEEP      = 0.25

# Where to store the emulator CA certificate locally
EMULATOR_PEM_PATH = os.getenv("EMULATOR_PEM_PATH", os.path.join(os.getcwd(), "emulator.pem"))

_write_count = 0  # module-level count of successful upserts (for pacing)


# =======================
# Ingestion Filters
# =======================
# These filters help avoid ingesting huge or noisy folders/files that don't help RAG
EXCLUDE_DIR_NAMES = {
    "datasets", "dataset", "samples", "sample_data", "cache",
    "checkpoints", "ckpt", "outputs", "artifacts", "logs",
    "log", "tmp", "temp", "generated", "build"
}

# Regex patterns for filenames to skip (common ML bookkeeping files)
EXCLUDE_NAME_PATTERNS = [
    r"^train[_\-].*\.txt$", r"^.*[_\-]train\.txt$",
    r"^test[_\-].*\.txt$",  r"^.*[_\-]test\.txt$",
    r"^val[_\-].*\.txt$",   r"^.*[_\-]val\.txt$",
    r"^results?[_\-].*\.txt$", r"^metrics?[_\-].*\.txt$",
    r"^loss(?:es)?[_\-].*\.txt$", r"^pred(?:ictions)?[_\-].*\.txt$",
    r"^labels?[_\-].*\.txt$", r"^output[_\-].*\.txt$",
    r"^.*[_\-](labels?|preds?|metrics?)\.txt$",
]
MAX_FILE_BYTES = 2 * 1024 * 1024  # hard cap on file size (2 MB) to keep MVP simple
_name_res = [re.compile(pat, re.IGNORECASE) for pat in EXCLUDE_NAME_PATTERNS]


# =======================
# Helper Predicates (filtering/noise checks)
# =======================
def _is_transient_cosmos(e: Exception) -> bool:
    """
    Heuristics to decide if a Cosmos exception is likely transient (worth retrying).
    Considers HTTP status, Azure core request/response errors, and socket-level issues.
    """
    status = getattr(e, "status_code", None)
    if status in TRANSIENT_STATUSES:
        return True
    if isinstance(e, (ServiceRequestError, ServiceResponseError)):
        return True
    if isinstance(e, (http.client.RemoteDisconnected, ConnectionError, socket.timeout)):
        return True
    return False

def _path_has_excluded_dir(path: str) -> bool:
    """
    Return True if any component of the path (relative to DOCS_DIR) is in EXCLUDE_DIR_NAMES.
    We skip the last path component because that's the filename, not a directory.
    """
    try:
        rel = os.path.relpath(path, DOCS_DIR)
    except Exception:
        rel = path
    if rel.startswith(".."):  # outside root
        return False
    parts = [p.lower() for p in os.path.normpath(rel).split(os.sep)]
    return any(p in EXCLUDE_DIR_NAMES for p in parts[:-1])

def should_ingest_file(path: str) -> bool:
    """
    Apply filters before reading/embedding a file:
      - Size limit
      - Excluded directories
      - Filename regex rules
    """
    try:
        if os.path.getsize(path) > MAX_FILE_BYTES:
            return False
    except Exception:
        # If stat fails, let it through—we'll fail later when reading if it's truly bad
        pass
    if _path_has_excluded_dir(path):
        return False
    fname = os.path.basename(path)
    for rx in _name_res:
        if rx.match(fname):
            return False
    return True

def chunk_noise_score(text: str) -> float:
    """
    A trivial 'noise' metric: ratio of digits+punctuation to total characters.
    Used to filter out chunks that look like logs or binary garbage.
    """
    if not text:
        return 1.0
    nums = sum(ch.isdigit() for ch in text)
    punct = sum(ch in r"""!@#$%^&*()_+-=[]{}|;:'",.<>/?\`~""" for ch in text)
    return (nums + punct) / max(1, len(text))

def should_ingest_chunk(text: str) -> bool:
    """
    Decide whether a chunk is worth storing/embedding:
      - Reject if 'noisy' by heuristic
      - Reject very short chunks (likely headers or tiny fragments)
      - Reject chunks with a single extremely long line (minified or junk)
    """
    if chunk_noise_score(text) >= 0.75:
        return False
    if len(text.strip()) < 40:
        return False
    lines = text.splitlines()
    if lines and max(len(ln) for ln in lines) > 2000:
        return False
    return True


# =======================
# TLS & Cosmos Emulator Helpers
# =======================
def _try_fetch_pem(host: str, dest: str) -> bool:
    """
    Try to download the Cosmos Emulator Explorer certificate from a host
    and save to 'dest'. Returns True on success (certificate text present).
    """
    url = f"https://{host}:8081/_explorer/emulator.pem"
    try:
        r = requests.get(url, verify=False, timeout=10)  # skip verify to bootstrap trust
        if r.ok and b"BEGIN CERTIFICATE" in r.content:
            with open(dest, "wb") as f:
                f.write(r.content)
            return True
    except Exception:
        pass
    return False

def ensure_emulator_ca() -> str:
    """
    Ensure 'emulator.pem' is downloaded locally and set REQUESTS_CA_BUNDLE
    so Python requests (and Cosmos SDK 'connection_cert') can trust the emulator.
    Will keep trying for ~60s because the Explorer endpoint comes up a bit late.
    """
    pem_path = EMULATOR_PEM_PATH
    if os.path.isfile(pem_path):
        try:
            with open(pem_path, "rb") as f:
                if b"BEGIN CERTIFICATE" in f.read(65536):
                    os.environ["REQUESTS_CA_BUNDLE"] = pem_path
                    return pem_path
        except Exception:
            pass

    deadline = time.time() + 60
    while time.time() < deadline:
        if _try_fetch_pem("localhost", pem_path) or _try_fetch_pem("127.0.0.1", pem_path):
            os.environ["REQUESTS_CA_BUNDLE"] = pem_path
            return pem_path
        time.sleep(1.5)
    raise RuntimeError("Could not fetch emulator.pem (Explorer not ready yet).")

def cosmos_https_up(timeout=3.0) -> bool:
    """
    Probe the Explorer HTTP(S) endpoint. Any 'normal' status means the TLS socket is live.
    This doesn't validate certs—it's just a liveness check.
    """
    try:
        r = requests.get(f"{COSMOS_DB_URL.rstrip('/')}/_explorer/index.html",
                         verify=False, timeout=timeout)
        return r.status_code in (200, 302, 401, 403, 405)
    except Exception:
        return False

def wait_for_cosmos(max_wait: float = 240.0):
    """
    Wait for the emulator to be fully ready in two phases:
      1) Explorer HTTPS responds (socket up).
      2) Cosmos SDK can perform a trivial query.
    Raises TimeoutError if readiness doesn't happen within max_wait.
    """
    end = time.time() + max_wait
    last_err = None

    # Phase 1: Explorer reachable
    while time.time() < end:
        if cosmos_https_up(timeout=2.0):
            break
        time.sleep(2.0)
    else:
        raise TimeoutError("Explorer HTTPS endpoint never became reachable.")

    # Phase 2: SDK trivial op
    global container
    while time.time() < end:
        try:
            container = get_container()  # (re)create client/db/container
            list(container.query_items(
                query="SELECT TOP 1 c.id FROM c",
                enable_cross_partition_query=True
            ))
            return
        except Exception as e:
            last_err = e
            container = None
            time.sleep(2.0)

    raise TimeoutError(f"Cosmos Emulator not ready. Last error: {last_err!r}")

def get_cosmos_client():
    """
    Construct a CosmosClient for the emulator.
    If --no-verify was passed (CLI_NO_VERIFY), skip TLS verification entirely.
    Otherwise, download the emulator CA and use it to verify TLS.
    """
    if not USE_EMULATOR:
        raise RuntimeError("Only emulator is wired in this MVP.")

    endpoint = COSMOS_DB_URL.rstrip("/")
    # Default emulator key (public in docs; never use in production)
    key = ("C2y6yDjf5/R+ob0N8A7Cgv30VRDJIWEHLM+4QDU5DE2nQ9nDuVTqobD4b8mGGyPMb"
           "IZnqyMsEcaGQy67XIw/Jw==")

    # If user explicitly asked to skip verification
    if 'CLI_NO_VERIFY' in globals() and CLI_NO_VERIFY:
        return CosmosClient(
            endpoint,
            credential=key,
            consistency_level="Session",
            connection_mode="Gateway",     # Important in emulator; Direct can try 172.17.x.x
            connection_verify=False,       # don't verify TLS
            enable_endpoint_discovery=False,  # avoid internal IPs
            connection_timeout=60,
            timeout=120,
        )

    # Normal path: download cert and use it
    try:
        ca_path = ensure_emulator_ca()
        return CosmosClient(
            endpoint,
            credential=key,
            consistency_level="Session",
            connection_mode="Gateway",
            connection_verify=True,        # verify TLS
            connection_cert=ca_path,       # trust the emulator's CA
            enable_endpoint_discovery=False,
            connection_timeout=60,
            timeout=120,
        )
    except Exception as e:
        # If cert flow fails, fall back (still usable for local dev)
        print(f"[warn] TLS configure failed ({e}); falling back to no-verify.")
        return CosmosClient(
            endpoint,
            credential=key,
            consistency_level="Session",
            connection_mode="Gateway",
            connection_verify=False,
            enable_endpoint_discovery=False,
            connection_timeout=60,
            timeout=120,
        )

def get_container():
    """
    Create (or open) the database and container if they don't exist.
    Uses a simple partition key '/pk'. Index policy is broad for MVP.
    """
    client = get_cosmos_client()
    db = client.create_database_if_not_exists(DATABASE_NAME)
    container = db.create_container_if_not_exists(
        id=CONTAINER_NAME,
        partition_key=PartitionKey(path="/pk"),
        indexing_policy={
            "automatic": True,
            "indexingMode": "consistent",
            "includedPaths": [{"path": "/*"}],
            "excludedPaths": [{"path": "/_etag/?"}],
            # Vector index config is emulator/SDK-version-specific; omitted in MVP.
        },
    )
    return container


# =======================
# Globals (lazy init)
# =======================
container = None  # will be created on first use
embedder  = OllamaEmbeddings(model=EMBEDDINGS_MODEL)  # local embeddings via ollama
llm       = ChatOllama(model=CHAT_MODEL, temperature=0.2)  # local chat LLM via ollama

def cosmos():
    """
    Return a cached Cosmos container (create on first call).
    Keeps a module-global handle that can be reset by retry logic.
    """
    global container
    if container is None:
        container = get_container()
    return container


# =======================
# Text Utilities
# =======================
def read_text_file(path: str) -> str:
    """Read a text file as UTF-8 (ignore errors to survive odd encodings)."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def chunk_text(text: str, size: int, overlap: int) -> List[str]:
    """
    Simple fixed-size sliding window chunker.
    - size: chunk length
    - overlap: overlap between consecutive chunks
    """
    out, i, step = [], 0, max(1, size - overlap)
    while i < len(text):
        out.append(text[i:i+size])
        i += step
    return out


# =======================
# ID / Partition Key Helpers
# =======================
def _id_prefix_for(fp: str) -> str:
    """
    Build a stable, file-relative prefix for item IDs by using the relative path
    and replacing unsafe characters with underscores.
    """
    rel = os.path.relpath(fp, DOCS_DIR)
    rel = rel.replace("\\", "_").replace("/", "_").replace("?", "_").replace("#", "_")
    return re.sub(r"[^a-zA-Z0-9._-]", "_", rel)

def safe_id(path: str, i: int) -> str:
    """Construct a unique chunk ID: '<relpath_sanitized>::<chunk_index>'."""
    return f"{_id_prefix_for(path)}::{i}"

def meta_id_for(fp: str) -> str:
    """Construct the meta document ID for a file: 'meta::<relpath_sanitized>'."""
    return f"meta::{_id_prefix_for(fp)}"

def pick_pk_for(path: str) -> str:
    """
    Pick a partition key based on the top-level directory under DOCS_DIR.
    - If the file is at root, use 'root'.
    - Else use the name of the first segment (good distribution heuristic).
    """
    rel = os.path.relpath(path, DOCS_DIR)
    parts = os.path.normpath(rel).split(os.sep)
    return (parts[0] if parts and parts[0] not in (".", "") else "root")


# =======================
# Ingest Cache Helpers (quick-skip / strong-skip)
# =======================
def _stat_tuple(fp: str) -> Tuple[int, int]:
    """Return (size, mtime_int) without reading file content."""
    st = os.stat(fp)
    return st.st_size, int(st.st_mtime)

def strong_signature(fp: str) -> str:
    """
    Compute a strong signature 'size-mtime-sha1' for a file.
    SHA1 is fine for change detection (not used for security).
    """
    st = os.stat(fp)
    size = st.st_size
    mtime = int(st.st_mtime)
    h = hashlib.sha1()
    # Read in 1MB chunks to avoid loading huge files into memory
    with open(fp, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return f"{size}-{mtime}-{h.hexdigest()}"

def quick_sig_from_strong(sig: str) -> Optional[str]:
    """
    Derive a quick 'size-mtime' signature from a legacy 'size-mtime-sha1' string.
    Useful for backward compatibility with older meta records.
    """
    try:
        size, mtime, _ = sig.split("-", 2)
        return f"{int(size)}-{int(mtime)}"
    except Exception:
        return None

# Backward-compat alias
file_signature = strong_signature

def read_file_meta(fp: str, pk: str):
    """
    Load the meta document for a given file. Return None if missing.
    Meta holds the last seen signature and chunk count.
    """
    mid = meta_id_for(fp)
    try:
        return cosmos().read_item(item=mid, partition_key=pk)
    except CosmosResourceNotFoundError:
        return None

def upsert_file_meta(fp: str, pk: str, signature: str, n_chunks: int):
    """
    Create/update the meta document for a file to record:
      - strong signature
      - derived quick signature
      - file size/mtime
      - number of chunks last ingested
      - updated_ts (unix seconds)
    """
    size, mtime = _stat_tuple(fp)
    item = {
        "id": meta_id_for(fp),
        "pk": pk,
        "type": "meta",
        "path": os.path.relpath(fp, DOCS_DIR),
        "signature": signature,                 # strong "size-mtime-sha1"
        "quick_sig": f"{size}-{mtime}",         # quick "size-mtime"
        "size": int(size),
        "mtime": int(mtime),
        "n_chunks": int(n_chunks),
        "updated_ts": int(time.time()),
    }
    upsert_doc_resilient(item)

def delete_chunks_for_path(fp: str, pk: str):
    """
    Delete all chunk documents for a given file path in the specified partition.
    This is called before re-ingesting changed files to avoid stale chunks.
    """
    rel = os.path.relpath(fp, DOCS_DIR)
    q = """
    SELECT c.id FROM c
    WHERE c.pk = @pk AND c.path = @path
          AND (c.type = 'chunk' OR NOT IS_DEFINED(c.type))
    """
    params = [{"name": "@pk", "value": pk}, {"name": "@path", "value": rel}]
    for row in cosmos().query_items(query=q, parameters=params, enable_cross_partition_query=True):
        try:
            cosmos().delete_item(item=row["id"], partition_key=pk)
            if WRITE_THROTTLE:
                time.sleep(WRITE_THROTTLE)  # gently pace deletes
        except Exception:
            # If a delete fails transiently, ignore; next ingest will overwrite anyway.
            pass

def seed_meta_for_existing(skip_if_meta_exists: bool = True):
    """
    Scan for files that already have chunk docs but no meta doc (e.g., from older runs)
    and create their meta docs so future ingests can quick/strong skip them.
    """
    files = gather_files(DOCS_DIR, DOC_EXTS) + gather_files(DOCS_DIR, CODE_EXTS)
    seeded = 0
    for fp in files:
        if not should_ingest_file(fp):
            continue
        pk = pick_pk_for(fp)
        if skip_if_meta_exists and read_file_meta(fp, pk):
            continue
        rel = os.path.relpath(fp, DOCS_DIR)
        q = """
        SELECT VALUE COUNT(1)
        FROM c
        WHERE c.pk = @pk AND c.path = @path
              AND (c.type = 'chunk' OR NOT IS_DEFINED(c.type))
        """
        params = [{"name":"@pk","value":pk},{"name":"@path","value":rel}]
        res = list(cosmos().query_items(query=q, parameters=params, enable_cross_partition_query=True))
        n_chunks = int(res[0]) if res else 0
        if n_chunks > 0:
            upsert_file_meta(fp, pk, file_signature(fp), n_chunks)
            seeded += 1
    print(f"Seeded meta for {seeded} file(s).")


# =======================
# Resilient Writes (Cosmos retry loop)
# =======================
def upsert_doc_resilient(item: dict):
    """
    Upsert an item with retry/backoff on transient failures.
    Also:
      - Applies WRITE_THROTTLE to reduce emulator pressure.
      - Micro-pauses every PAUSE_EVERY writes to let emulator breathe.
      - If the connection appears refused, nulls the cached container so the
        next call recreates the client/container.
    """
    global _write_count, container
    attempt = 0
    while True:
        try:
            if WRITE_THROTTLE:
                time.sleep(WRITE_THROTTLE)
            cosmos().upsert_item(item)
            _write_count += 1
            if PAUSE_EVERY and (_write_count % PAUSE_EVERY == 0):
                time.sleep(PAUSE_SEC)
            return
        except (CosmosHttpResponseError, ServiceRequestError, ServiceResponseError,
                http.client.RemoteDisconnected, ConnectionError, socket.timeout) as e:
            msg = str(e)
            # If the emulator refuses the socket, force a fresh client/container on next call
            if ("WinError 10061" in msg) or ("actively refused" in msg) or ("Failed to establish a new connection" in msg):
                try:
                    time.sleep(1.0)
                    container = None  # force re-create on next cosmos()
                except Exception:
                    pass
            if _is_transient_cosmos(e) and attempt < MAX_TRIES:
                # Backoff: prefer server-provided retry_after, else exponential jitter
                retry_after = getattr(e, "retry_after", None)
                delay = float(retry_after) if retry_after else (BASE_SLEEP * (2 ** attempt)) * (1.0 + random.random()*0.3)
                attempt += 1
                time.sleep(delay)
                continue
            # Non-transient or too many attempts: surface the error
            raise


# =======================
# Embedding Helpers
# =======================
def embed_documents_with_retry(batch: List[str]) -> List[List[float]]:
    """
    Embed a batch of strings using Ollama embeddings with retry/backoff.
    Returns a list of vectors aligned with input order.
    """
    attempt = 0
    while True:
        try:
            return embedder.embed_documents(batch)
        except Exception:
            if attempt >= EMBED_RETRIES:
                raise
            time.sleep(EMBED_BACKOFF * (2 ** attempt))
            attempt += 1

def embed_in_batches(pieces: List[str]) -> Iterable[Tuple[str, List[float]]]:
    """
    Generator that yields (piece, vector) pairs by slicing 'pieces' into batches,
    embedding each batch, and then zipping the results back to the original text.
    """
    for i in range(0, len(pieces), EMBED_BATCH):
        batch = pieces[i:i+EMBED_BATCH]
        vecs = embed_documents_with_retry(batch)
        for piece, vec in zip(batch, vecs):
            yield piece, vec


# =======================
# File Discovery & Ingestion
# =======================
def _top_repo_segment(fp: str) -> str:
    """
    Return the first path segment under DOCS_DIR (useful to treat the top-level
    folder as a 'repo' name). For root-level files, return "".
    """
    rel = os.path.relpath(fp, DOCS_DIR)
    seg = os.path.normpath(rel).split(os.sep)[0]
    return "" if seg in (".", "") else seg

def pick_first_repo_with_targets() -> Optional[str]:
    """
    Return the name of the first top-level folder (alphabetical) that contains
    any ingestable target files. If only root-level files exist, return "".
    If no files exist at all, return None.
    """
    files = gather_files(DOCS_DIR, DOC_EXTS) + gather_files(DOCS_DIR, CODE_EXTS)
    if not files:
        return None
    repos = sorted({_top_repo_segment(fp) for fp in files if _top_repo_segment(fp)})
    if repos:
        return repos[0]
    # fallback: only root-level files exist
    return ""

def gather_files(root: str, patterns: List[str]) -> List[str]:
    """
    Recursively find all files under 'root' matching any glob pattern in 'patterns',
    then apply 'should_ingest_file' filter and return sorted list.
    """
    files: List[str] = []
    for pat in patterns:
        files += glob.glob(os.path.join(root, "**", pat), recursive=True)
    files = [fp for fp in files if should_ingest_file(fp)]
    return sorted(files)

def make_item(fp: str, idx: int, text: str, embedding: List[float]) -> dict:
    """
    Create a Cosmos item representing a retrievable chunk with an embedding vector.
    """
    return {
        "id": safe_id(fp, idx),
        "pk": pick_pk_for(fp),
        "type": "chunk",  # retrievable
        "path": os.path.relpath(fp, DOCS_DIR),
        "text": text,
        "embedding": embedding,
        "meta": {"size": len(text), "ext": os.path.splitext(fp)[1].lower()},
    }

def ingest_one_file(fp: str, size: int, overlap: int, force: bool = False) -> int:
    """
    Ingest a single file into chunk docs + meta doc:
      1) Quick-skip: compare 'size-mtime' (no file read)
      2) Strong-check: compute 'size-mtime-sha1' if quick changed
      3) If changed, delete old chunks, re-chunk, embed in batches, upsert, then update meta.
    Returns the number of chunks written (0 if skipped).
    """
    if not should_ingest_file(fp):
        return 0

    pk = pick_pk_for(fp)
    meta = read_file_meta(fp, pk)

    # --- QUICK SKIP: compare "size-mtime" without reading file contents ---
    fsize, fmtime = _stat_tuple(fp)
    quick_now = f"{fsize}-{fmtime}"

    meta_quick = None
    if meta:
        meta_quick = meta.get("quick_sig") or quick_sig_from_strong(meta.get("signature", ""))

    if (not force) and meta_quick and meta_quick == quick_now:
        print(f"SKIP (quick) {fp}")
        return 0

    # --- STRONG CHECK: only hash if stat changed or no meta exists ---
    sig_now = file_signature(fp)  # strong size-mtime-sha1
    if (not force) and meta and (meta.get("signature") == sig_now):
        print(f"SKIP (strong) {fp}")
        # Refresh quick fields if they were missing
        upsert_file_meta(fp, pk, sig_now, meta.get("n_chunks", 0))
        return 0

    # Changed or new → clean old chunks, (re)chunk & embed
    delete_chunks_for_path(fp, pk)

    # Read file and produce candidate chunks
    raw = read_text_file(fp)
    pieces_all = chunk_text(raw, size, overlap)
    pieces = [p for p in pieces_all if should_ingest_chunk(p)]
    if not pieces:
        # still record meta so future runs can quick-skip
        upsert_file_meta(fp, pk, sig_now, 0)
        return 0

    wrote = 0
    idx = 0
    # Embed in batches; upsert items one by one (easier to handle partial failures)
    for piece, vec in embed_in_batches(pieces):
        item = make_item(fp, idx, piece, vec)
        upsert_doc_resilient(item)
        wrote += 1
        idx += 1

    # Update meta with the final signature and chunk count
    upsert_file_meta(fp, pk, sig_now, wrote)
    return wrote

def _filter_by_mtime(files: List[str], cutoff_ts: Optional[float]) -> List[str]:
    """
    Filter a list of file paths by modification time >= cutoff_ts.
    If cutoff_ts is None, return the input list unchanged.
    """
    if cutoff_ts is None:
        return files
    kept: List[str] = []
    for fp in files:
        try:
            if os.path.getmtime(fp) >= cutoff_ts:
                kept.append(fp)
        except Exception:
            pass
    return kept

def ingest_dir(force: bool = False, since_hours: Optional[float] = None, only_repo: Optional[str] = None):
    """
    Ingest all target files under DOCS_DIR, with optional:
      - force: reingest even if unchanged
      - since_hours: only consider files modified in the last N hours
      - only_repo: limit to a specific top-level folder ('' means root-only)
    """
    # Gather eligible files by pattern (then filtered via should_ingest_file)
    doc_files  = gather_files(DOCS_DIR, DOC_EXTS)
    code_files = gather_files(DOCS_DIR, CODE_EXTS)

    # Optionally filter by mtime (relative cutoff)
    cutoff = (time.time() - since_hours * 3600.0) if since_hours else None

    def _filter_by_mtime_local(files, cutoff_ts):
        if cutoff_ts is None:
            return files
        kept = []
        for fp in files:
            try:
                if os.path.getmtime(fp) >= cutoff_ts:
                    kept.append(fp)
            except Exception:
                pass
        return kept

    doc_files  = _filter_by_mtime_local(doc_files,  cutoff)
    code_files = _filter_by_mtime_local(code_files, cutoff)

    # Optionally limit to a single 'repo' (top-level folder) or to root files only
    if only_repo is not None:
        if only_repo == "":
            # keep root-level files only (no subdir segment)
            doc_files  = [fp for fp in doc_files  if _top_repo_segment(fp) == ""]
            code_files = [fp for fp in code_files if _top_repo_segment(fp) == ""]
        else:
            doc_files  = [fp for fp in doc_files  if _top_repo_segment(fp) == only_repo]
            code_files = [fp for fp in code_files if _top_repo_segment(fp) == only_repo]
        print(f"Selected repo: {only_repo or '(root)'}")

    if not (doc_files or code_files):
        print(f"No target files found in {DOCS_DIR}" + ("" if only_repo is None else f" for repo '{only_repo or '(root)'}'"))
        return

    total = 0
    # Ingest doc-like files
    for fp in doc_files:
        n = ingest_one_file(fp, CHUNK_SIZE, CHUNK_OVERLAP, force=force)
        total += n
        print(f"Ingested DOC  {fp} ({n} chunks)")
    # Ingest code-like files
    for fp in code_files:
        n = ingest_one_file(fp, CODE_CHUNK_SIZE, CODE_OVERLAP, force=force)
        total += n
        print(f"Ingested CODE {fp} ({n} chunks)")

    print(f"Done. Total chunks written this run: {total}")


# =======================
# Vector Search / RAG
# =======================
def vector_search(query_text: str, top_k: int = TOP_K):
    """
    Perform a vector search in Cosmos using VectorDistance:
      - Embed the query text
      - Use a SQL-like query ordering by VectorDistance to the query vector
    Returns a list of items with 'id', 'text', and 'distance'.
    """
    qvec = embedder.embed_query(query_text)
    query_sql = """
    SELECT TOP @top_k
      c.id, c.text,
      VectorDistance(c.embedding, @q, false) AS distance
    FROM c
    WHERE (c.type = 'chunk' OR NOT IS_DEFINED(c.type))
    ORDER BY VectorDistance(c.embedding, @q, false)
    """
    params = [{"name": "@top_k", "value": int(top_k)}, {"name": "@q", "value": qvec}]
    return list(cosmos().query_items(
        query=query_sql,
        parameters=params,
        enable_cross_partition_query=True
    ))

# System prompt for constrained RAG answers
SYSTEM = "You are a helpful assistant. Answer using only the provided context. If unsure, say you don't know."

def rag_answer(question: str, top_k: int = TOP_K) -> str:
    """
    Retrieve top_k chunks for a question and send them (as 'Context') to the local LLM.
    The LLM is instructed to only use the provided context and be concise.
    """
    hits = vector_search(question, top_k=top_k)
    context = "\n\n---\n\n".join([h.get("text", "") for h in hits])
    msgs = [
        SystemMessage(content=SYSTEM),
        HumanMessage(content=f"Question: {question}\n\nContext:\n{context}\n\nAnswer concisely.")
    ]
    resp = llm.invoke(msgs)

    # Print the retrieved items and their distances to help debug relevance
    print("\n=== Retrieved ===")
    for h in hits:
        dist = h.get("distance")
        if isinstance(dist, (int, float)):
            print(f"- {h['id']} (distance={dist:.4f})")
        else:
            print(f"- {h['id']}")

    return resp.content

def count_chunks() -> int:
    """
    Return the number of retrievable chunk docs in the container.
    """
    q = "SELECT VALUE COUNT(1) FROM c WHERE (c.type = 'chunk' OR NOT IS_DEFINED(c.type))"
    res = list(cosmos().query_items(query=q, enable_cross_partition_query=True))
    return int(res[0]) if res else 0


# =======================
# CLI
# =======================
if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser(description="RAG MVP with Cosmos Emulator + Ollama + incremental ingest")
    # --- CLI flags ---
    parser.add_argument("--ingest", action="store_true", help="Ingest files from DOCS_DIR (quick-skip unchanged)")
    parser.add_argument("--force", action="store_true", help="Reingest even if file signature unchanged")
    parser.add_argument("--since-hours", type=float, help="Only consider files modified in the last N hours")
    parser.add_argument("--seed-meta", action="store_true", help="Create meta docs for already-stored files")
    parser.add_argument("--search", type=str, help="Vector search query")
    parser.add_argument("--ask", type=str, help="RAG question")
    parser.add_argument("--ping", action="store_true", help="Test emulator HTTPS/TLS")
    parser.add_argument("--rebuild-pem", action="store_true", help="Force re-download emulator.pem")
    parser.add_argument("--count", action="store_true", help="Count retrievable chunk docs")
    parser.add_argument("--first-repo", action="store_true",
                        help="Ingest only the first top-level folder (repo) that contains target files, then exit")
    parser.add_argument("--no-verify", action="store_true",
                        help="(Emulator) Disable TLS verification for the Cosmos SDK connection")

    args = parser.parse_args()

    # Honor --no-verify for Cosmos SDK (disables TLS verification)
    if args.no_verify:
        CLI_NO_VERIFY = True

    # One-shot: re-download emulator.pem for a clean TLS state
    if args.rebuild_pem:
        try:
            if os.path.exists(EMULATOR_PEM_PATH):
                os.remove(EMULATOR_PEM_PATH)
            ensure_emulator_ca()
            print(f"Rebuilt emulator PEM at {EMULATOR_PEM_PATH}")
        except Exception as e:
            print(f"Failed to rebuild PEM: {e}")
        sys.exit(0)

    # One-shot: simple HTTPS probe of Explorer
    if args.ping:
        url = f"{COSMOS_DB_URL.rstrip('/')}/_explorer/index.html"
        try:
            r = requests.get(url, verify=False, timeout=5)
            print(f"GET {url} -> {r.status_code}")
        except Exception as e:
            print(f"Ping failed: {e}")
        sys.exit(0)

    # One-shot: count chunk docs
    if args.count:
        if not cosmos_https_up(): print("Waiting for emulator HTTPS ..."); time.sleep(2)
        wait_for_cosmos()
        print(count_chunks()); sys.exit(0)

    # One-shot: seed meta docs for legacy ingests missing meta
    if args.seed_meta:
        if not cosmos_https_up(): print("Waiting for emulator HTTPS ..."); time.sleep(2)
        wait_for_cosmos()
        seed_meta_for_existing()
        sys.exit(0)

    # Actions that require the emulator and the container to be ready:
    if args.ingest:
        if not cosmos_https_up(): print("Waiting for emulator HTTPS ..."); time.sleep(2)
        wait_for_cosmos()

        only_repo = None
        if args.first_repo:
            repo = pick_first_repo_with_targets()
            if repo is None:
                print(f"No target files found in {DOCS_DIR}")
                sys.exit(0)
            only_repo = repo

        ingest_dir(force=args.force, since_hours=(args.since_hours if hasattr(args, "since_hours") else None), only_repo=only_repo)
        sys.exit(0)

    elif args.search:
        if not cosmos_https_up(): print("Waiting for emulator HTTPS ..."); time.sleep(2)
        wait_for_cosmos()
        results = vector_search(args.search, top_k=TOP_K)
        for r in results:
            dist = r.get("distance")
            dtxt = f"{dist:.4f}" if isinstance(dist, (int, float)) else str(dist)
            print(f"[distance={dtxt}] {r['id']}\n{r.get('text','')[:300]}...\n")

    elif args.ask:
        if not cosmos_https_up(): print("Waiting for emulator HTTPS ..."); time.sleep(2)
        wait_for_cosmos()
        print(rag_answer(args.ask, top_k=TOP_K))

    else:
        # If no action chosen, show help
        parser.print_help()
