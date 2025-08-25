#!/usr/bin/env python3
"""
Rank and optionally clone GitHub repositories referenced by URLs.

Pipeline:
1) Read repo URLs/slugs from a text file (one per line).
2) For each repo (in parallel, but modestly), fetch lightweight metadata.
3) (Optionally) Probe the repo tree once to see if it contains any .ipynb files.
4) Score each repo by recency, popularity, language preference, and notebook presence.
5) Write a ranked CSV, and optionally shallow-clone the top N repos.

Notes:
- Auth: set GITHUB_TOKEN in your environment to raise rate limits and reduce 403s.
- Rate limiting:
  * Soft global RPS limit via a sliding window across all threads.
  * Server-driven 403 rate limit handling with X-RateLimit-Reset.
  * Exponential backoff for transient statuses.

CSV columns produced:
  score, full_name, html_url, description, stars, forks, language, license,
  archived, fork, has_issues, pushed_at, topics, has_ipynb, readme_hit

'topics' and 'readme_hit' are included for compatibility/extension: this script
does not fetch topics or scan readme, so they are written as empty strings.
"""

# =======================
# Imports & Globals
# =======================
import os
import csv
import time
import math
import argparse
import requests
import threading
from datetime import datetime
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque

# GitHub REST API base
API = "https://api.github.com"

# If provided, used for authentication to raise rate limits and reduce 403s
TOKEN = os.getenv("GITHUB_TOKEN")

# =======================
# Tuning knobs
# =======================
MAX_WORKERS = 8            # Mild parallelism; keep small to stay friendly to the API
GLOBAL_RPS_LIMIT = 5       # Soft cap on requests per second across all threads
REQUEST_TIMEOUT = 20       # Per-request timeout (seconds)
RETRY_STATUSES = {429, 502, 503, 504}  # Transient/retryable status codes
MAX_BACKOFF = 60           # Max exponential backoff (seconds)

# =======================
# Shared session & rate limiter state
# =======================
SESSION = requests.Session()
SESSION.headers.update({"Accept": "application/vnd.github+json"})
if TOKEN:
    # Bearer token if provided; otherwise unauthenticated (lower limits)
    SESSION.headers["Authorization"] = f"Bearer {TOKEN}"

_rate_lock = threading.Lock()  # protects the deque below
_recent_calls = deque()        # timestamps of recent requests across all threads


# =======================
# Rate limiting
# =======================
def _respect_rps_limit():
    """
    Enforce a soft, global sliding-window rate limit across all threads.

    We allow up to GLOBAL_RPS_LIMIT requests within any 1-second window.
    If we hit the cap, sleep just enough for the earliest call to drop out
    of the 1s window.
    """
    with _rate_lock:
        now = time.time()
        # Drop timestamps older than a 1-second window
        while _recent_calls and (now - _recent_calls[0]) > 1.0:
            _recent_calls.popleft()
        # If at capacity, sleep until the window has room
        if len(_recent_calls) >= GLOBAL_RPS_LIMIT:
            sleep_for = 1.0 - (now - _recent_calls[0])
            if sleep_for > 0:
                time.sleep(sleep_for)
        # Record this call
        _recent_calls.append(time.time())


# =======================
# GitHub request helper
# =======================
def gh_get(path, params=None):
    """
    GET a GitHub API path with:
      - global soft RPS rate limiting
      - exponential backoff for transient errors
      - hard rate-limit handling via 'X-RateLimit-Reset'

    Returns JSON-decoded response on success or raises HTTPError on non-retryable failures.
    """
    url = f"{API}{path}"
    backoff = 2  # initial backoff for transient statuses (seconds)

    while True:
        _respect_rps_limit()
        r = SESSION.get(url, params=params, timeout=REQUEST_TIMEOUT)

        # Hard rate limit: wait until reset time indicated by GitHub headers.
        # This typically appears as 403 with a message referencing rate limit.
        if r.status_code == 403 and "rate limit" in r.text.lower():
            reset = r.headers.get("X-RateLimit-Reset")
            if reset:
                wait = max(0, int(reset) - int(time.time())) + 1
                print(f"[rate-limit] sleeping {wait}s…")
                time.sleep(wait)
                continue  # retry after sleeping

        # Transient errors: use exponential backoff and retry
        if r.status_code in RETRY_STATUSES:
            print(f"[retry] {r.status_code} on {path}; sleeping {backoff}s")
            time.sleep(backoff)
            backoff = min(MAX_BACKOFF, backoff * 2)
            continue

        # Raise for other non-2xx statuses (will propagate to caller)
        r.raise_for_status()
        return r.json()


# =======================
# Repo utilities
# =======================
def repo_slug(url: str):
    """
    Convert a URL like 'https://github.com/owner/repo' to 'owner/repo'.
    Returns None if the path does not look like a repo URL.
    """
    p = urlparse(url)
    parts = [x for x in p.path.split("/") if x]
    if len(parts) < 2:
        return None
    return f"{parts[0]}/{parts[1]}"


def get_repo_info(slug: str):
    """
    Fetch lightweight repository metadata (no tree walk).

    Returns a dict with:
      - full_name, html_url, description
      - stars, forks, language
      - archived, fork, has_issues
      - license (SPDX id or "")
      - pushed_at (ISO string) and default_branch
    """
    data = gh_get(f"/repos/{slug}")
    return {
        "full_name": data["full_name"],
        "html_url": data["html_url"],
        "description": (data.get("description") or "")[:300],  # short, safe description
        "stars": data.get("stargazers_count", 0),
        "forks": data.get("forks_count", 0),
        "language": data.get("language") or "",
        "archived": data.get("archived", False),
        "fork": data.get("fork", False),
        "has_issues": data.get("has_issues", True),
        "license": (data.get("license") or {}).get("spdx_id") or "",
        "pushed_at": data.get("pushed_at") or "",
        "default_branch": data.get("default_branch") or "main",
    }


def has_ipynb_via_tree(slug: str, branch: str) -> bool:
    """
    Probe for any .ipynb file using a single recursive tree call:

      1) Resolve branch ref -> commit SHA via:
           GET /repos/{slug}/git/refs/heads/{branch}
         Fallback to:
           GET /repos/{slug}/branches/{branch}

      2) GET /repos/{slug}/git/trees/{sha}?recursive=1
         Iterate the entries for any path ending with '.ipynb'.

    This avoids a full search or multiple directory walks.
    Returns True if any notebook is found, else False.
    """
    sha = None

    # Try the refs endpoint first (works for active branches)
    try:
        ref = gh_get(f"/repos/{slug}/git/refs/heads/{branch}")
        sha = ref.get("object", {}).get("sha")
    except Exception:
        # swallow and try the branches endpoint
        pass

    # Fallback: /branches endpoint
    if not sha:
        try:
            br = gh_get(f"/repos/{slug}/branches/{branch}")
            sha = br.get("commit", {}).get("sha")
        except Exception:
            # If we can't resolve SHA, bail out with False
            return False

    if not sha:
        return False

    # Recursive tree: gets a flat list of all files (paths)
    tree = gh_get(f"/repos/{slug}/git/trees/{sha}", params={"recursive": "1"})
    for entry in tree.get("tree", []):
        p = entry.get("path", "")
        if p.endswith(".ipynb"):
            return True
    return False


# =======================
# Scoring
# =======================
def score_repo(r, now=None, prefer_python=True):
    """
    Compute a composite score for a repo dict.

    Components:
      - Recency: linearly decays from 1.0 (current) to 0.0 at 24 months
      - Popularity: log10(stars + 1)
      - Language bonus: +0.2 if primary language is Python (configurable)
      - Notebook bonus: +0.2 if repo contains any .ipynb
      - Penalties: archived (-0.8), fork (-0.4), no license (-0.1), issues disabled (-0.1)

    Returns a rounded float for stable sorting.
    """
    now = now or datetime.utcnow()

    # Recency (based on pushed_at). If missing/bad, recency=0.
    try:
        pushed = datetime.strptime(r["pushed_at"].replace("Z", ""), "%Y-%m-%dT%H:%M:%S")
        months = (now - pushed).days / 30.44
        recency = max(0.0, 1.0 - min(months, 24) / 24)
    except Exception:
        recency = 0.0

    # Popularity uses a log transform to reduce skew
    pop = math.log10(r.get("stars", 0) + 1)

    # Bonuses
    lang_bonus = 0.2 if (prefer_python and (r.get("language") or "").lower() == "python") else 0.0
    nb_bonus   = 0.2 if r.get("has_ipynb") else 0.0

    # Penalties
    penalty = 0.0
    if r.get("archived"):       penalty += 0.8
    if r.get("fork"):           penalty += 0.4
    if not r.get("license"):    penalty += 0.1
    if not r.get("has_issues"): penalty += 0.1

    score = (0.5 * recency) + (0.45 * pop) + lang_bonus + nb_bonus - penalty
    return round(score, 3)


# =======================
# Per-repo processing
# =======================
def process_repo(slug: str, now: datetime, args):
    """
    Full per-repo pipeline:
      - Fetch metadata.
      - Apply quick filters (archived, forks, min stars, age, language).
      - If still eligible, probe once for .ipynb via the recursive tree.
      - Score and return a metadata dict with 'score' and 'has_ipynb'.

    Returns:
      dict on success, or None if filtered out or on error.
    """
    try:
        info = get_repo_info(slug)

        # Quick filters first to avoid extra API calls
        if args.exclude_archived and info["archived"]:
            return None
        if args.exclude_forks and info["fork"]:
            return None
        if info["stars"] < args.min_stars:
            return None
        try:
            pushed = datetime.strptime(info["pushed_at"].replace("Z", ""), "%Y-%m-%dT%H:%M:%S")
            age_months = (now - pushed).days / 30.44
            if age_months > args.max_months_old:
                return None
        except Exception:
            # Ignore parse errors and keep the repo (might get low recency score)
            pass
        if args.only_python and (info["language"] or "").lower() != "python":
            return None

        # Single-call notebook probe only for survivors
        try:
            has_nb = has_ipynb_via_tree(slug, info["default_branch"])
        except Exception:
            has_nb = False  # treat probe failure as "no notebooks" to be conservative

        meta = dict(info)
        meta["has_ipynb"] = has_nb
        meta["score"] = score_repo(meta, now=now)
        return meta

    except Exception as e:
        print(f"[warn] {slug}: {e}")
        return None


# =======================
# Main
# =======================
def main():
    """
    CLI entrypoint. See --help for options.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", default="referenced_repos.txt",
                    help="Input file containing repo URLs (one per line)")
    ap.add_argument("--out", dest="outfile", default="repos_ranked.csv",
                    help="Output CSV file path")
    ap.add_argument("--min-stars", type=int, default=10,
                    help="Minimum stargazers required to consider a repo")
    ap.add_argument("--max-months-old", type=int, default=36,
                    help="Maximum age (months since last push) to include")
    ap.add_argument("--only-python", action="store_true",
                    help="Keep only repos whose primary language is Python")
    ap.add_argument("--exclude-forks", action="store_true",
                    help="Exclude repos that are forks")
    ap.add_argument("--exclude-archived", action="store_true",
                    help="Exclude repos that are archived")
    ap.add_argument("--top-n", type=int, default=150,
                    help="Write only the top N repos by score")
    ap.add_argument("--clone-dir", default="",
                    help="Optional directory to shallow-clone the top repos")
    args = ap.parse_args()

    # Read and normalize slugs from URLs in the input file
    with open(args.infile, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip()]
    slugs = []
    for u in urls:
        s = repo_slug(u)
        if s:
            slugs.append(s)
    slugs = sorted(set(slugs))  # de-duplicate, keep deterministic order
    print(f"[info] evaluating {len(slugs)} repos…")

    rows = []
    now = datetime.utcnow()

    # Parallel processing with modest concurrency (be nice to the API)
    processed = 0
    print_every = max(1, len(slugs) // 20)  # ~5% progress ticks

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(process_repo, slug, now, args): slug for slug in slugs}
        for i, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            if res:
                rows.append(res)
            processed += 1
            if processed % print_every == 0 or processed == len(slugs):
                print(f"  …{processed}/{len(slugs)}")

    # Rank by score descending and keep top N
    rows.sort(key=lambda r: r["score"], reverse=True)
    top = rows[:args.top_n]

    # Prepare CSV schema (includes optional fields for future extensions)
    fields = [
        "score", "full_name", "html_url", "description", "stars", "forks",
        "language", "license", "archived", "fork", "has_issues", "pushed_at",
        "topics", "has_ipynb", "readme_hit"
    ]
    # Write the output CSV
    with open(args.outfile, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in top:
            r = dict(r)
            # These two fields aren't populated by this script; keep empty for compatibility.
            r["topics"] = ",".join(r.get("topics", [])) if isinstance(r.get("topics"), list) else ""
            r["readme_hit"] = r.get("readme_hit", "")
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"[done] wrote {args.outfile} ({len(top)} rows)")

    # Optional shallow clone of the top repos
    if args.clone_dir:
        import subprocess
        os.makedirs(args.clone_dir, exist_ok=True)
        for r in top:
            url = r["html_url"]
            owner_repo = r["full_name"].replace("/", "__")  # single flat folder instead of nested owner/repo
            target = os.path.join(args.clone_dir, owner_repo)
            # Skip if already present and non-empty
            if os.path.exists(target) and os.listdir(target):
                print(f"[skip] exists {target}")
                continue
            print(f"[clone] {url} -> {target}")
            try:
                # Shallow clone for speed + space savings
                subprocess.run(["git", "clone", "--depth", "1", url, target], check=False)
            except Exception as e:
                print(f"[warn] clone failed for {url}: {e}")


# =======================
# Entrypoint
# =======================
if __name__ == "__main__":
    main()
