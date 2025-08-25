#!/usr/bin/env python3
"""
Scrape a GitHub repository's README page for outbound links to other GitHub repos.
Optionally:
  - Follow links to Markdown files inside the same source repo and collect any
    additional repo links found there.
  - Shallow-clone all discovered repositories into a local directory.

Inputs:
  --repo            The source repo URL (any GitHub URL within that repo is OK; we canonicalize).
  --out-list        File to write the discovered repo URLs (one per line).
  --clone-dir       If provided, shallow-clone each discovered repo into this directory.
  --crawl-internal-md
                    If set, also fetch and scan Markdown links within the same repo (README → MD files),
                    to discover more outbound repos linked from those docs.

Notes:
  - We fetch the HTML-rendered README by simply hitting the repo root page on GitHub; GitHub renders README there.
  - We identify repo links with a strict regex and ensure path depth is exactly /owner/repo to avoid issues/blobs/tree pages.
"""

import argparse
import os
import re
import subprocess
import time
from urllib.parse import urlparse, urljoin

import requests
from bs4 import BeautifulSoup

# Strict pattern for GitHub repository URLs:
#   - scheme: http/https
#   - host: github.com
#   - path: /owner/repo (no trailing path elements used to determine canonical)
#   - optional trailing /, #, ?, etc. are tolerated
GITHUB_RE = re.compile(r"^https?://github\.com/([^/]+)/([^/#?]+)(?:[/#?].*)?$")


def is_repo_url(url: str) -> bool:
    """
    Return True if 'url' looks like a root GitHub repo URL (https://github.com/owner/repo),
    and not one of the common subpaths (issues, pulls, blob, tree, etc.).

    We check:
      - Host matches github.com
      - Path depth is exactly two segments: /owner/repo
    """
    m = GITHUB_RE.match(url)
    if not m:
        return False
    # Additional guard: require exactly two path segments
    parsed = urlparse(url)
    parts = [p for p in parsed.path.split("/") if p]
    return len(parts) == 2  # only /owner/repo qualifies


def canonical_repo(url: str) -> str:
    """
    Normalize any GitHub URL within a repo to its canonical root:
      https://github.com/<owner>/<repo>

    Returns empty string if URL does not match the expected GitHub pattern.
    """
    m = GITHUB_RE.match(url)
    if not m:
        return ""
    owner, repo = m.group(1), m.group(2)
    return f"https://github.com/{owner}/{repo}"


def fetch_html(url: str, retries: int = 3, sleep: float = 1.0) -> str:
    """
    Fetch a URL with basic retry/sleep. Returns response text or raises RuntimeError.

    Params:
      retries: number of attempts
      sleep:   seconds to wait between attempts
    """
    last = None
    headers = {"User-Agent": "repo-scraper/1.0"}
    for _ in range(retries):
        try:
            r = requests.get(url, headers=headers, timeout=15)
            r.raise_for_status()
            return r.text
        except Exception as e:
            last = e
            time.sleep(sleep)
    raise RuntimeError(f"GET failed for {url}: {last}")


def extract_links_from_html(html: str, base_url: str) -> list[str]:
    """
    Parse HTML and extract all anchor hrefs, resolving relative URLs against base_url.

    Returns a list of absolute URLs (may include non-GitHub and non-repo links).
    """
    soup = BeautifulSoup(html, "html.parser")
    links: list[str] = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        # Normalize relative links (e.g., ./docs/foo.md) to absolute URLs
        full = urljoin(base_url, href)
        links.append(full)
    return links


def is_internal_markdown(link: str, repo_root: str) -> bool:
    """
    Heuristic to decide whether 'link' points to a Markdown file inside the same repo.

    We allow:
      - Direct Markdown files ending with .md (rare when browsing HTML view, but cheap check)
      - GitHub blob URLs that end with .md, e.g.:
          https://github.com/owner/repo/blob/<branch>/path/to/file.md

    We require that 'link' starts with repo_root so we don't wander into other repos.
    """
    if not link.startswith(repo_root):
        return False
    # Quick checks for typical MD patterns on GitHub
    return link.endswith(".md") or ("/blob/" in link and link.endswith(".md"))


def github_readme_html_url(repo_url: str) -> str:
    """
    Derive the HTML URL for a repo's README:
      - For GitHub, the README is rendered on the repo root page itself.
      - So the canonical repo URL is sufficient.
    """
    return repo_url.rstrip("/")


def clone_repo(repo_url: str, dest_dir: str) -> None:
    """
    Shallow-clone a repository into dest_dir/owner__repo (flat namespace).

    Skips cloning if the target directory already exists and is non-empty.
    """
    os.makedirs(dest_dir, exist_ok=True)

    # Extract owner/repo to build a folder name like "owner__repo"
    m = GITHUB_RE.match(repo_url)
    assert m, f"Unexpected non-repo URL passed to clone_repo: {repo_url}"
    owner, repo = m.group(1), m.group(2)
    target = os.path.join(dest_dir, f"{owner}__{repo}")

    # Quick skip if directory exists and contains files
    if os.path.exists(target) and os.listdir(target):
        print(f"[skip] already exists: {target}")
        return

    # Shallow clone (depth=1) for speed and to minimize disk usage
    cmd = ["git", "clone", "--depth", "1", repo_url, target]
    print("[clone]", " ".join(cmd))
    subprocess.run(cmd, check=False)  # don't raise on failure; continue to next


def main():
    """
    CLI entrypoint. Example:

      python scrape_repos_from_readme.py \
        --repo https://github.com/satellite-image-deep-learning/techniques \
        --out-list referenced_repos.txt \
        --clone-dir ./clones \
        --crawl-internal-md
    """
    p = argparse.ArgumentParser(
        description="Collect and optionally clone repos referenced from a GitHub README."
    )
    p.add_argument(
        "--repo",
        required=True,
        help="Source GitHub repo, e.g. https://github.com/satellite-image-deep-learning/techniques",
    )
    p.add_argument(
        "--out-list",
        default="referenced_repos.txt",
        help="Where to save the list of repo URLs",
    )
    p.add_argument(
        "--clone-dir",
        default="",
        help="If set, shallow-clone all referenced repos into this directory",
    )
    p.add_argument(
        "--crawl-internal-md",
        action="store_true",
        help="Also follow README links to Markdown files inside the same repo",
    )
    args = p.parse_args()

    # Canonicalize the repo URL (any URL under the repo is normalized to /owner/repo)
    src_repo = canonical_repo(args.repo)
    if not src_repo:
        raise SystemExit("Provide a valid GitHub repo URL like https://github.com/owner/repo")

    print(f"[info] Scanning README for: {src_repo}")

    # 1) Fetch the repo root page (GitHub renders README on this page)
    readme_html_url = github_readme_html_url(src_repo)
    html = fetch_html(readme_html_url)

    # 2) Extract all hyperlinks from the page
    links = extract_links_from_html(html, base_url=readme_html_url)

    # Prepare containers:
    repo_links: set[str] = set()        # discovered repo roots
    md_links_to_crawl: set[str] = set() # internal markdown pages to optionally crawl

    # 3) First pass: collect repo links directly found on README; queue internal MD links
    for link in links:
        if is_repo_url(link):
            repo_links.add(canonical_repo(link))
        elif args.crawl_internal_md and is_internal_markdown(link, src_repo):
            md_links_to_crawl.add(link)

    # 4) Optional second pass: crawl internal Markdown pages to find more outbound repo links
    if args.crawl_internal_md and md_links_to_crawl:
        print(f"[info] Crawling {len(md_links_to_crawl)} internal markdown files...")
        for md_link in md_links_to_crawl:
            try:
                md_html = fetch_html(md_link)  # GitHub will render MD to HTML at blob URLs
                more = extract_links_from_html(md_html, base_url=md_link)
                for link in more:
                    if is_repo_url(link):
                        repo_links.add(canonical_repo(link))
            except Exception as e:
                # Non-fatal: keep going even if a page fails to load
                print(f"[warn] Failed to crawl {md_link}: {e}")

    # 5) Produce sorted list for determinism
    repo_list = sorted(repo_links)
    print(f"[done] Found {len(repo_list)} unique repos.")
    for r in repo_list:
        print(" -", r)

    # 6) Write the list to disk
    with open(args.out_list, "w", encoding="utf-8") as f:
        f.write("\n".join(repo_list))
    print(f"[write] {args.out_list}")

    # 7) Optionally clone each discovered repo (shallow)
    if args.clone_dir:
        for r in repo_list:
            clone_repo(r, args.clone_dir)


if __name__ == "__main__":
    main()
