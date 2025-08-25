#!/usr/bin/env python3
"""
Clone GitHub repos from a CSV produced elsewhere (e.g., search results),
skipping any repo that already exists with non-empty contents.

Expected CSV columns:
- html_url   : the HTTPS URL to clone, e.g., https://github.com/owner/repo
- full_name  : the GitHub "owner/repo" string (used to name folders locally)
"""

# =======================
# Imports & Config
# =======================
import os
import csv
import subprocess

CSV_FILE  = "repos_ranked.csv"     # Input CSV you already have
CLONE_DIR = "../data/repos"        # Parent directory where clones will live

# Create the target directory if it doesn't exist (safe if it does)
os.makedirs(CLONE_DIR, exist_ok=True)

# =======================
# Main: read CSV & clone
# =======================
with open(CSV_FILE, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)  # Reads rows into dicts keyed by column header

    for row in reader:
        # Pull the HTTPS repo URL and full_name ("owner/repo") from the CSV
        url = row["html_url"]
        # Replace "/" with "__" so we get a single folder like "owner__repo"
        # instead of creating nested directories owner/repo
        full_name = row["full_name"].replace("/", "__")

        # Compose the final clone path, e.g., ../data/repos/owner__repo
        target = os.path.join(CLONE_DIR, full_name)

        # Quick skip: if the target directory exists AND is non-empty,
        # assume it's already cloned and skip doing it again.
        if os.path.exists(target) and os.listdir(target):
            print(f"[skip] exists {target}")
            continue

        # Otherwise, clone shallowly (depth=1) to keep it fast and small
        print(f"[clone] {url} -> {target}")
        # Note: check=False so the loop continues even if a particular clone fails
        # You can set check=True to raise on first failure.
        subprocess.run(
            ["git", "clone", "--depth", "1", url, target],
            check=False
        )
