#!/usr/bin/env python
"""
Generate the development history document from the repository itself.

    python scripts/generate_history_document.py
    python scripts/generate_history_document.py --no-diffs      # summary only
    python scripts/generate_history_document.py --out FILE

WHY THIS IS GENERATED RATHER THAN WRITTEN
-----------------------------------------
A record of work is only worth keeping if it is accurate, and a hand-written
account of two months of engineering drifts from the truth the moment the code
moves. This reads the git history and produces the document from it, so every
statement in the output corresponds to a commit that exists, and re-running it
after further work produces a correct document rather than a stale one.

Nothing is invented, summarised or paraphrased: commit messages appear verbatim,
diffs are the real patches, and the statistics are counted rather than
estimated.

OUTPUT
------
Markdown, structured for conversion to PDF:

    pandoc delivery/A1-DEVELOPMENT-HISTORY.md -o A1-DEVELOPMENT-HISTORY.pdf \\
           --toc --toc-depth=2 -V geometry:margin=2cm

Every commit becomes a level-2 heading, so the generated table of contents is a
usable index into two months of work.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent

# Diffs against these paths are recorded as a stat line only. They are
# generated artefacts and lock files: including their full text would add tens
# of thousands of lines that document nothing about the engineering decisions.
NOISY = (".lock", "package-lock.json", "-lock.yaml", ".min.js", ".min.css")


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=_ROOT, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    if result.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed:\n{result.stderr}")
    return result.stdout


def commit_hashes() -> list[str]:
    """Oldest first, so the document reads forwards through the work."""
    return [h for h in git("rev-list", "--reverse", "HEAD").splitlines() if h]


def field(sha: str, fmt: str) -> str:
    return git("show", "-s", f"--format={fmt}", sha).strip()


def is_noisy(path: str) -> bool:
    return any(path.endswith(suffix) or suffix in path for suffix in NOISY)


def commit_diff(sha: str, max_lines: int) -> tuple[str, bool]:
    """Full patch, minus generated files. Returns (text, was_truncated)."""
    raw = git("show", sha, "--format=", "--no-color", "--unified=3")
    lines = raw.splitlines()

    kept: list[str] = []
    skipping = False
    for line in lines:
        if line.startswith("diff --git "):
            parts = line.split(" b/")
            skipping = len(parts) > 1 and is_noisy(parts[-1])
            if skipping:
                kept.append(f"{line}\n[generated file — patch omitted, see stat above]")
                continue
        if not skipping:
            kept.append(line)

    truncated = len(kept) > max_lines
    if truncated:
        kept = kept[:max_lines] + [
            f"... [patch truncated at {max_lines} lines; "
            f"full text: git show {sha[:12]}]"
        ]
    return "\n".join(kept), truncated


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(_ROOT / "delivery/A1-DEVELOPMENT-HISTORY.md"))
    ap.add_argument("--no-diffs", action="store_true", help="omit patches entirely")
    ap.add_argument("--max-diff-lines", type=int, default=4000,
                    help="cap per-commit patch length")
    args = ap.parse_args()

    hashes = commit_hashes()
    if not hashes:
        print("no commits found")
        return 1

    print(f"reading {len(hashes)} commits ...")
    parts: list[str] = []
    authors: Counter[str] = Counter()
    by_day: Counter[str] = Counter()
    files_touched: Counter[str] = Counter()
    total_ins = total_del = 0

    # ---- gather first, so the summary can be written before the body -----
    records = []
    for n, sha in enumerate(hashes, 1):
        subject = field(sha, "%s")
        body = field(sha, "%b")
        author = field(sha, "%an")
        date = field(sha, "%ad")
        iso = field(sha, "%ad")
        day = git("show", "-s", "--format=%ad", "--date=short", sha).strip()
        stat = git("show", sha, "--format=", "--stat=100")

        authors[author] += 1
        by_day[day] += 1
        for line in git("show", sha, "--format=", "--numstat").splitlines():
            cols = line.split("\t")
            if len(cols) == 3:
                ins, dele, path = cols
                files_touched[path] += 1
                if ins.isdigit():
                    total_ins += int(ins)
                if dele.isdigit():
                    total_del += int(dele)

        records.append((n, sha, subject, body, author, day, stat))
        if n % 10 == 0:
            print(f"  {n}/{len(hashes)}", end="\r", flush=True)

    first_day = records[0][5]
    last_day = records[-1][5]

    # ---- header and summary --------------------------------------------
    parts.append(f"""# Development History — NexGen iMATCH

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')} ·
**Commits:** {len(hashes)} · **Period:** {first_day} to {last_day}

This document is generated directly from the repository's version-control
history. Commit messages appear **verbatim**; patches are the real diffs;
statistics are counted from the history rather than estimated. Re-running
`scripts/generate_history_document.py` reproduces it against the current state
of the work.

It exists to record what was done, in what order, and why — including the
attempts that failed, which are retained rather than removed because a record
that keeps only the successes is not a record of the work.

---

## Summary

| | |
|---|---|
| Commits | {len(hashes)} |
| First commit | {first_day} |
| Most recent commit | {last_day} |
| Active days | {len(by_day)} |
| Lines added | {total_ins:,} |
| Lines removed | {total_del:,} |
| Distinct files touched | {len(files_touched):,} |

### Contributors

| Author | Commits |
|---|---|
""")
    for name, count in authors.most_common():
        parts.append(f"| {name} | {count} |\n")

    parts.append("\n### Activity by day\n\n| Date | Commits |\n|---|---|\n")
    for day in sorted(by_day):
        parts.append(f"| {day} | {by_day[day]} |\n")

    parts.append("\n### Most frequently changed files\n\n| File | Commits touching it |\n|---|---|\n")
    for path, count in files_touched.most_common(30):
        parts.append(f"| `{path}` | {count} |\n")

    parts.append("\n### Index of commits\n\n| # | Date | Commit | Subject |\n|---|---|---|---|\n")
    for n, sha, subject, _, _, day, _ in records:
        safe = subject.replace("|", "\\|")
        parts.append(f"| {n} | {day} | `{sha[:10]}` | {safe} |\n")

    parts.append("\n---\n\n# Commit record\n\n")

    # ---- body -----------------------------------------------------------
    truncations = 0
    for n, sha, subject, body, author, day, stat in records:
        parts.append(f"\n## {n}. {subject}\n\n")
        parts.append(f"**Commit:** `{sha}`  \n**Date:** {day}  \n**Author:** {author}\n\n")
        if body.strip():
            parts.append("### Rationale as recorded at the time\n\n```text\n")
            parts.append(body.strip())
            parts.append("\n```\n\n")
        if stat.strip():
            parts.append("### Files changed\n\n```text\n")
            parts.append(stat.strip())
            parts.append("\n```\n\n")
        if not args.no_diffs:
            diff, was_cut = commit_diff(sha, args.max_diff_lines)
            truncations += int(was_cut)
            if diff.strip():
                parts.append("### Changes\n\n```diff\n")
                parts.append(diff)
                parts.append("\n```\n\n")

    text = "".join(parts)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")

    words = len(text.split())
    print(f"\nwrote {out}")
    print(f"  {len(text.splitlines()):,} lines   {words:,} words")
    print(f"  ~{words / 450:.0f} pages at 450 w/p   ~{words / 250:.0f} pages at 250 w/p")
    if truncations:
        print(f"  {truncations} commit(s) had patches truncated at {args.max_diff_lines} lines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
