"""Submit uploaded YouTube recordings to r/3DCellularAutomata.

Workflow:

  1. Read ``recordings/upload_log.jsonl`` (written by youtube_pipeline)
     to find recordings already on YouTube.
  2. Skip entries already in ``recordings/reddit_log.jsonl``.
  3. Skip Shorts only if ``--no-shorts`` is passed (Shorts are the
     primary content stream, so by default they DO get posted).
  4. Enforce ``--max-per-day`` (default 4) by counting today's entries
     in ``recordings/reddit_log.jsonl``. Once hit, this run is a no-op.
  5. For each remaining entry, locate its sidecar JSON under
     ``recordings/uploaded/YYYY-MM-DD/`` and build a Reddit submission.
  6. Submit as a link post with the YouTube URL, plus the markdown
     reproduction info as our own top-level comment.

CLI:

  $ python -m reddit_pipeline                  # post newest unposted (any type)
  $ python -m reddit_pipeline --dry-run        # show what would be posted
  $ python -m reddit_pipeline --file <name.mp4>  # one specific recording
  $ python -m reddit_pipeline --no-shorts      # skip Shorts (long-form only)
  $ python -m reddit_pipeline --limit 1        # post at most N (default: 1)
  $ python -m reddit_pipeline --max-per-day 4  # daily cap (default: 4)
  $ python -m reddit_pipeline --watch          # daemon: poll every 6 h

Daily cap is the main spam control. Reddit's anti-spam logic punishes
high-frequency self-promo (especially from new accounts), so we cap
at 4 posts per calendar day even if the YouTube channel uploads more.
Jobs are processed newest-first so a fresh upload posts within hours
instead of waiting for an old backlog to drain.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RECORDINGS_DIR = REPO_ROOT / 'recordings'
UPLOADED_DIR = RECORDINGS_DIR / 'uploaded'
UPLOAD_LOG = RECORDINGS_DIR / 'upload_log.jsonl'
REDDIT_LOG = RECORDINGS_DIR / 'reddit_log.jsonl'

DEFAULT_SUBREDDIT = '3DCellularAutomata'
WATCH_INTERVAL_SEC = 6 * 60 * 60   # 6 hours


def _read_upload_log() -> list[dict]:
    """Return all entries from upload_log.jsonl, oldest first."""
    if not UPLOAD_LOG.exists():
        return []
    entries = []
    for line in UPLOAD_LOG.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f'  (skipping malformed upload_log line: {e})',
                  file=sys.stderr)
    return entries


def _read_reddit_log() -> set[str]:
    """Return the set of YouTube video IDs already posted to Reddit."""
    if not REDDIT_LOG.exists():
        return set()
    posted = set()
    for line in REDDIT_LOG.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
            vid = entry.get('video_id')
            if vid:
                posted.add(vid)
        except json.JSONDecodeError:
            pass
    return posted


def _append_reddit_log(entry: dict) -> None:
    REDDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with REDDIT_LOG.open('a') as f:
        f.write(json.dumps(entry) + '\n')


def _find_sidecar(mp4_filename: str) -> Path | None:
    """Locate the sidecar JSON for an uploaded recording.

    The youtube uploader moves files into uploaded/YYYY-MM-DD/, so we
    need to scan those date dirs. Filename match is exact (modulo the
    .mp4 → .json swap).
    """
    json_name = mp4_filename.removesuffix('.mp4') + '.json'
    if not UPLOADED_DIR.exists():
        return None
    # Most recent dates first — uploads are usually for fresh videos.
    for date_dir in sorted(UPLOADED_DIR.iterdir(), reverse=True):
        candidate = date_dir / json_name
        if candidate.exists():
            return candidate
    return None


def _count_today_posts() -> int:
    """How many entries in reddit_log.jsonl have today's local date."""
    if not REDDIT_LOG.exists():
        return 0
    today = datetime.now().strftime('%Y-%m-%d')
    n = 0
    for line in REDDIT_LOG.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ts = json.loads(line).get('timestamp', '')
        except json.JSONDecodeError:
            continue
        if ts.startswith(today):
            n += 1
    return n


def _build_jobs(only_file: str | None, skip_shorts: bool,
                ) -> list[tuple[dict, Path]]:
    """Return (upload_log_entry, sidecar_path) pairs ready to post.

    Filters: not already posted, sidecar locatable, shorts policy.
    Order: newest first — we want fresh uploads to post promptly
    rather than draining an old backlog into the sub. Old recordings
    that aged out of the daily cap simply stay un-posted.
    """
    log = _read_upload_log()
    posted = _read_reddit_log()
    jobs: list[tuple[dict, Path]] = []
    for entry in reversed(log):
        vid = entry.get('video_id')
        if not vid:
            continue
        if vid in posted:
            continue
        if only_file is not None and entry.get('file') != only_file:
            continue
        if entry.get('shorts') and skip_shorts:
            continue
        sidecar = _find_sidecar(entry['file'])
        if sidecar is None:
            print(f'  (skipping {entry["file"]}: sidecar not found)',
                  file=sys.stderr)
            continue
        jobs.append((entry, sidecar))
    return jobs


def _format_dry_run(submission: dict, url: str) -> str:
    return (
        '─── DRY RUN ───\n'
        f'  subreddit  : r/{DEFAULT_SUBREDDIT}\n'
        f'  link       : {url}\n'
        f'  flair      : {submission["flair"] or "(none)"}\n'
        f'  title      : {submission["title"]}\n'
        f'  body       :\n'
        + '\n'.join('    ' + ln for ln in submission['body'].splitlines())
    )


def _resolve_flair_id(subreddit, flair_text: str | None) -> str | None:
    """Look up the flair template id for the given flair text.

    Returns None if no matching flair exists (or subreddit has no
    flair templates configured) — caller should submit without flair
    in that case rather than crash.
    """
    if not flair_text:
        return None
    try:
        templates = list(subreddit.flair.link_templates.user_selectable())
    except Exception as e:  # noqa: BLE001 — flair API can 403 etc.
        print(f'  (could not fetch flair templates: {e})',
              file=sys.stderr)
        return None
    for tpl in templates:
        # PRAW returns dicts here with 'flair_text' and 'flair_template_id'.
        if tpl.get('flair_text', '').strip().lower() == flair_text.lower():
            return tpl.get('flair_template_id')
    return None


def submit_one(reddit, entry: dict, sidecar: Path, *,
               subreddit_name: str, dry_run: bool) -> dict | None:
    """Submit one recording. Returns the reddit log entry on success."""
    from .metadata import build_submission

    url = entry['url']
    submission = build_submission(sidecar, url)

    print(f'\n→ {entry["file"]}')
    print(f'  category : {submission["category"]}')
    print(f'  title    : {submission["title"]}')

    if dry_run:
        print(_format_dry_run(submission, url))
        return None

    sub = reddit.subreddit(subreddit_name)
    flair_id = _resolve_flair_id(sub, submission['flair'])

    # Reddit link submissions can't carry a markdown body in the same
    # post. Two-step: submit the link, then post the body as our own
    # top comment so the post body stays clickable to the video.
    post = sub.submit(
        title=submission['title'],
        url=url,
        flair_id=flair_id,
        send_replies=True,
    )
    print(f'  ✓ posted: {post.shortlink}')

    try:
        comment = post.reply(submission['body'])
        comment_id = comment.id if comment else None
    except Exception as e:  # noqa: BLE001
        print(f'  (warning: failed to post info comment: {e})',
              file=sys.stderr)
        comment_id = None

    log_entry = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'video_id': entry['video_id'],
        'video_url': url,
        'file': entry['file'],
        'subreddit': subreddit_name,
        'submission_id': post.id,
        'shortlink': post.shortlink,
        'comment_id': comment_id,
        'title': submission['title'],
        'flair': submission['flair'],
        'shorts': submission['shorts'],
    }
    _append_reddit_log(log_entry)
    return log_entry


def run_once(*, subreddit_name: str, only_file: str | None,
             skip_shorts: bool, limit: int, max_per_day: int,
             dry_run: bool) -> int:
    """Process the queue once. Returns process exit code."""
    jobs = _build_jobs(only_file, skip_shorts)
    if not jobs:
        print('Nothing new to post.')
        return 0

    # Apply daily cap before --limit so the cap is the hard ceiling.
    if max_per_day and max_per_day > 0 and not dry_run:
        already = _count_today_posts()
        budget = max_per_day - already
        if budget <= 0:
            print(f'Daily cap reached ({already}/{max_per_day} posted '
                  f'today). Skipping.')
            return 0
        jobs = jobs[:budget]

    if limit and limit > 0:
        jobs = jobs[:limit]
    print(f'Posting {len(jobs)} recording(s) to r/{subreddit_name} '
          f'{"(DRY RUN)" if dry_run else ""}')
    reddit = None
    if not dry_run:
        from .auth import get_reddit
        reddit = get_reddit()
        print(f'  authenticated as u/{reddit.user.me().name}')

    failures = 0
    for entry, sidecar in jobs:
        try:
            submit_one(reddit, entry, sidecar,
                       subreddit_name=subreddit_name, dry_run=dry_run)
        except Exception as e:  # noqa: BLE001
            print(f'  ✗ failed: {e}', file=sys.stderr)
            failures += 1
    return 1 if failures else 0


def submit_for_file(mp4_filename: str, *,
                    subreddit_name: str = DEFAULT_SUBREDDIT,
                    max_per_day: int = 4,
                    dry_run: bool = False) -> int:
    """Public entry point for chaining from youtube_pipeline.

    Posts a single recording (by .mp4 filename) to Reddit, subject to
    the daily cap. Designed to be called immediately after a successful
    YouTube upload — silently no-ops if the cap is hit, the entry is
    already posted, or the upload_log row hasn't been flushed yet.
    Never raises: failures are printed and swallowed so they don't
    abort the calling YouTube upload loop.
    """
    try:
        return run_once(subreddit_name=subreddit_name,
                        only_file=mp4_filename,
                        skip_shorts=False,
                        limit=1,
                        max_per_day=max_per_day,
                        dry_run=dry_run)
    except Exception as e:  # noqa: BLE001
        print(f'  (reddit cross-post failed: {e})', file=sys.stderr)
        return 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog='reddit_pipeline',
        description='Post uploaded CA recordings to r/3DCellularAutomata.')
    p.add_argument('--subreddit', default=DEFAULT_SUBREDDIT,
                   help=f'Target subreddit (default: {DEFAULT_SUBREDDIT}).')
    p.add_argument('--file', default=None,
                   help='Post one specific recording by .mp4 filename.')
    p.add_argument('--no-shorts', action='store_true',
                   help='Skip Shorts (long-form only). Default: post '
                   'Shorts too, since they are the primary content.')
    p.add_argument('--limit', type=int, default=1,
                   help='Maximum posts per invocation (default: 1).')
    p.add_argument('--max-per-day', type=int, default=4,
                   help='Hard ceiling on posts per calendar day '
                   '(default: 4). Counts entries in reddit_log.jsonl.')
    p.add_argument('--dry-run', action='store_true',
                   help='Show what would be posted without contacting Reddit.')
    p.add_argument('--watch', action='store_true',
                   help=f'Daemon mode: poll every '
                   f'{WATCH_INTERVAL_SEC // 3600}h.')
    args = p.parse_args(argv)

    if args.watch:
        try:
            while True:
                run_once(subreddit_name=args.subreddit,
                         only_file=args.file,
                         skip_shorts=args.no_shorts,
                         limit=args.limit,
                         max_per_day=args.max_per_day,
                         dry_run=args.dry_run)
                time.sleep(WATCH_INTERVAL_SEC)
        except KeyboardInterrupt:
            print('\n(stopped)')
            return 0
    return run_once(subreddit_name=args.subreddit,
                    only_file=args.file,
                    skip_shorts=args.no_shorts,
                    limit=args.limit,
                    max_per_day=args.max_per_day,
                    dry_run=args.dry_run)
