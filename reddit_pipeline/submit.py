"""Submit uploaded YouTube recordings to r/3DCellularAutomata.

Workflow:

  1. Read ``recordings/upload_log.jsonl`` (written by youtube_pipeline)
     to find recordings already on YouTube.
  2. Skip entries already in ``recordings/reddit_log.jsonl``.
  3. Skip Shorts unless ``--include-shorts`` is passed (they post 3-5×
     a day; we don't want to flood the sub).
  4. For each remaining entry, locate its sidecar JSON under
     ``recordings/uploaded/YYYY-MM-DD/`` and build a Reddit submission.
  5. Submit as a link post with the YouTube URL, plus a markdown
     selftext-style body via the post body field.

CLI:

  $ python -m reddit_pipeline                  # post all unposted long-form
  $ python -m reddit_pipeline --dry-run        # show what would be posted
  $ python -m reddit_pipeline --file <name.mp4>  # one specific recording
  $ python -m reddit_pipeline --include-shorts # also post Shorts
  $ python -m reddit_pipeline --limit 1        # post at most N (default: 1)
  $ python -m reddit_pipeline --watch          # daemon: poll every 6 h

Posting cadence is intentionally conservative — default ``--limit 1``
per invocation and a 6-hour watch interval. Reddit's anti-spam logic
penalises high-frequency self-promo, especially from new accounts.
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


def _build_jobs(only_file: str | None, include_shorts: bool,
                ) -> list[tuple[dict, Path]]:
    """Return (upload_log_entry, sidecar_path) pairs ready to post.

    Filters: not already posted, sidecar locatable, shorts policy.
    Order: oldest first (so the queue drains in upload order).
    """
    log = _read_upload_log()
    posted = _read_reddit_log()
    jobs: list[tuple[dict, Path]] = []
    for entry in log:
        vid = entry.get('video_id')
        if not vid:
            continue
        if vid in posted:
            continue
        if only_file is not None and entry.get('file') != only_file:
            continue
        if entry.get('shorts') and not include_shorts:
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
             include_shorts: bool, limit: int, dry_run: bool) -> int:
    """Process the queue once. Returns process exit code."""
    jobs = _build_jobs(only_file, include_shorts)
    if not jobs:
        print('Nothing new to post.')
        return 0
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


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog='reddit_pipeline',
        description='Post uploaded CA recordings to r/3DCellularAutomata.')
    p.add_argument('--subreddit', default=DEFAULT_SUBREDDIT,
                   help=f'Target subreddit (default: {DEFAULT_SUBREDDIT}).')
    p.add_argument('--file', default=None,
                   help='Post one specific recording by .mp4 filename.')
    p.add_argument('--include-shorts', action='store_true',
                   help='Also post Shorts (default skips them — they '
                   'flood the sub at 3-5/day).')
    p.add_argument('--limit', type=int, default=1,
                   help='Maximum posts per invocation (default: 1).')
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
                         include_shorts=args.include_shorts,
                         limit=args.limit,
                         dry_run=args.dry_run)
                time.sleep(WATCH_INTERVAL_SEC)
        except KeyboardInterrupt:
            print('\n(stopped)')
            return 0
    return run_once(subreddit_name=args.subreddit,
                    only_file=args.file,
                    include_shorts=args.include_shorts,
                    limit=args.limit,
                    dry_run=args.dry_run)
