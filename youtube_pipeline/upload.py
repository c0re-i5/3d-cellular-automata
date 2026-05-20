"""Upload queued recordings to YouTube via the Data API v3.

Default behaviour:

  $ python -m youtube_pipeline                  # upload everything in queue, public
  $ python -m youtube_pipeline --privacy unlisted
  $ python -m youtube_pipeline --dry-run        # show what would happen
  $ python -m youtube_pipeline --file path.mp4  # upload one specific file
  $ python -m youtube_pipeline --watch          # daemon: poll queue every 30s

After a successful upload the source ``.mp4`` and its sidecar are
moved to ``recordings/uploaded/YYYY-MM-DD/`` and an entry is appended
to ``recordings/upload_log.jsonl``.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

from .auth import get_youtube_service
from .metadata import build_metadata

REPO_ROOT = Path(__file__).resolve().parent.parent
QUEUE_DIR = REPO_ROOT / 'recordings' / 'upload_queue'
UPLOADED_DIR = REPO_ROOT / 'recordings' / 'uploaded'
LOG_PATH = REPO_ROOT / 'recordings' / 'upload_log.jsonl'
# Persistent flag so a per-process invocation (e.g. `for f in ...;
# do python -m youtube_pipeline --file "$f"; done`) doesn't re-spam the
# API after the limit has already been hit. Delete the file by hand to
# resume; this script never clears it automatically.
LIMIT_MARKER = REPO_ROOT / 'recordings' / '.upload_limit'

# ── chunked resumable upload (handles bad networks gracefully) ─────────
CHUNK_SIZE = 8 * 1024 * 1024

# YouTube API error reasons that mean "stop trying, this won't get better by
# retrying the next file". Hitting any of these aborts the rest of the
# queue instead of spamming a doomed attempt for every file.
# See: https://developers.google.com/youtube/v3/docs/errors
_FATAL_QUOTA_REASONS = frozenset({
    'quotaExceeded',
    'uploadLimitExceeded',
    'dailyLimitExceeded',
    'rateLimitExceeded',
    'userRequestsExceedRateLimit',
    'youtubeSignupRequired',
    'channelClosed',
    'channelSuspended',
    'authenticatedUserAccountSuspended',
    'authenticatedUserAccountClosed',
})


class UploadLimitReached(Exception):
    """Raised when the API tells us further uploads cannot succeed today
    (quota, daily upload count, suspended channel, etc.). Aborts the
    remaining queue."""


def _classify_http_error(e: HttpError) -> tuple[str | None, str]:
    """Return (reason, message) parsed from a googleapiclient HttpError.
    `reason` is None when we can't extract one."""
    reason = None
    message = str(e)
    try:
        body = json.loads(e.content.decode('utf-8', 'replace'))
        err = body.get('error', {})
        message = err.get('message', message)
        errors = err.get('errors') or []
        if errors:
            reason = errors[0].get('reason')
    except Exception:  # noqa: BLE001  malformed JSON, treat as missing
        pass
    return reason, message


def _write_limit_marker(reason: str | None, message: str) -> None:
    """Persist the limit hit so subsequent processes bail fast. The user
    deletes the file by hand when they want uploads to resume."""
    payload = {
        'hit_at': datetime.now().isoformat(timespec='seconds'),
        'reason': reason,
        'message': message,
    }
    try:
        LIMIT_MARKER.parent.mkdir(parents=True, exist_ok=True)
        LIMIT_MARKER.write_text(json.dumps(payload, indent=2))
    except OSError as e:
        print(f'  (could not write limit marker: {e})', file=sys.stderr)


def _read_limit_marker() -> dict | None:
    """Return marker payload if it exists, else None. Never auto-clears."""
    if not LIMIT_MARKER.exists():
        return None
    try:
        return json.loads(LIMIT_MARKER.read_text())
    except (OSError, json.JSONDecodeError):
        # Corrupt file still counts as "limit hit" — fail safe.
        return {'hit_at': 'unknown', 'reason': 'unknown',
                'message': '(unreadable marker file)'}


def _find_jobs(queue_dir: Path) -> list[Path]:
    """Return .mp4 files that have a matching .json sidecar."""
    jobs = []
    for mp4 in sorted(queue_dir.glob('*.mp4')):
        sidecar = mp4.with_suffix('.json')
        if sidecar.exists():
            jobs.append(mp4)
        else:
            print(f'  skip {mp4.name}: no .json sidecar', file=sys.stderr)
    return jobs


def _log_upload(entry: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open('a') as f:
        f.write(json.dumps(entry) + '\n')


def _archive(mp4_path: Path) -> Path:
    """Move uploaded .mp4 + sidecars into recordings/uploaded/<date>/."""
    date_dir = UPLOADED_DIR / datetime.now().strftime('%Y-%m-%d')
    date_dir.mkdir(parents=True, exist_ok=True)
    moved = []
    for p in [mp4_path,
              mp4_path.with_suffix('.json'),
              mp4_path.with_name(mp4_path.stem + '_overrides.json')]:
        if p.exists():
            target = date_dir / p.name
            shutil.move(str(p), str(target))
            moved.append(target)
    return moved[0] if moved else mp4_path


def upload_one(youtube, mp4_path: Path, privacy: str,
               dry_run: bool = False) -> dict | None:
    """Upload a single recording. Returns the YouTube response dict or None."""
    sidecar = mp4_path.with_suffix('.json')
    meta = build_metadata(sidecar)
    body = {
        'snippet': {
            'title': meta['title'],
            'description': meta['description'],
            'tags': meta['tags'],
            'categoryId': meta.get('category_id', '28'),
        },
        'status': {
            'privacyStatus': privacy,
            'selfDeclaredMadeForKids': False,
        },
    }

    print(f'\n[{mp4_path.name}]')
    print(f'  shorts:      {meta["shorts"]}')
    print(f'  title:       {meta["title"]}')
    print(f'  privacy:     {privacy}')
    print(f'  tags:        {", ".join(meta["tags"])}')
    print(f'  description: {meta["description"][:120]}…')

    if dry_run:
        print('  (dry run — not uploading)')
        return None

    media = MediaFileUpload(str(mp4_path), mimetype='video/mp4',
                            chunksize=CHUNK_SIZE, resumable=True)
    request = youtube.videos().insert(
        part='snippet,status', body=body, media_body=media)

    response = None
    last_pct = -1
    while response is None:
        try:
            status, response = request.next_chunk()
        except HttpError as e:
            reason, message = _classify_http_error(e)
            # Stop the whole queue if the account/quota is the problem —
            # spamming the next file would just hit the same error.
            if reason in _FATAL_QUOTA_REASONS or e.resp.status == 401:
                _write_limit_marker(reason, message)
                raise UploadLimitReached(
                    f'{e.resp.status} {reason or "auth"}: {message}') from e
            # Retry transient 5xx errors a couple of times before giving up.
            if e.resp.status in (500, 502, 503, 504):
                print(f'  transient error {e.resp.status}, retrying…')
                time.sleep(2)
                continue
            raise
        if status:
            pct = int(status.progress() * 100)
            if pct != last_pct:
                print(f'  upload {pct}%')
                last_pct = pct

    video_id = response.get('id')
    url = f'https://youtu.be/{video_id}'
    print(f'  ✓ uploaded: {url}')

    _log_upload({
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'file': mp4_path.name,
        'video_id': video_id,
        'url': url,
        'privacy': privacy,
        'shorts': meta['shorts'],
        'title': meta['title'],
    })
    _archive(mp4_path)
    return response


def run_once(privacy: str, dry_run: bool, only: Path | None,
             force: bool = False, reddit: bool = False,
             reddit_max_per_day: int = 4) -> int:
    """Process the queue once. Returns process exit code."""
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    # Bail early if a previous run hit a quota/limit error. Marker
    # persists until the user removes it by hand; --force ignores it.
    if not dry_run and not force:
        marker = _read_limit_marker()
        if marker is not None:
            print(
                f'Upload limit marker present (hit at {marker.get("hit_at")}, '
                f'reason: {marker.get("reason") or "unknown"}). Skipping.\n'
                f'  delete  {LIMIT_MARKER}  (or pass --force) to resume.',
                file=sys.stderr,
            )
            return 2
    if only is not None:
        if not only.exists():
            print(f'No such file: {only}', file=sys.stderr)
            return 2
        jobs = [only]
    else:
        jobs = _find_jobs(QUEUE_DIR)

    if not jobs:
        print(f'Queue empty ({QUEUE_DIR}).')
        return 0

    youtube = None if dry_run else get_youtube_service()
    failures = 0
    aborted = False
    for mp4 in jobs:
        try:
            upload_one(youtube, mp4, privacy, dry_run=dry_run)
        except UploadLimitReached as e:
            remaining = len(jobs) - jobs.index(mp4) - 1
            print(f'  ✗ upload limit reached: {e}', file=sys.stderr)
            if remaining > 0:
                print(f'  aborting queue: {remaining} file(s) left untouched.',
                      file=sys.stderr)
            failures += 1
            aborted = True
            break
        except HttpError as e:
            # Non-fatal API error on one file: log and move on.
            reason, message = _classify_http_error(e)
            print(f'  ✗ failed ({e.resp.status} {reason or ""}): {message}',
                  file=sys.stderr)
            failures += 1
            continue
        except Exception as e:  # noqa: BLE001 — surface anything to user
            print(f'  ✗ failed: {e}', file=sys.stderr)
            failures += 1
            continue
        # Optional Reddit cross-post. Only fires on a successful upload
        # (continue statements above skip it on failure). Daily cap and
        # all error handling live inside reddit_pipeline so a Reddit
        # hiccup never breaks the YouTube loop.
        if reddit and not dry_run:
            try:
                from reddit_pipeline.submit import submit_for_file
                submit_for_file(mp4.name, max_per_day=reddit_max_per_day)
            except Exception as e:  # noqa: BLE001
                print(f'  (reddit cross-post skipped: {e})',
                      file=sys.stderr)
    # rc=2 lets --watch back off instead of polling every 30 s into the
    # same quota error.
    if aborted:
        return 2
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog='youtube_pipeline',
                                description='Upload queued CA recordings.')
    p.add_argument('--privacy', choices=('public', 'unlisted', 'private'),
                   default='public')
    p.add_argument('--dry-run', action='store_true',
                   help='Show what would be uploaded; do not contact YouTube.')
    p.add_argument('--file', type=Path, default=None,
                   help='Upload one specific file instead of scanning queue.')
    p.add_argument('--watch', action='store_true',
                   help='Daemon mode: poll queue every 30 s.')
    p.add_argument('--force', action='store_true',
                   help='Ignore the .upload_limit cooldown marker.')
    p.add_argument('--reddit', action='store_true',
                   help='After each successful upload, cross-post to '
                   'r/3DCellularAutomata via reddit_pipeline (subject '
                   'to its daily cap).')
    p.add_argument('--reddit-max-per-day', type=int, default=4,
                   help='Daily cap for Reddit cross-posts (default: 4). '
                   'Once hit, further uploads in this run skip Reddit '
                   'silently.')
    p.add_argument('--print-channel-description', action='store_true',
                   help='Print the YouTube channel About text to stdout '
                   'and exit. Paste into YouTube Studio → Customisation '
                   '→ Basic info → Description.')
    args = p.parse_args(argv)

    if args.print_channel_description:
        from .metadata import channel_description
        print(channel_description())
        return 0

    if args.watch:
        try:
            while True:
                rc = run_once(args.privacy, args.dry_run, args.file,
                              force=args.force, reddit=args.reddit,
                              reddit_max_per_day=args.reddit_max_per_day)
                if rc == 2:
                    # Limit marker is set — don't poll. User clears it
                    # manually when ready.
                    print('  (limit marker set — exiting watch mode; '
                          f'delete {LIMIT_MARKER} to resume)')
                    return 2
                time.sleep(30)
        except KeyboardInterrupt:
            print('\n(stopped)')
            return 0
    return run_once(args.privacy, args.dry_run, args.file, force=args.force,
                    reddit=args.reddit,
                    reddit_max_per_day=args.reddit_max_per_day)
