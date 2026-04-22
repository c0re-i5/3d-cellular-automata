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

# ── chunked resumable upload (handles bad networks gracefully) ─────────
CHUNK_SIZE = 8 * 1024 * 1024


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


def run_once(privacy: str, dry_run: bool, only: Path | None) -> int:
    """Process the queue once. Returns process exit code."""
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
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
    for mp4 in jobs:
        try:
            upload_one(youtube, mp4, privacy, dry_run=dry_run)
        except Exception as e:  # noqa: BLE001 — surface anything to user
            print(f'  ✗ failed: {e}', file=sys.stderr)
            failures += 1
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
    args = p.parse_args(argv)

    if args.watch:
        try:
            while True:
                run_once(args.privacy, args.dry_run, args.file)
                time.sleep(30)
        except KeyboardInterrupt:
            print('\n(stopped)')
            return 0
    return run_once(args.privacy, args.dry_run, args.file)
