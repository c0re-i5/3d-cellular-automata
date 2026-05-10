# Reddit submission pipeline

Private tooling for posting uploaded YouTube recordings to
**r/3DCellularAutomata**. Not part of the public repository — lives
only on the `youtube-pipeline` branch alongside the YouTube uploader.

## What it does

1. Reads `recordings/upload_log.jsonl` (written by `youtube_pipeline`)
   to find recordings that are already on YouTube.
2. Skips entries already in `recordings/reddit_log.jsonl`.
3. Skips Shorts by default (they post 3-5/day; we don't want to flood
   the sub). Pass `--include-shorts` to override.
4. For each remaining entry, locates the sidecar JSON under
   `recordings/uploaded/YYYY-MM-DD/`, builds a Reddit submission, and
   posts it to r/3DCellularAutomata as a YouTube link with a markdown
   reproduction comment.

## One-time setup

### 1. Install dependencies

```bash
.venv/bin/pip install -r reddit_pipeline/requirements.txt
```

### 2. Create a Reddit script app

1. Go to <https://www.reddit.com/prefs/apps> while logged in as the
   account that should post (the channel owner).
2. Scroll down → **create another app...**
3. Fill in:
   - **name**: anything, e.g. `ca-reddit-bot`
   - **type**: **script**
   - **description / about url**: optional
   - **redirect uri**: `http://localhost:8080`
     (script apps don't actually use this, but the field is required)
4. Click **create app**.
5. The client id is the 14-character string under the app name; the
   secret is the longer `secret` field.

### 3. Write the credentials JSON

Save this as `reddit_pipeline/credentials/reddit_secrets.json`:

```json
{
    "client_id":     "<14-char id under the app name>",
    "client_secret": "<secret string>",
    "username":      "<your reddit username>",
    "password":      "<your reddit password>",
    "user_agent":    "ca-reddit-bot/0.1 by <your username>"
}
```

The `credentials/` directory is gitignored.

> ⚠️ Reddit requires the password of the account hosting the script
> app. If your account has 2FA enabled, append the current 6-digit
> code to the password (`yourpassword:123456`) — but the token only
> lasts a few minutes, so 2FA is impractical for a daemon. Either
> disable 2FA on the bot account or use a dedicated bot account.

### 4. Smoke test

```bash
.venv/bin/python -m reddit_pipeline --dry-run --limit 5
```

This previews up to 5 unposted recordings without contacting Reddit.
Inspect the titles, bodies, and flair selections.

## Usage

```bash
# Post the next unposted long-form recording (default: 1 per invocation).
.venv/bin/python -m reddit_pipeline

# Preview what would be posted without contacting Reddit.
.venv/bin/python -m reddit_pipeline --dry-run

# Post one specific recording by filename.
.venv/bin/python -m reddit_pipeline --file 20260501_120000_Crystal_Snowflake_2560x1440.mp4

# Also post Shorts (off by default — would otherwise spam the sub).
.venv/bin/python -m reddit_pipeline --include-shorts

# Post up to 3 in one invocation.
.venv/bin/python -m reddit_pipeline --limit 3

# Daemon mode: poll every 6 hours.
.venv/bin/python -m reddit_pipeline --watch
```

## Posting cadence

The defaults are intentionally conservative:

- **`--limit 1`** per invocation
- **6-hour** poll interval in `--watch` mode
- **Shorts off** by default

Reddit's spam filter penalises high-frequency self-promo, especially
from new accounts. A single thoughtful post per day from the bot,
combined with you participating in the sub manually, will perform
much better than auto-flooding.

## Per-recording overrides

To customise a specific post without editing code, drop a
`<sidecar>_reddit_overrides.json` next to the sidecar:

```json
{
    "title": "Custom title that overrides the auto-generated one",
    "flair": "Discovery"
}
```

Any fields in the override file replace the auto-generated values.

## Files written

- `recordings/reddit_log.jsonl` — append-only log of posted submissions,
  one JSON object per line. Used to dedupe.
- `reddit_pipeline/credentials/` — your secrets (gitignored).
