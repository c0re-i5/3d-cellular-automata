# YouTube upload pipeline

Private tooling for uploading CA recordings to YouTube. **Not part of the public repository.**

## One-time setup

### 1. Install Python dependencies

```bash
.venv/bin/pip install -r youtube_pipeline/requirements.txt
```

### 2. Create OAuth 2.0 credentials in Google Cloud Console

1. Go to <https://console.cloud.google.com/>.
2. Create a new project (or pick an existing one).
3. **APIs & Services → Library** → enable **YouTube Data API v3**.
4. **APIs & Services → OAuth consent screen**:
   - User type: **External**
   - App name: anything (e.g. `ca-uploader`)
   - User support email + developer email: your address
   - Scopes: add `https://www.googleapis.com/auth/youtube.upload`
   - Test users: add your own Google account (the channel owner)
5. **APIs & Services → Credentials → Create Credentials → OAuth client ID**:
   - Application type: **Desktop app**
   - Name: anything
   - Click **Download JSON**
6. Save that file as:

   ```
   youtube_pipeline/credentials/client_secret.json
   ```

The `credentials/` directory is gitignored.

### 3. First run — grant the app permission to your channel

```bash
.venv/bin/python -m youtube_pipeline --dry-run
```

Wait — `--dry-run` skips auth too. To trigger the OAuth flow, run a real upload:

```bash
.venv/bin/python -m youtube_pipeline
```

A browser will open and ask you to grant the app upload permission to your YouTube channel. After consent, `youtube_pipeline/credentials/token.json` is written and reused on subsequent runs.

> **App is in "Testing" mode** in Google Cloud Console — this means the OAuth refresh token expires after 7 days and you'll need to re-grant consent. To get a permanent token, push the OAuth consent screen to **In Production** (no review needed for upload-only scope on your own channel).

## Usage

```bash
# Upload everything in recordings/upload_queue/  (default: public)
.venv/bin/python -m youtube_pipeline

# Privacy override
.venv/bin/python -m youtube_pipeline --privacy unlisted
.venv/bin/python -m youtube_pipeline --privacy private

# Preview without uploading
.venv/bin/python -m youtube_pipeline --dry-run

# Upload a specific file (does not need to be in the queue)
.venv/bin/python -m youtube_pipeline --file recordings/foo.mp4

# Daemon mode: scan queue every 30 s
.venv/bin/python -m youtube_pipeline --watch

# Print the YouTube channel "About" description and exit
# (paste into YouTube Studio → Customisation → Basic info → Description)
.venv/bin/python -m youtube_pipeline --print-channel-description
```

## Generated metadata

For each upload the pipeline builds:

- **Title** — `{label}: {hook} | 3D Cellular Automata` for long-form,
  `{label} — 3D {category} #Shorts` for vertical recordings. The
  `{hook}` is auto-extracted from the first em-dash clause of the
  preset's in-app description (e.g. `Vicsek-style active matter`,
  `Wavepacket hits a potential barrier`); category is mapped from the
  rule name (`Reaction-Diffusion`, `Quantum Mechanics`, `Active Matter`,
  `Crystal Growth`, …).
- **Description** — full preset description, "What you're seeing"
  one-liner with grid size + duration + frame count, parameter table,
  run details (rule shader, category, renderer, resolution, seed, dt,
  discovery score), the project blurb, source-code link, hashtags.
- **Tags** — generic CA / simulation / GPU tags plus the rule name,
  label and category, trimmed to YouTube's 500-char total.

To preview what would be generated without uploading:

```bash
.venv/bin/python -m youtube_pipeline --dry-run --file path/to/recording.mp4
```

## How it works

1. The simulator's recording UI has a **"Queue for upload"** checkbox. When enabled, the `.mp4` and `.json` sidecar are written to `recordings/upload_queue/` instead of `recordings/`.
2. Run `python -m youtube_pipeline` to upload everything queued.
3. The pipeline reads each `.json` sidecar, builds title/description/tags from the rule label, parameters, and discovery score, then uploads via the YouTube Data API v3 with chunked resumable transfer.
4. Routing is automatic from the resolution tag in the filename: vertical recordings (`*_1080x1920.mp4`) get `#Shorts` appended to the title; everything else uploads as a regular video.
5. After a successful upload, the source files are moved to `recordings/uploaded/YYYY-MM-DD/` and an entry is appended to `recordings/upload_log.jsonl`.

## Per-recording overrides

To override the auto-generated title / description / tags for one specific recording, drop a JSON file next to the sidecar with `_overrides.json` appended to the basename:

```
recordings/upload_queue/20260422_153012_Lenia_3D_2560x1440.mp4
recordings/upload_queue/20260422_153012_Lenia_3D_2560x1440.json
recordings/upload_queue/20260422_153012_Lenia_3D_2560x1440_overrides.json   ← optional
```

Override file (any subset of these keys):

```json
{
  "title": "Custom title here",
  "description": "Custom description here",
  "tags": ["custom", "tags"],
  "shorts": false,
  "category_id": "28"
}
```

## Gitignore

The following are gitignored on this branch (`youtube-pipeline`):

```
youtube_pipeline/credentials/
recordings/upload_queue/
recordings/uploaded/
recordings/upload_log.jsonl
```
