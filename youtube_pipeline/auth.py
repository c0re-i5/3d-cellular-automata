"""OAuth 2.0 dance for the YouTube Data API v3.

First run pops a browser, asks you to grant the app permission to
upload to your channel, and writes ``token.json`` next to
``client_secret.json``. Subsequent runs reuse and refresh the token
silently.

Files written by this module are gitignored — they contain credentials.
"""
from __future__ import annotations

import os
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# Upload + manage videos on the authenticated user's channel.
SCOPES = ['https://www.googleapis.com/auth/youtube.upload']

CREDENTIALS_DIR = Path(__file__).parent / 'credentials'
CLIENT_SECRET_PATH = CREDENTIALS_DIR / 'client_secret.json'
TOKEN_PATH = CREDENTIALS_DIR / 'token.json'


def get_youtube_service():
    """Return an authenticated YouTube Data API v3 service object.

    Raises FileNotFoundError if ``client_secret.json`` is missing —
    follow the steps in ``youtube_pipeline/README.md`` to create one.
    """
    CREDENTIALS_DIR.mkdir(exist_ok=True)
    creds: Credentials | None = None

    if TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)

    # If there are no (valid) credentials, run the install flow.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not CLIENT_SECRET_PATH.exists():
                raise FileNotFoundError(
                    f"Missing {CLIENT_SECRET_PATH}.\n"
                    f"Create OAuth credentials in Google Cloud Console "
                    f"(see youtube_pipeline/README.md) and place the "
                    f"downloaded JSON at that path."
                )
            flow = InstalledAppFlow.from_client_secrets_file(
                str(CLIENT_SECRET_PATH), SCOPES)
            # ``run_local_server`` opens a browser; the user grants
            # consent, the local server captures the redirect.
            creds = flow.run_local_server(port=0)
        TOKEN_PATH.write_text(creds.to_json())
        os.chmod(TOKEN_PATH, 0o600)

    return build('youtube', 'v3', credentials=creds, cache_discovery=False)
