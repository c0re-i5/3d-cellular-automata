"""PRAW authentication for the Reddit submission pipeline.

Uses Reddit's "script app" OAuth flow: the bot logs in as the channel
owner using username + password + the script app's client_id / secret.
Credentials live in ``credentials/reddit_secrets.json`` (gitignored).

Create a script app at <https://www.reddit.com/prefs/apps>:
  type:        script
  name:        anything (e.g. "ca-reddit-bot")
  redirect uri: http://localhost:8080  (unused by script apps but required)

Then write ``reddit_pipeline/credentials/reddit_secrets.json``:
  {
      "client_id":     "<14-char id under the app name>",
      "client_secret": "<secret string>",
      "username":      "<your reddit username>",
      "password":      "<your reddit password>",
      "user_agent":    "ca-reddit-bot/0.1 by <your username>"
  }
"""
from __future__ import annotations

import json
from pathlib import Path

import praw

CREDENTIALS_DIR = Path(__file__).parent / 'credentials'
SECRETS_PATH = CREDENTIALS_DIR / 'reddit_secrets.json'


def get_reddit() -> praw.Reddit:
    """Return an authenticated PRAW Reddit instance.

    Raises FileNotFoundError if ``reddit_secrets.json`` is missing —
    follow the steps in ``reddit_pipeline/README.md``.
    """
    if not SECRETS_PATH.exists():
        raise FileNotFoundError(
            f"Missing {SECRETS_PATH}.\n"
            f"Create a Reddit script app at "
            f"https://www.reddit.com/prefs/apps and write the credentials "
            f"as JSON (see reddit_pipeline/README.md)."
        )
    secrets = json.loads(SECRETS_PATH.read_text())
    required = {'client_id', 'client_secret', 'username',
                'password', 'user_agent'}
    missing = required - secrets.keys()
    if missing:
        raise ValueError(
            f"{SECRETS_PATH}: missing keys {sorted(missing)}")
    reddit = praw.Reddit(
        client_id=secrets['client_id'],
        client_secret=secrets['client_secret'],
        username=secrets['username'],
        password=secrets['password'],
        user_agent=secrets['user_agent'],
        # We want submissions, not just reads.
        check_for_async=False,
    )
    # Force a token fetch so bad creds fail fast with a clear error
    # instead of inside .submit() with a confusing 401.
    _ = reddit.user.me()
    return reddit
