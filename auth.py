"""
auth.py
Handles OAuth2 client-credentials authentication for the OpenSky API.

OpenSky issues short-lived (30 min) bearer tokens. This module exposes
a single shared TokenManager instance that fetcher.py (or any other
module that talks to OpenSky) can call to get a always-valid token,
without needing to know anything about expiry or refresh logic.
"""

import os
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load CLIENT_ID / CLIENT_SECRET from a local .env file if present.
# In production (e.g. Streamlit Cloud secrets), these would instead
# be set directly as real environment variables.
load_dotenv()

TOKEN_URL = (
    "https://auth.opensky-network.org/auth/realms/opensky-network"
    "/protocol/openid-connect/token"
)

# How many seconds before expiry to proactively refresh the token.
TOKEN_REFRESH_MARGIN = 30


class TokenManager:
    """
    Manages a single OAuth2 access token for the OpenSky API.

    Call `.headers()` before every request — it returns valid auth
    headers and transparently refreshes the token if it's expired
    or about to expire.
    """

    def __init__(self, client_id: str | None = None, client_secret: str | None = None):
        # Fall back to environment variables if not passed explicitly.
        self.client_id = client_id or os.environ.get("OPENSKY_CLIENT_ID")
        self.client_secret = client_secret or os.environ.get("OPENSKY_CLIENT_SECRET")

        if not self.client_id or not self.client_secret:
            raise ValueError(
                "Missing OpenSky credentials. Set OPENSKY_CLIENT_ID and "
                "OPENSKY_CLIENT_SECRET as environment variables or in a .env file."
            )

        self.token = None
        self.expires_at = None

    def get_token(self, force_refresh: bool = False) -> str:
        """Return a valid access token, refreshing automatically if needed."""
        if (
            not force_refresh
            and self.token
            and self.expires_at
            and datetime.now() < self.expires_at
        ):
            return self.token
        return self._refresh()

    def _refresh(self) -> str:
        """Fetch a new access token from the OpenSky authentication server."""
        response = requests.post(
            TOKEN_URL,
            data={
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            },
            timeout=10,
        )
        response.raise_for_status()

        data = response.json()
        self.token = data["access_token"]
        expires_in = data.get("expires_in", 1800)
        self.expires_at = datetime.now() + timedelta(
            seconds=expires_in - TOKEN_REFRESH_MARGIN
        )
        return self.token

    def headers(self, force_refresh: bool = False) -> dict:
        """Return request headers with a valid Bearer token."""
        return {"Authorization": f"Bearer {self.get_token(force_refresh=force_refresh)}"}


# Single shared instance — every other module imports THIS,
# rather than creating its own TokenManager, so the token is
# only fetched/refreshed once across the whole app.
#
# Built lazily on first use (not at import time) so that simply
# importing this module doesn't crash before .env is set up.
_tokens_instance = None


class _LazyTokenManager:
    """Defers TokenManager creation until the first real call is made."""

    def _get(self):
        global _tokens_instance
        if _tokens_instance is None:
            _tokens_instance = TokenManager()
        return _tokens_instance

    def headers(self, force_refresh: bool = False) -> dict:
        return self._get().headers(force_refresh=force_refresh)

    def get_token(self, force_refresh: bool = False) -> str:
        return self._get().get_token(force_refresh=force_refresh)


tokens = _LazyTokenManager()


if __name__ == "__main__":
    # Quick manual test: confirms credentials work and a token is issued.
    # This only fails here (not on import) if .env isn't set up yet.
    print("Requesting token...")
    headers = tokens.headers()
    print("Got headers:", headers)