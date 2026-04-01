"""Local launcher for the Shiny app with explicit websocket keepalive settings.

Using an explicit Uvicorn startup path gives us control over websocket ping
settings, which helps avoid transient keepalive assertion failures that can
appear when browser tabs are left idle and then re-focused.
"""

from __future__ import annotations

import uvicorn

from app import app


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        log_level="info",
        # Desktop/local users frequently background tabs for short periods.
        # A 30s ping interval and 60s timeout are conservative enough to keep
        # connections healthy without aggressively dropping briefly idle clients.
        ws_ping_interval=30.0,
        ws_ping_timeout=60.0,
    )
