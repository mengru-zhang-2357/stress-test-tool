# Stress Test Tool

## Launch the app

Run the app through the dedicated launcher so websocket keepalive settings are applied explicitly:

```bash
python run_app.py
```

Then open http://127.0.0.1:8000 in your browser.

## Why use `run_app.py`

The launcher starts the ASGI app with explicit websocket keepalive settings (`ws_ping_interval=30`, `ws_ping_timeout=60`) that are more stable for local desktop usage where tabs may be briefly idle/backgrounded.
