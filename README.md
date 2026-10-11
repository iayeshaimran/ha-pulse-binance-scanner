# HA Pulse — Binance Spot Scanner

A new standalone Streamlit web app for Binance USDT spot pairs. It does not modify or depend on the user's existing website.

## Features
- Separate, independent 1-hour and 3-minute scanner tabs.
- Binance public spot-market data; no API key needed.
- Uses fully closed candles for signals.
- READY when the latest closed Heikin Ashi candle turns green and price is above EMA 50.
- BUY when two consecutive closed Heikin Ashi candles are green, the second candle's lower wick is within the configured threshold, and price is above EMA 50.
- Entry at the signal candle's close; TP1 +1%; TP2 +2%.
- Stop-loss a small buffer below the lowest actual low in the configured recent swing lookback. The displayed risk percentage may exceed 1.5% because a structural swing-low stop is not artificially capped.
- Browser alert for fresh BUY signals, CSV export, and TradingView chart links.
- Select all Binance USDT spot pairs or a top-volume subset.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy with GitHub + Streamlit Community Cloud
1. Create a new GitHub repository (keep this project separate from your existing site).
2. Upload `app.py`, `requirements.txt`, and `README.md`.
3. In Streamlit Community Cloud, deploy the new repository and choose `app.py` as the main file.

## Important behavior
- Both tabs only scan when their own Scan button is pressed or their own auto-refresh option is enabled. 1H and 3M results are stored separately.
- Auto-refresh runs when Streamlit reruns the page; if your hosting setup sleeps or the browser tab is closed, it will not be a 24/7 background service.
- Browser alerts may be blocked by the browser or embedded component. Keep the app open and allow alerts if prompted.
- The app may take time or be rate-limited when scanning all pairs. Try a top-volume subset or lower parallel requests if needed.
- This is an informational tool, not financial advice. Test before live use.
