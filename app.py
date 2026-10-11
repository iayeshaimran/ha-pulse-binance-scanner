import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components

APP_TITLE = "HA Pulse — Binance Spot Scanner"
BASE_URL = "https://data-api.binance.vision"

st.set_page_config(page_title=APP_TITLE, page_icon="🟢", layout="wide")

# ---------- Styling ----------
st.markdown("""
<style>
  .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
  .hero {padding: 1.25rem 1.4rem; border-radius: 18px; background: linear-gradient(120deg,#10251f,#172033); border:1px solid #29433b; margin-bottom:1rem;}
  .hero h1 {margin:0; color:#f3fff8; font-size:2rem;}
  .hero p {margin:.35rem 0 0; color:#b8c8c1;}
  .signal-ready {color:#eab308; font-weight:800;}
  .signal-buy {color:#22c55e; font-weight:900;}
  .muted {color:#8a96a3; font-size:.9rem;}
  div[data-testid="stMetric"] {background:rgba(128,128,128,.07); padding:.7rem; border-radius:12px; border:1px solid rgba(128,128,128,.15);}
</style>
<div class="hero"><h1>🟢 HA Pulse — Binance Spot Scanner</h1><p>Independent 1H and 3M Heikin Ashi setups · EMA 50 trend filter · Entry / targets / swing-low stop</p></div>
""", unsafe_allow_html=True)

# ---------- HTTP helpers ----------
@st.cache_data(ttl=300, show_spinner=False)
def get_spot_symbols():
    r = requests.get(f"{BASE_URL}/api/v3/exchangeInfo", timeout=20)
    r.raise_for_status()
    payload = r.json()
    symbols = []
    for s in payload.get("symbols", []):
        if (s.get("status") == "TRADING" and s.get("quoteAsset") == "USDT"
                and s.get("isSpotTradingAllowed", False) and s.get("symbol", "").endswith("USDT")):
            symbols.append(s["symbol"])
    return sorted(set(symbols))

@st.cache_data(ttl=60, show_spinner=False)
def get_tickers_24h():
    r = requests.get(f"{BASE_URL}/api/v3/ticker/24hr", timeout=25)
    r.raise_for_status()
    data = r.json()
    return {x["symbol"]: x for x in data if "symbol" in x}


def get_klines(symbol, interval, limit=100):
    # Binance kline endpoint; last candle may be in progress and is removed below.
    r = requests.get(f"{BASE_URL}/api/v3/klines", params={"symbol": symbol, "interval": interval, "limit": limit}, timeout=12)
    r.raise_for_status()
    raw = r.json()
    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "trades", "taker_base", "taker_quote", "ignore"]
    df = pd.DataFrame(raw, columns=cols)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce")
    df["close_time"] = pd.to_numeric(df["close_time"], errors="coerce")
    # Keep the in-progress candle for early READY detection; BUY is evaluated only on closed candles.
    return df.reset_index(drop=True)


def add_heikin_ashi(df):
    out = df.copy()
    if out.empty:
        return out
    ha_close = (out["open"] + out["high"] + out["low"] + out["close"]) / 4
    ha_open = [float((out.iloc[0]["open"] + out.iloc[0]["close"]) / 2)]
    for i in range(1, len(out)):
        ha_open.append((ha_open[i-1] + float(ha_close.iloc[i-1])) / 2)
    out["ha_open"] = ha_open
    out["ha_close"] = ha_close
    out["ha_high"] = out[["high"]].join(pd.DataFrame({"ha_open": out["ha_open"], "ha_close": ha_close})).max(axis=1)
    out["ha_low"] = out[["low"]].join(pd.DataFrame({"ha_open": out["ha_open"], "ha_close": ha_close})).min(axis=1)
    out["ha_green"] = out["ha_close"] > out["ha_open"]
    out["lower_wick"] = (out[["ha_open", "ha_close"]].min(axis=1) - out["ha_low"]).clip(lower=0)
    out["ha_range"] = (out["ha_high"] - out["ha_low"]).clip(lower=0)
    out["lower_wick_pct"] = out.apply(lambda r: (r["lower_wick"] / r["ha_range"] * 100) if r["ha_range"] > 0 else 100.0, axis=1)
    out["ema50"] = out["close"].ewm(span=50, adjust=False).mean()
    return out


def analyse_symbol(symbol, interval, wick_limit_pct, swing_lookback):
    try:
        df = get_klines(symbol, interval, limit=max(100, swing_lookback + 60))
        if len(df) < 56:
            return None
        now_ms = int(time.time() * 1000)
        closed_df = df[df["close_time"] < now_ms].copy().reset_index(drop=True)
        live_df = df[df["close_time"] >= now_ms].copy().reset_index(drop=True)
        if len(closed_df) < 55:
            return None

        # BUY is based exclusively on the two latest CLOSED candles.
        closed = add_heikin_ashi(closed_df)
        prev, cur = closed.iloc[-2], closed.iloc[-1]
        price = float(cur["close"])
        ema_ok = price > float(cur["ema50"])
        prev_green, cur_green = bool(prev["ha_green"]), bool(cur["ha_green"])
        small_wick = float(cur["lower_wick_pct"]) <= wick_limit_pct
        buy = prev_green and cur_green and small_wick and ema_ok

        # READY is earlier: evaluate the currently forming candle if it has turned HA-green,
        # while the most recent closed candle was not green. This is provisional and can disappear.
        ready = False
        ready_price = price
        ready_wick_pct = float(cur["lower_wick_pct"])
        if not live_df.empty:
            combined = pd.concat([closed_df.tail(80), live_df.tail(1)], ignore_index=True)
            live_ha = add_heikin_ashi(combined)
            live_prev, live_cur = live_ha.iloc[-2], live_ha.iloc[-1]
            ready_price = float(live_cur["close"])
            ready = bool(live_cur["ha_green"]) and not bool(live_prev["ha_green"]) and ready_price > float(live_cur["ema50"])
            ready_wick_pct = float(live_cur["lower_wick_pct"])

        # Lowest actual low over recent CLOSED candles before the BUY signal candle.
        swing_window = closed.iloc[-(swing_lookback + 1):-1]
        swing_low = float(swing_window["low"].min()) if not swing_window.empty else float(cur["low"])
        # Place stop 0.1% below the structural swing low; no arbitrary fixed-% cap.
        stop = swing_low * 0.999
        risk_pct = max(0.0, (price - stop) / price * 100) if price else 0.0
        if stop >= price:
            stop = float(cur["low"]) * 0.999
            risk_pct = max(0.0, (price - stop) / price * 100) if price else 0.0
        tp1, tp2 = price * 1.01, price * 1.02
        close_time = datetime.fromtimestamp(float(cur["close_time"]) / 1000, tz=timezone.utc)
        return {
            "Coin": symbol,
            "Signal": "BUY" if buy else ("READY" if ready else "—"),
            "Price": price,
            "EMA 50": float(cur["ema50"]),
            "HA lower wick %": ready_wick_pct if ready and not buy else float(cur["lower_wick_pct"]),
            "Swing low": swing_low,
            "Stop loss": stop,
            "SL risk %": risk_pct,
            "TP1 (+1%)": tp1,
            "TP2 (+2%)": tp2,
            "Candle close (UTC)": close_time.strftime("%Y-%m-%d %H:%M"),
            "_buy": buy,
            "_ready": ready and not buy,
            "_close_ms": int(cur["close_time"]),
            "_interval": interval,
        }
    except Exception:
        return None


def scan_market(interval, symbols, wick_limit_pct, swing_lookback, workers=6, progress_slot=None):
    rows = []
    total = len(symbols)
    if total == 0:
        return rows
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(analyse_symbol, sym, interval, wick_limit_pct, swing_lookback): sym for sym in symbols}
        for future in as_completed(futures):
            completed += 1
            try:
                item = future.result()
                if item:
                    rows.append(item)
            except Exception:
                pass
            if progress_slot is not None:
                progress_slot.progress(min(completed / total, 1.0), text=f"Scanning {interval} · {completed:,}/{total:,} pairs")
    return rows


def show_browser_alerts(rows, interval):
    buys = [r for r in rows if r.get("_buy")]
    if not buys:
        return
    # The key avoids repeated alerts for the same closed candle during reruns.
    latest = [{"symbol": r["Coin"], "close": r["_close_ms"], "interval": interval} for r in buys]
    import json
    payload = json.dumps(latest)
    components.html(f"""
    <script>
    const alerts = {payload};
    try {{
      const key = 'ha_pulse_seen_buys_v1';
      const seen = JSON.parse(localStorage.getItem(key) || '{{}}');
      const fresh = alerts.filter(a => seen[a.interval + ':' + a.symbol] !== a.close);
      if (fresh.length) {{
        fresh.forEach(a => {{ seen[a.interval + ':' + a.symbol] = a.close; }});
        localStorage.setItem(key, JSON.stringify(seen));
        const msg = fresh.map(a => a.symbol + ' BUY (' + a.interval + ')').join('\\n');
        window.alert('HA PULSE BUY SIGNAL\\n' + msg);
      }}
    }} catch (e) {{}}
    </script>
    """, height=0)


def fmt_price(x):
    try:
        x = float(x)
        if x >= 1000: return f"{x:,.3f}"
        if x >= 1: return f"{x:.5f}"
        return f"{x:.8f}".rstrip("0").rstrip(".")
    except Exception:
        return str(x)


def present_results(rows, interval, tickers):
    if not rows:
        st.warning("No valid candle data returned. Please try again in a moment.")
        return
    show_browser_alerts(rows, interval)
    buys = [r for r in rows if r["_buy"]]
    ready = [r for r in rows if r["_ready"]]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pairs scanned", f"{len(rows):,}")
    c2.metric("BUY signals", f"{len(buys):,}")
    c3.metric("READY setups", f"{len(ready):,}")
    c4.metric("Timeframe", interval)
    st.caption("Signals use CLOSED candles only. Browser alerts may require allowing pop-ups/alerts for this site; keep the tab open while scanning.")

    df = pd.DataFrame(rows)
    # Sort BUY first, then READY, then by 24h quote volume where available.
    priority = {"BUY": 0, "READY": 1, "—": 2}
    df["_priority"] = df["Signal"].map(priority).fillna(3)
    df["_volume"] = df["Coin"].map(lambda s: float(tickers.get(s, {}).get("quoteVolume", 0) or 0))
    df = df.sort_values(["_priority", "_volume"], ascending=[True, False])
    display_cols = ["Coin", "Signal", "Price", "EMA 50", "HA lower wick %", "Swing low", "Stop loss", "SL risk %", "TP1 (+1%)", "TP2 (+2%)", "Candle close (UTC)"]
    out = df[display_cols].copy()
    for col in ["Price", "EMA 50", "Swing low", "Stop loss", "TP1 (+1%)", "TP2 (+2%)"]:
        out[col] = out[col].map(fmt_price)
    out["HA lower wick %"] = out["HA lower wick %"].map(lambda v: f"{v:.2f}%")
    out["SL risk %"] = out["SL risk %"].map(lambda v: f"{v:.2f}%")
    st.dataframe(out, use_container_width=True, hide_index=True, height=560,
                 column_config={"Coin": st.column_config.TextColumn("Coin", help="Binance USDT spot pair"),
                                "Signal": st.column_config.TextColumn("Signal", help="BUY / READY / no setup")})
    if buys:
        st.subheader("BUY setups")
        cards = st.columns(min(3, len(buys)))
        for i, row in enumerate(buys[:9]):
            with cards[i % len(cards)]:
                st.markdown(f"### 🟢 {row['Coin']}")
                st.write(f"**Entry:** {fmt_price(row['Price'])}")
                st.write(f"**TP1 (+1%):** {fmt_price(row['TP1 (+1%)'])}")
                st.write(f"**TP2 (+2%):** {fmt_price(row['TP2 (+2%)'])}")
                st.write(f"**SL below swing low:** {fmt_price(row['Stop loss'])}")
                st.write(f"**SL distance:** {row['SL risk %']:.2f}%")
                st.link_button("Open TradingView chart", f"https://www.tradingview.com/chart/?symbol=BINANCE%3A{row['Coin']}", use_container_width=True)
    csv = out.to_csv(index=False).encode("utf-8")
    st.download_button(f"Download {interval} scan as CSV", csv, file_name=f"ha_pulse_{interval}_scan.csv", mime="text/csv")

# ---------- Sidebar settings ----------
with st.sidebar:
    st.header("Scanner settings")
    wick_limit = st.slider("Maximum lower wick (% of HA candle range)", 0.0, 30.0, 8.0, 0.5)
    swing_lookback = st.slider("Swing-low lookback candles", 3, 20, 7)
    max_pairs = st.selectbox("Pairs to scan", ["All Binance USDT spot pairs", "Top 150 by 24h volume", "Top 300 by 24h volume"], index=0)
    workers = st.slider("Parallel requests", 2, 10, 6)
    st.caption("More pairs take longer and may hit Binance public API limits. Reduce parallel requests if scans fail.")
    st.divider()
    st.markdown("**Risk note**")
    st.caption("Targets are price levels, not guaranteed profit. Swing-low stops can be wider than 1–1.5%; check the displayed SL risk before entering.")

# Load pair metadata once for the current cache window.
try:
    with st.spinner("Loading Binance spot pairs…"):
        all_symbols = get_spot_symbols()
        tickers = get_tickers_24h()
except Exception as e:
    st.error(f"Could not load Binance market data: {e}")
    st.info("Check your internet connection or try again later. The app uses Binance public market-data endpoints; no API key is required.")
    st.stop()

ranked_symbols = sorted(all_symbols, key=lambda s: float(tickers.get(s, {}).get("quoteVolume", 0) or 0), reverse=True)
if max_pairs == "Top 150 by 24h volume":
    scan_symbols = ranked_symbols[:150]
elif max_pairs == "Top 300 by 24h volume":
    scan_symbols = ranked_symbols[:300]
else:
    scan_symbols = ranked_symbols

# Two deliberately independent modes: only the selected timeframe is scanned.
mode_1h, mode_3m = st.tabs(["🕐 1H Heikin Ashi Scanner", "⚡ 3M Heikin Ashi Scanner"])

with mode_1h:
    st.subheader("1-hour scanner")
    st.markdown("**READY:** the currently forming HA candle turns green while price is above EMA 50 (provisional). **BUY:** two consecutive closed green HA candles, second candle has a small lower wick, and price remains above EMA 50.")
    st.caption(f"Universe: {len(scan_symbols):,} Binance USDT spot pairs · This tab scans only 1h candles.")
    col_a, col_b = st.columns([1, 2])
    with col_a:
        scan_1h = st.button("Scan 1H market", type="primary", use_container_width=True, key="scan_1h")
    with col_b:
        auto_1h = st.checkbox("Auto-refresh 1H scan", value=False, key="auto_1h")
    refresh_1h = st.selectbox("1H refresh interval", [1, 3, 5, 10, 15], index=2, format_func=lambda x: f"Every {x} minute(s)", key="refresh_1h")
    if scan_1h or (auto_1h and st.session_state.get("last_auto_1h", 0) + refresh_1h * 60 <= time.time()):
        if auto_1h:
            st.session_state["last_auto_1h"] = time.time()
        prog = st.progress(0, text="Preparing 1H scan…")
        with st.spinner("Scanning 1H candles across Binance spot pairs…"):
            rows_1h = scan_market("1h", scan_symbols, wick_limit, swing_lookback, workers, prog)
        prog.empty()
        st.session_state["rows_1h"] = rows_1h
        st.session_state["scan_time_1h"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if "rows_1h" in st.session_state:
        st.caption("Last scan: " + st.session_state.get("scan_time_1h", "—"))
        present_results(st.session_state["rows_1h"], "1h", tickers)
    else:
        st.info("Press **Scan 1H market** to scan the 1-hour timeframe. The 3-minute scanner does not affect this scanner.")

with mode_3m:
    st.subheader("3-minute scanner")
    st.markdown("**READY:** the currently forming HA candle turns green while price is above EMA 50 (provisional). **BUY:** two consecutive closed green HA candles, second candle has a small lower wick, and price remains above EMA 50.")
    st.caption(f"Universe: {len(scan_symbols):,} Binance USDT spot pairs · This tab scans only 3m candles.")
    col_a, col_b = st.columns([1, 2])
    with col_a:
        scan_3m = st.button("Scan 3M market", type="primary", use_container_width=True, key="scan_3m")
    with col_b:
        auto_3m = st.checkbox("Auto-refresh 3M scan", value=False, key="auto_3m")
    refresh_3m = st.selectbox("3M refresh interval", [1, 2, 3, 5, 10], index=2, format_func=lambda x: f"Every {x} minute(s)", key="refresh_3m")
    if scan_3m or (auto_3m and st.session_state.get("last_auto_3m", 0) + refresh_3m * 60 <= time.time()):
        if auto_3m:
            st.session_state["last_auto_3m"] = time.time()
        prog = st.progress(0, text="Preparing 3M scan…")
        with st.spinner("Scanning 3M candles across Binance spot pairs…"):
            rows_3m = scan_market("3m", scan_symbols, wick_limit, swing_lookback, workers, prog)
        prog.empty()
        st.session_state["rows_3m"] = rows_3m
        st.session_state["scan_time_3m"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if "rows_3m" in st.session_state:
        st.caption("Last scan: " + st.session_state.get("scan_time_3m", "—"))
        present_results(st.session_state["rows_3m"], "3m", tickers)
    else:
        st.info("Press **Scan 3M market** to scan the 3-minute timeframe. The 1-hour scanner does not affect this scanner.")

st.divider()
st.caption("HA Pulse is an informational scanner, not financial advice. Public Binance data can be delayed or rate-limited. Always verify the chart and manage risk before trading.")
