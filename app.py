import streamlit as st
import requests
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from zoneinfo import ZoneInfo

# ==========================================================
# HA PULSE - BINANCE USDT SPOT REVERSAL SCANNER
# READY = early/provisional setup; BUY = closed-candle confirmation
# Informational only. It does not place trades.
# ==========================================================

st.set_page_config(
    page_title="HA Pulse - Binance Spot Scanner",
    page_icon="🟢",
    layout="wide",
)

BASE_URL = "https://data-api.binance.vision"
PKT = ZoneInfo("Asia/Karachi")

st.markdown("""
<style>
.stApp { background-color: #0b1220; color: #e5e7eb; }
h1, h2, h3 { color: #f8fafc; }
.signal-card {
    background: #111c2e; border: 1px solid #27364d;
    border-radius: 12px; padding: 16px; margin-bottom: 10px;
}
.small-note { color: #94a3b8; font-size: 13px; }
</style>
""", unsafe_allow_html=True)

st.title("🟢 HA Pulse")
st.caption("Binance USDT Spot · RSI Reversal + Heikin Ashi + EMA 9")

st.warning(
    "READY is an early, provisional setup and can disappear before candle close. "
    "BUY is based on closed candles. This scanner does not execute trades and no signal guarantees profit."
)

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Scanner Settings")

universe_size = st.sidebar.selectbox("Coins to scan", [25, 50, 100, 200, 500], index=2)
max_workers = st.sidebar.selectbox("Parallel requests", [3, 5, 8, 10], index=1)
wick_limit = st.sidebar.slider("Maximum lower wick / HA body", 0.0, 1.0, 0.45, 0.05)
oversold_rsi = st.sidebar.slider("RSI oversold level", 20, 40, 35, 1)
recent_bars = st.sidebar.selectbox("Oversold must have occurred within last N closed candles", [3, 5, 8], index=1)
use_ema50_filter = st.sidebar.checkbox("Require price above EMA 50 (stricter, may be later)", value=False)
auto_refresh = st.sidebar.checkbox("Auto-rescan", value=True)
refresh_minutes = st.sidebar.selectbox("Auto-rescan interval (minutes)", [1, 2, 3, 5], index=2)

st.sidebar.markdown("---")
st.sidebar.write("🟡 READY: RSI turns upward from oversold area + current HA candle turns green.")
st.sidebar.write("🟢 BUY: RSI rising + closed HA candle green + close above EMA 9.")
st.sidebar.write("🔎 A no-chase filter hides signals after a sharp move.")

# Auto-refresh without an extra package. Meta refresh reloads the page and starts a fresh scan
# only when the user enables auto-rescan and clicks "Start / Rescan".
if auto_refresh:
    st.markdown(
        f'<meta http-equiv="refresh" content="{int(refresh_minutes) * 60}">',
        unsafe_allow_html=True,
    )

# ---------------- BINANCE DATA ----------------
@st.cache_data(ttl=300, show_spinner=False)
def get_usdt_symbols():
    r = requests.get(f"{BASE_URL}/api/v3/exchangeInfo", timeout=20)
    r.raise_for_status()
    symbols = []
    for item in r.json().get("symbols", []):
        if (
            item.get("status") == "TRADING"
            and item.get("quoteAsset") == "USDT"
            and item.get("isSpotTradingAllowed", False)
        ):
            symbols.append(item["symbol"])
    return sorted(set(symbols))


@st.cache_data(ttl=20, show_spinner=False)
def get_tickers():
    r = requests.get(f"{BASE_URL}/api/v3/ticker/24hr", timeout=30)
    r.raise_for_status()
    rows = []
    for item in r.json():
        symbol = item.get("symbol", "")
        if symbol.endswith("USDT"):
            try:
                rows.append({
                    "Coin": symbol,
                    "Price": float(item["lastPrice"]),
                    "24h Change %": float(item["priceChangePercent"]),
                    "24h Volume USDT": float(item["quoteVolume"]),
                })
            except (ValueError, KeyError, TypeError):
                continue
    return pd.DataFrame(rows)


def get_klines(symbol, interval, limit=120):
    r = requests.get(
        f"{BASE_URL}/api/v3/klines",
        params={"symbol": symbol, "interval": interval, "limit": limit},
        timeout=15,
    )
    r.raise_for_status()
    data = r.json()
    columns = [
        "time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_base",
        "taker_quote", "ignore",
    ]
    df = pd.DataFrame(data, columns=columns)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    df["close_time"] = pd.to_numeric(df["close_time"], errors="coerce")
    return df.dropna(subset=["open", "high", "low", "close", "close_time"]).reset_index(drop=True)


# ---------------- INDICATORS ----------------
def ema(series, period):
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def rsi(series, period=14):
    delta = series.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    value = 100 - (100 / (1 + rs))
    value = value.where(avg_loss != 0, 100)
    value = value.where(~((avg_gain == 0) & (avg_loss == 0)), 50)
    return value.fillna(50)


def heikin_ashi(df):
    ha = pd.DataFrame(index=df.index)
    ha["close"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    ha_open = np.zeros(len(df))
    if len(df) == 0:
        return ha
    ha_open[0] = (df["open"].iloc[0] + df["close"].iloc[0]) / 2
    for i in range(1, len(df)):
        ha_open[i] = (ha_open[i - 1] + ha["close"].iloc[i - 1]) / 2
    ha["open"] = ha_open
    ha["high"] = pd.concat([df["high"], ha["open"], ha["close"]], axis=1).max(axis=1)
    ha["low"] = pd.concat([df["low"], ha["open"], ha["close"]], axis=1).min(axis=1)
    return ha


def is_green(candle):
    return float(candle["close"]) > float(candle["open"])


def small_lower_wick(candle, limit):
    body = abs(float(candle["close"]) - float(candle["open"]))
    lower = min(float(candle["open"]), float(candle["close"])) - float(candle["low"])
    return body > 0 and lower <= body * limit


def tradingview_url(symbol, interval):
    # TradingView interval values: 3, 5, 15, 60
    tv_interval = {"3m": "3", "5m": "5", "15m": "15", "1h": "60", "4h": "240"}.get(interval, "3")
    return f"https://www.tradingview.com/chart/?symbol=BINANCE%3A{symbol}&interval={tv_interval}"


# ---------------- ANALYZE COIN ----------------
def analyze_coin(symbol, interval, wick_limit, oversold_level, recent_bars, use_ema50):
    try:
        raw = get_klines(symbol, interval, limit=120)
        if len(raw) < 60:
            return None

        # Binance kline close_time is in milliseconds. Remove the still-forming candle.
        now_ms = int(datetime.now().timestamp() * 1000)
        is_last_closed = float(raw["close_time"].iloc[-1]) < now_ms
        if is_last_closed:
            closed = raw.copy()
            live = None
        else:
            closed = raw.iloc[:-1].copy()
            live = raw.iloc[-1].copy()

        if len(closed) < 55:
            return None

        closed = closed.reset_index(drop=True)
        ha = heikin_ashi(closed)
        rsi_values = rsi(closed["close"], 14)
        ema9 = ema(closed["close"], 9)
        ema50 = ema(closed["close"], 50)

        if pd.isna(ema9.iloc[-1]) or pd.isna(ema50.iloc[-1]):
            return None

        last = closed.iloc[-1]
        last_ha = ha.iloc[-1]
        prev_ha = ha.iloc[-2]
        last_rsi = float(rsi_values.iloc[-1])
        prev_rsi = float(rsi_values.iloc[-2])
        last_ema9 = float(ema9.iloc[-1])
        last_ema50 = float(ema50.iloc[-1])
        close_price = float(last["close"])

        # Genuine-reversal filter: RSI is rising and was oversold recently,
        # or has just crossed up through the oversold level.
        lookback_start = max(0, len(closed) - recent_bars - 1)
        recent_low_rsi = float(rsi_values.iloc[lookback_start:-1].min()) if lookback_start < len(rsi_values) - 1 else last_rsi
        recently_oversold = recent_low_rsi <= oversold_level
        rsi_turning_up = last_rsi > prev_rsi
        rsi_reversal = recently_oversold and rsi_turning_up and last_rsi < 60
        ema50_ok = (close_price > last_ema50) if use_ema50 else True

        # No-chase: avoid signals if last closed candle is a large green impulse
        # or price has already stretched too far above EMA9.
        candle_range = max(float(last["high"]) - float(last["low"]), 1e-12)
        candle_body = abs(float(last["close"]) - float(last["open"]))
        big_impulse = (
            float(last["close"]) > float(last["open"])
            and candle_body / candle_range >= 0.75
            and candle_body / max(close_price, 1e-12) >= 0.018
        )
        stretched = close_price > last_ema9 * 1.025
        no_chase_ok = not big_impulse and not stretched

        # READY: early, live/provisional HA green forming, while closed RSI setup is valid.
        ready = False
        live_price = close_price
        if live is not None:
            live_price = float(live["close"])
            live_ha_close = (
                float(live["open"]) + float(live["high"])
                + float(live["low"]) + float(live["close"])
            ) / 4
            live_ha_open = (float(last_ha["open"]) + float(last_ha["close"])) / 2
            live_green = live_ha_close > live_ha_open
            live_above_ema9 = live_price > last_ema9
            live_not_stretched = live_price <= last_ema9 * 1.025
            ready = (
                rsi_reversal and live_green and live_above_ema9
                and ema50_ok and live_not_stretched
            )

        # BUY: closed HA candle green, close above EMA9, RSI turns upward,
        # recent oversold reading, no chase and optional EMA50 trend filter.
        two_green = is_green(prev_ha) and is_green(last_ha)
        buy = (
            rsi_reversal
            and is_green(last_ha)
            and close_price > last_ema9
            and small_lower_wick(last_ha, wick_limit)
            and no_chase_ok
            and ema50_ok
        )

        # Prioritize BUY. If no current valid setup exists, return None so
        # the next rescan removes the coin from the live list automatically.
        if buy:
            signal = "BUY"
        elif ready:
            signal = "READY"
        else:
            return None

        return {
            "Coin": symbol,
            "Signal": signal,
            "Price": live_price if signal == "READY" else close_price,
            "RSI 14": round(last_rsi, 2),
            "Previous RSI": round(prev_rsi, 2),
            "EMA 9": last_ema9,
            "EMA 50": last_ema50,
            "HA Green": "Yes" if is_green(last_ha) else "No",
            "2 HA Green": "Yes" if two_green else "No",
            "RSI Reversal": "Yes",
            "Timeframe": interval,
            "TradingView": tradingview_url(symbol, interval),
            "Updated (PKT)": datetime.now(PKT).strftime("%H:%M:%S"),
        }

    except (requests.RequestException, ValueError, KeyError, IndexError, TypeError, ZeroDivisionError):
        return None


# ---------------- MARKET SCAN ----------------
def scan_market(interval, wick_limit, universe_size, max_workers, oversold_level, recent_bars, use_ema50):
    symbols = get_usdt_symbols()
    tickers = get_tickers()
    if tickers.empty:
        return pd.DataFrame()

    tickers = tickers[tickers["Coin"].isin(symbols)]
    tickers = tickers.sort_values("24h Volume USDT", ascending=False).head(universe_size)
    scan_symbols = tickers["Coin"].tolist()
    if not scan_symbols:
        return pd.DataFrame()

    results = []
    progress = st.progress(0)
    status = st.empty()
    total = len(scan_symbols)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                analyze_coin, symbol, interval, wick_limit,
                oversold_level, recent_bars, use_ema50
            ): symbol
            for symbol in scan_symbols
        }
        completed = 0
        for future in as_completed(futures):
            completed += 1
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception:
                pass
            progress.progress(completed / total)
            status.caption(f"Scanning {interval}: {completed}/{total} coins checked")

    progress.empty()
    status.empty()

    if not results:
        return pd.DataFrame()

    out = pd.DataFrame(results)
    out["_order"] = out["Signal"].map({"BUY": 0, "READY": 1})
    return out.sort_values(["_order", "RSI 14"], ascending=[True, True]).drop(columns="_order").reset_index(drop=True)


# ---------------- RESULTS DISPLAY ----------------
def show_results(results, interval):
    if results.empty:
        st.info("No current valid signals. A fresh rescan will remove expired setups automatically.")
        return

    buys = results[results["Signal"] == "BUY"].copy()
    ready = results[results["Signal"] == "READY"].copy()
    a, b, c = st.columns(3)
    a.metric("Current Signals", len(results))
    b.metric("🟢 BUY", len(buys))
    c.metric("🟡 READY", len(ready))

    column_config = {
        "Price": st.column_config.NumberColumn("Price", format="%.8f"),
        "EMA 9": st.column_config.NumberColumn("EMA 9", format="%.8f"),
        "EMA 50": st.column_config.NumberColumn("EMA 50", format="%.8f"),
        "TradingView": st.column_config.LinkColumn("Open TradingView", display_text="Open chart"),
    }

    st.markdown("### 🟢 BUY Signals")
    if buys.empty:
        st.info("No confirmed BUY signals in the latest scan.")
    else:
        st.dataframe(buys, use_container_width=True, hide_index=True, column_config=column_config)

    st.markdown("### 🟡 READY Signals")
    if ready.empty:
        st.info("No early READY setups in the latest scan.")
    else:
        st.dataframe(ready, use_container_width=True, hide_index=True, column_config=column_config)

    st.markdown("### 📊 All Current Signals")
    st.dataframe(results, use_container_width=True, hide_index=True, column_config=column_config)

    st.download_button(
        f"⬇️ Download {interval} results CSV",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name=f"ha_pulse_{interval}_signals.csv",
        mime="text/csv",
        key=f"download_{interval}",
    )


def show_scanner(interval, label):
    st.subheader(label)
    st.caption(
        "READY = provisional green Heikin Ashi candle + RSI turning up after recent oversold + price above EMA9. "
        "BUY = closed green Heikin Ashi candle + close above EMA9 + RSI reversal + no-chase filter."
    )
    st.caption(f"Scanning top {universe_size} Binance USDT Spot pairs by 24-hour quote volume.")

    start_clicked = st.button(f"🔎 Start / Rescan {interval}", key=f"scan_{interval}", type="primary")
    if start_clicked:
        st.session_state[f"auto_scan_enabled_{interval}"] = True

    # After the first manual scan, page refreshes trigger fresh scans automatically.
    should_scan = start_clicked or (
        auto_refresh and st.session_state.get(f"auto_scan_enabled_{interval}", False)
    )
    if should_scan:
        try:
            with st.spinner(f"Scanning {interval} market..."):
                result = scan_market(
                    interval, wick_limit, universe_size, max_workers,
                    oversold_rsi, recent_bars, use_ema50_filter
                )
            # Replace old results entirely. Coins without a valid current setup disappear.
            st.session_state[f"results_{interval}"] = result
            st.session_state[f"scan_time_{interval}"] = datetime.now(PKT).strftime("%d %b %Y, %I:%M:%S %p PKT")
        except Exception as exc:
            st.error(f"Scan failed. Check your internet connection and try again. Details: {exc}")

    results = st.session_state.get(f"results_{interval}", pd.DataFrame())
    last_scan = st.session_state.get(f"scan_time_{interval}")
    if last_scan:
        st.caption(f"Last scan: {last_scan}")

    if auto_refresh:
        st.caption(f"Auto-refresh is enabled ({refresh_minutes} minute(s)). Click Start / Rescan once to begin; page refreshes on the selected interval.")
    show_results(results, interval)


# ---------------- MAIN APP ----------------
tab_1h, tab_3m = st.tabs(["🕐 1H Reversal Scanner", "⚡ 3M Scalp Scanner"])

with tab_1h:
    show_scanner("1h", "1-hour reversal scanner")

with tab_3m:
    show_scanner("3m", "3-minute spot scalp scanner")

st.markdown("---")
st.caption(
    "HA Pulse · Binance public Spot market data · READY is provisional; BUY uses closed candles. "
    "Signals are informational only, not financial advice. The scanner does not place orders."
)

