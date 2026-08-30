import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import datetime
import time
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Trend Strategy Backtester", layout="wide")

DEFAULT_TICKERS = "SI=F, XU030.IS"


# ============================================================
#  CORE STRATEGY LOGIC (same as the original script, just
#  parametrized so fee / tp-range / ema-length are configurable)
# ============================================================

def backtest_long_only(df, signal_col, long_tp, fee):
    """
    Long-only strategy driven by a generic +1/-1 trend/signal column.
    Enter long when signal flips to +1 at that bar's close; exit either
    when the signal flips to -1, or when the running total P&L on the
    open position reaches long_tp — whichever comes first.
    """
    prices = df["Close"].values
    signal = df[signal_col].values
    n = len(prices)
    strat_rets = np.zeros(n)
    in_position = 0
    entry_price = 0.0
    current_signal = 0

    for i in range(1, n):
        if signal[i] != current_signal:
            current_signal = signal[i]
            if current_signal == 1:
                in_position = 1
                entry_price = prices[i]
                strat_rets[i] -= fee
            else:
                in_position = 0
                strat_rets[i] -= fee
            continue

        if in_position == 1:
            daily_pct = (prices[i] - prices[i - 1]) / prices[i - 1]
            strat_rets[i] += daily_pct
            total_pnl = (prices[i] - entry_price) / entry_price
            if total_pnl >= long_tp:
                strat_rets[i] -= fee
                in_position = 0

    return strat_rets


def backtest_short_only(df, signal_col, short_tp, fee):
    """
    Informational-only short-side backtest, used solely to find an
    "optimal" short take-profit level so the status section can quote a
    sensible target when the trend is DOWN. Not part of the traded
    strategy (which stays long-only / flat).
    """
    prices = df["Close"].values
    signal = df[signal_col].values
    n = len(prices)
    strat_rets = np.zeros(n)
    in_position = 0
    entry_price = 0.0
    current_signal = 0

    for i in range(1, n):
        if signal[i] != current_signal:
            current_signal = signal[i]
            if current_signal == -1:
                in_position = 1
                entry_price = prices[i]
                strat_rets[i] -= fee
            else:
                in_position = 0
                strat_rets[i] -= fee
            continue

        if in_position == 1:
            daily_pct = (prices[i] - prices[i - 1]) / prices[i - 1]
            strat_rets[i] += -daily_pct
            total_pnl = (entry_price - prices[i]) / entry_price
            if total_pnl >= short_tp:
                strat_rets[i] -= fee
                in_position = 0

    return strat_rets


def get_trade_entry_and_tp_long_only(df, signal_col, long_tp):
    """Most recent live long entry price and its take-profit target."""
    prices = df["Close"].values
    signal = df[signal_col].values
    n = len(prices)
    in_position = 0
    entry_price = 0.0
    current_signal = 0
    last_entry_price = None
    last_was_long = False

    for i in range(1, n):
        if signal[i] != current_signal:
            current_signal = signal[i]
            if current_signal == 1:
                in_position = 1
                entry_price = prices[i]
                last_entry_price = entry_price
                last_was_long = True
            else:
                in_position = 0
                last_was_long = False
            continue

        if in_position == 1:
            total_pnl = (prices[i] - entry_price) / entry_price
            if total_pnl >= long_tp:
                in_position = 0
                last_was_long = False

    if last_entry_price is not None and last_was_long:
        tp_price = last_entry_price * (1 + long_tp)
        return round(last_entry_price, 2), round(tp_price, 2)
    return None, None


def calculate_metrics(returns):
    if len(returns) == 0 or np.std(returns) == 0:
        return 0, 0, 0, 0
    ann_ret = np.mean(returns) * 252
    ann_vol = np.std(returns) * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol != 0 else 0

    cum_ret = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cum_ret)
    mdd = np.max((peak - cum_ret) / peak) if len(cum_ret) else 0
    calmar = ann_ret / mdd if mdd != 0 else 0
    return ann_ret, ann_vol, sharpe, calmar


def get_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def optimize_long_tp(df, signal_col, tp_ranges, fee):
    best_sharpe = -np.inf
    best_tp = tp_ranges[0]
    for l_tp in tp_ranges:
        rets = backtest_long_only(df, signal_col, l_tp, fee)
        _, _, sharpe, _ = calculate_metrics(rets)
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_tp = l_tp
    return best_tp


def optimize_short_tp(df, signal_col, tp_ranges, fee):
    best_sharpe = -np.inf
    best_tp = tp_ranges[0]
    for s_tp in tp_ranges:
        rets = backtest_short_only(df, signal_col, s_tp, fee)
        _, _, sharpe, _ = calculate_metrics(rets)
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_tp = s_tp
    return best_tp


def trend_status(df, signal_col, long_tp, short_tp):
    """
    Live status for a given signal column: current side, how many days
    the current streak has run, the entry level, unrealized P&L, and
    the take-profit / limit-order target for closing the position.
    """
    signal = df[signal_col].values
    prices = df["Close"].values
    dates = df.index
    n = len(prices)

    current_sig = signal[-1]

    entry_idx = n - 1
    for i in range(n - 2, -1, -1):
        if signal[i] != current_sig:
            break
        entry_idx = i

    days_in_trend = n - entry_idx
    entry_price = float(prices[entry_idx])
    entry_date = dates[entry_idx].strftime("%Y-%m-%d")
    current_price = float(prices[-1])

    if current_sig == 1:
        side = "LONG"
        pnl_pct = (current_price - entry_price) / entry_price * 100
        target_price = entry_price * (1 + long_tp)
        target_label = "Limit order (take-profit) to CLOSE the long"
    else:
        side = "DOWN / FLAT (long-only strategy takes no position)"
        pnl_pct = (entry_price - current_price) / entry_price * 100
        target_price = entry_price * (1 - short_tp)
        target_label = "Informational SHORT target (not traded)"

    return {
        "side": side,
        "days_in_trend": days_in_trend,
        "entry_price": round(entry_price, 2),
        "entry_date": entry_date,
        "current_price": round(current_price, 2),
        "pnl_pct": round(pnl_pct, 2),
        "target_price": round(target_price, 2),
        "target_label": target_label,
    }


@st.cache_data(ttl=3600, show_spinner=False)
def load_data(ticker, start, end, max_retries=3):
    """Fetches with retry+backoff. Yahoo Finance rate-limits repeated
    sequential requests (more likely the more tickers you fetch in one
    run, especially from a shared/cloud IP) -- when that happens,
    yfinance often doesn't raise a clean error, it just returns an
    empty frame or one whose Close values are all NaN. Retrying after a
    short backoff, rather than accepting the bad response immediately,
    resolves most of these transient cases."""
    last_df = None
    for attempt in range(max_retries):
        try:
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        except Exception:
            df = None

        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            last_df = df
            if "Close" in df.columns and df["Close"].notna().any():
                return df  # usable data -- done

        if attempt < max_retries - 1:
            time.sleep(1.5 * (attempt + 1))  # backoff before retrying

    return last_df  # best available result after retries, even if still unusable -- caller checks it


@st.cache_data(ttl=120, show_spinner=False)
def get_live_quote(ticker):
    """Best-effort real-time/delayed quote -- separate from the daily
    historical series used for signals/backtesting. This is what
    should match the price shown on the Yahoo Finance website, since
    that reflects live/intraday (and sometimes pre/after-market)
    trading, whereas the daily bar used elsewhere in this app only
    updates once a session is fully settled. Short TTL (2 min) since
    the whole point of this value is to be current."""
    try:
        fi = yf.Ticker(ticker).fast_info
        price = fi.get("last_price") if isinstance(fi, dict) else getattr(fi, "last_price", None)
        if price:
            return float(price)
    except Exception:
        pass
    try:
        info = yf.Ticker(ticker).info
        price = info.get("regularMarketPrice") or info.get("currentPrice")
        if price:
            return float(price)
    except Exception:
        pass
    return None


def preview_signal_with_live_price(df, live_price, ema_length):
    """Forward-looking, PROVISIONAL preview only -- does not touch df,
    the backtest returns, or any stat shown elsewhere. Treats the live
    quote as a stand-in for "today's close" and recomputes just the
    HA/EMA trend classification for that one hypothetical bar, so you
    can see what the signal would become IF the session settled right
    now. This is intentionally NOT fed back into df/backtest_long_only/
    optimize_*_tp -- doing so would let a still-moving intraday price
    flip the trend classification back and forth before the real close
    prints, making the backtest and 'Current Action' non-reproducible
    within the same day. Keep this strictly a preview."""
    if live_price is None or len(df) == 0:
        return None

    last_close = float(df["Close"].iloc[-1])
    prev_ha_close = float(df["HA_Close"].iloc[-1])
    prev_ha_open = float(df["HA_Open"].iloc[-1])
    prev_ema = float(df["EMA_Val"].iloc[-1])

    # Synthetic hypothetical bar: last settled close -> live price.
    o, c = last_close, live_price
    h, l = max(o, c), min(o, c)

    ha_close_new = (o + h + l + c) / 4
    ha_open_new = (prev_ha_open + prev_ha_close) / 2
    ha_trend_new = 1 if ha_close_new >= ha_open_new else -1

    alpha = 2 / (ema_length + 1)
    ema_new = c * alpha + prev_ema * (1 - alpha)
    ema_trend_new = 1 if c >= ema_new else -1

    return {"ha_trend": ha_trend_new, "ema_trend": ema_trend_new}


def build_signals(df, ema_length):
    df = df.copy()
    df["HA_Close"] = (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4
    ha_open = np.zeros(len(df))
    ha_open[0] = (df["Open"].iloc[0] + df["Close"].iloc[0]) / 2
    for i in range(1, len(df)):
        ha_open[i] = (ha_open[i - 1] + df["HA_Close"].iloc[i - 1]) / 2
    df["HA_Open"] = ha_open
    df["HA_Trend"] = np.where(df["HA_Close"] >= df["HA_Open"], 1, -1)

    df["EMA_Val"] = df["Close"].ewm(span=ema_length, adjust=False).mean()
    df["EMA_Trend"] = np.where(df["Close"] >= df["EMA_Val"], 1, -1)

    df["RSI"] = get_rsi(df["Close"])
    return df


# ============================================================
#  UI
# ============================================================

st.title("📈 Trend-Following Strategy Backtester")
st.caption(
    "Heikin-Ashi vs EMA trend, long-only, optimal take-profit search vs Buy & Hold "
    "— same logic as the original script, run across as many tickers as you like."
)

START_DATE = datetime.date(2026, 1, 1)
END_DATE = datetime.date.today() + datetime.timedelta(days=1)

with st.sidebar:
    st.header("Settings")
    tickers_input = st.text_input("Tickers (comma-separated)", value=DEFAULT_TICKERS)
    st.caption(f"Backtest window: **{START_DATE}** → **{END_DATE}** (fixed, start of 2026 to latest close)")
    fee = st.number_input(
        "Fee per trade (fraction)", value=0.0020, step=0.0005, format="%.4f"
    )
    ema_length = st.number_input("EMA length", value=9, min_value=2, max_value=200, step=1)

    st.subheader("Take-profit grid search")
    tp_min = st.number_input("Min TP", value=0.025, step=0.005, format="%.3f")
    tp_max = st.number_input("Max TP", value=0.15, step=0.005, format="%.3f")
    tp_step = st.number_input("Step", value=0.005, step=0.001, format="%.3f")

    run_button = st.button("Run Backtest", type="primary", use_container_width=True)

if not run_button:
    st.info("Enter one or more tickers in the sidebar (comma-separated) and click **Run Backtest**.")
    st.stop()

tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
if not tickers:
    st.warning("Please enter at least one ticker.")
    st.stop()

tp_ranges = np.arange(tp_min, tp_max + tp_step / 2, tp_step)
tp_ranges = tp_ranges[tp_ranges > 0]

for idx, ticker in enumerate(tickers):
    if idx > 0:
        time.sleep(0.8)  # brief pacing between sequential Yahoo Finance requests

    st.divider()
    st.subheader(f"📊 {ticker}")

    with st.spinner(f"Downloading & backtesting {ticker}..."):
        try:
            df = load_data(ticker, START_DATE.strftime("%Y-%m-%d"), END_DATE.strftime("%Y-%m-%d"))
        except Exception as e:
            st.error(f"Failed to download {ticker}: {e}")
            continue

        if df is None or df.empty or len(df) < 30:
            st.warning(f"No / insufficient price history for **{ticker}** in this date range.")
            continue

        # Some tickers (esp. near-continuously-traded futures like SI=F,
        # unlike single-daily-close equities/indices such as XU030.IS)
        # can come back from yfinance with a trailing row whose Close is
        # NaN -- an unsettled/incomplete bar. Drop those before anything
        # downstream reads df.iloc[-1], or "current price" silently
        # becomes NaN even though the ticker's history is otherwise fine.
        n_before = len(df)
        df = df.dropna(subset=["Close"])
        if len(df) < n_before:
            st.caption(f"ℹ️ Dropped {n_before - len(df)} trailing row(s) with no settled Close price for {ticker}.")

        if df.empty or len(df) < 30:
            st.warning(
                f"No / insufficient *settled* price history for **{ticker}** after removing "
                f"incomplete rows and retrying the fetch. This is often Yahoo Finance "
                f"rate-limiting rather than the ticker itself -- try clicking **Run Backtest** "
                f"again in a few seconds, or fetch fewer tickers at once."
            )
            continue

        try:
            df = build_signals(df, ema_length)

            best_tp_ha = optimize_long_tp(df, "HA_Trend", tp_ranges, fee)
            best_tp_ema = optimize_long_tp(df, "EMA_Trend", tp_ranges, fee)
            best_short_tp_ha = optimize_short_tp(df, "HA_Trend", tp_ranges, fee)
            best_short_tp_ema = optimize_short_tp(df, "EMA_Trend", tp_ranges, fee)

            ha_rets = backtest_long_only(df, "HA_Trend", best_tp_ha, fee)
            ema_rets = backtest_long_only(df, "EMA_Trend", best_tp_ema, fee)

            ha_ann_r, ha_ann_v, ha_sha, ha_cal = calculate_metrics(ha_rets)
            ema_ann_r, ema_ann_v, ema_sha, ema_cal = calculate_metrics(ema_rets)

            bh_rets = df["Close"].pct_change().fillna(0).values
            bh_ann_r, bh_ann_v, bh_sha, bh_cal = calculate_metrics(bh_rets)

            ha_trade_price, ha_tp_price = get_trade_entry_and_tp_long_only(df, "HA_Trend", best_tp_ha)
            ema_trade_price, ema_tp_price = get_trade_entry_and_tp_long_only(df, "EMA_Trend", best_tp_ema)

            cum_ha = np.cumprod(1 + ha_rets)
            cum_ema = np.cumprod(1 + ema_rets)
            cum_bh = np.cumprod(1 + bh_rets)

            last = df.iloc[-1]
        except Exception as e:
            st.error(f"Error while analyzing {ticker}: {e}")
            continue

    ha_status = trend_status(df, "HA_Trend", best_tp_ha, best_short_tp_ha)
    ema_status = trend_status(df, "EMA_Trend", best_tp_ema, best_short_tp_ema)

    current_price = float(last["Close"])
    live_price = get_live_quote(ticker)

    top1, top2, top3 = st.columns(3)
    settled_date_str = df.index[-1].strftime("%Y-%m-%d")
    top1.metric("Last Settled Close (used in backtest)", f"{current_price:.2f}")
    top1.caption(f"As of {settled_date_str}")
    if live_price is not None:
        gap_vs_settled = (live_price - current_price) / current_price * 100
        top2.metric("Live Quote (from Yahoo Finance)", f"{live_price:.2f}",
                    delta=f"{gap_vs_settled:+.2f}% vs settled close",
                    help="Real-time/delayed quote -- this is what the Yahoo Finance website shows, "
                         "and can differ from the settled daily close, especially for near-"
                         "continuously-traded tickers like futures, during/after market hours, or "
                         "for exchanges (like BIST) where the official close comes from a separate "
                         "closing auction rather than the last continuous trade.")
    else:
        top2.metric("Live Quote (from Yahoo Finance)", "n/a")
    top3.metric("RSI(14)", f"{float(last['RSI']):.1f}" if not np.isnan(last["RSI"]) else "n/a")

    st.caption("ℹ️ **Last Settled Close** is the completed daily bar the backtest and target levels "
               "below are calculated from. **Live Quote** is a separate, real-time lookup meant to "
               "match what you'd see on the Yahoo Finance website right now -- the two can genuinely "
               "differ until the current session settles.")

    if live_price is not None:
        preview = preview_signal_with_live_price(df, live_price, ema_length)
        if preview is not None:
            ha_now_long = ha_status["side"] == "LONG"
            ema_now_long = ema_status["side"] == "LONG"
            ha_preview_long = preview["ha_trend"] == 1
            ema_preview_long = preview["ema_trend"] == 1

            def _side_str(is_long):
                return "🟢 LONG" if is_long else "⚪ FLAT"

            def _flip_note(now_long, preview_long):
                return " *(would flip)*" if now_long != preview_long else ""

            with st.expander("🔮 Live preview — what the signal would be if today settled right now (provisional)"):
                st.caption(
                    "This is NOT part of the backtest, the table below, or the plot -- it's a "
                    "what-if using the live quote as a stand-in for today's close. It will keep "
                    "changing until the session actually settles, and can flip back before it does."
                )
                p1, p2 = st.columns(2)
                p1.write(f"**Heikin-Ashi:** {_side_str(ha_now_long)} (settled) → "
                          f"{_side_str(ha_preview_long)} (if settled at {live_price:.2f})"
                          f"{_flip_note(ha_now_long, ha_preview_long)}")
                p2.write(f"**EMA({ema_length}):** {_side_str(ema_now_long)} (settled) → "
                          f"{_side_str(ema_preview_long)} (if settled at {live_price:.2f})"
                          f"{_flip_note(ema_now_long, ema_preview_long)}")

    st.markdown("#### 🎯 Current Action & Target Levels (based on optimal TP thresholds)")
    a1, a2 = st.columns(2)
    for col, name, status, l_tp, s_tp in [
        (a1, "Heikin-Ashi", ha_status, best_tp_ha, best_short_tp_ha),
        (a2, f"EMA({ema_length})", ema_status, best_tp_ema, best_short_tp_ema),
    ]:
        with col:
            is_long = status["side"] == "LONG"
            action_label = "🟢 LONG — holding" if is_long else "⚪ FLAT — no position (long-only strategy)"
            st.markdown(f"**{name}**")
            st.write(f"Action: **{action_label}**")
            gap_pct = (status["target_price"] - current_price) / current_price * 100
            g1, g2 = st.columns(2)
            g1.metric("Settled Close", f"{current_price:.2f}")
            g2.metric(
                "Target Price" if is_long else "Informational Target",
                f"{status['target_price']:.2f}",
                delta=f"{gap_pct:+.2f}% away",
            )
            st.caption(
                f"Entry {status['entry_date']} @ {status['entry_price']} · "
                f"day {status['days_in_trend']} of trend · "
                f"unrealized P&L {status['pnl_pct']:+.2f}% · "
                f"optimal {'long' if is_long else 'short'} TP = "
                f"{(l_tp if is_long else s_tp)*100:.1f}%"
            )

    with st.expander("Full performance summary (vs Buy & Hold)"):
        summary_data = [
            {
                "Strategy": "Heikin-Ashi (long-only)",
                "Opt Long TP": f"{best_tp_ha*100:.1f}%",
                "Ann Ret": f"{ha_ann_r*100:.1f}%",
                "Ann Vol": f"{ha_ann_v*100:.1f}%",
                "Sharpe": round(ha_sha, 2),
                "Calmar": round(ha_cal, 2),
                "Trade Price": ha_trade_price,
                "TP Price": ha_tp_price,
                "TP Diff": None if ha_trade_price is None else round(ha_tp_price - ha_trade_price, 2),
            },
            {
                "Strategy": f"EMA({ema_length}) (long-only)",
                "Opt Long TP": f"{best_tp_ema*100:.1f}%",
                "Ann Ret": f"{ema_ann_r*100:.1f}%",
                "Ann Vol": f"{ema_ann_v*100:.1f}%",
                "Sharpe": round(ema_sha, 2),
                "Calmar": round(ema_cal, 2),
                "Trade Price": ema_trade_price,
                "TP Price": ema_tp_price,
                "TP Diff": None if ema_trade_price is None else round(ema_tp_price - ema_trade_price, 2),
            },
            {
                "Strategy": "Buy & Hold",
                "Opt Long TP": "—",
                "Ann Ret": f"{bh_ann_r*100:.1f}%",
                "Ann Vol": f"{bh_ann_v*100:.1f}%",
                "Sharpe": round(bh_sha, 2),
                "Calmar": round(bh_cal, 2),
                "Trade Price": round(float(df["Close"].iloc[0]), 2),
                "TP Price": "—",
                "TP Diff": "—",
            },
        ]
        results_df = pd.DataFrame(summary_data)
        st.dataframe(results_df, use_container_width=True, hide_index=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df.index, cum_ha, label=f"Heikin-Ashi (TP={best_tp_ha*100:.1f}%)", linewidth=1.8)
    ax.plot(df.index, cum_ema, label=f"EMA({ema_length}) (TP={best_tp_ema*100:.1f}%)", linewidth=1.8)
    ax.plot(df.index, cum_bh, label="Buy & Hold", linewidth=1.8, linestyle="--", color="gray")
    ax.set_title(f"{ticker} — Long-Only Strategy Comparison: Heikin-Ashi vs EMA vs Buy & Hold",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return (Growth of $1)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)









# import streamlit as st
# import pandas as pd
# import numpy as np
# import yfinance as yf
# import matplotlib.pyplot as plt
# import datetime
# import warnings

# warnings.filterwarnings("ignore")

# st.set_page_config(page_title="Trend Strategy Backtester", layout="wide")

# DEFAULT_TICKERS = "SI=F, XU030.IS"


# # ============================================================
# #  CORE STRATEGY LOGIC (same as the original script, just
# #  parametrized so fee / tp-range / ema-length are configurable)
# # ============================================================

# def backtest_long_only(df, signal_col, long_tp, fee):
#     """
#     Long-only strategy driven by a generic +1/-1 trend/signal column.
#     Enter long when signal flips to +1 at that bar's close; exit either
#     when the signal flips to -1, or when the running total P&L on the
#     open position reaches long_tp — whichever comes first.
#     """
#     prices = df["Close"].values
#     signal = df[signal_col].values
#     n = len(prices)
#     strat_rets = np.zeros(n)
#     in_position = 0
#     entry_price = 0.0
#     current_signal = 0

#     for i in range(1, n):
#         if signal[i] != current_signal:
#             current_signal = signal[i]
#             if current_signal == 1:
#                 in_position = 1
#                 entry_price = prices[i]
#                 strat_rets[i] -= fee
#             else:
#                 in_position = 0
#                 strat_rets[i] -= fee
#             continue

#         if in_position == 1:
#             daily_pct = (prices[i] - prices[i - 1]) / prices[i - 1]
#             strat_rets[i] += daily_pct
#             total_pnl = (prices[i] - entry_price) / entry_price
#             if total_pnl >= long_tp:
#                 strat_rets[i] -= fee
#                 in_position = 0

#     return strat_rets


# def backtest_short_only(df, signal_col, short_tp, fee):
#     """
#     Informational-only short-side backtest, used solely to find an
#     "optimal" short take-profit level so the status section can quote a
#     sensible target when the trend is DOWN. Not part of the traded
#     strategy (which stays long-only / flat).
#     """
#     prices = df["Close"].values
#     signal = df[signal_col].values
#     n = len(prices)
#     strat_rets = np.zeros(n)
#     in_position = 0
#     entry_price = 0.0
#     current_signal = 0

#     for i in range(1, n):
#         if signal[i] != current_signal:
#             current_signal = signal[i]
#             if current_signal == -1:
#                 in_position = 1
#                 entry_price = prices[i]
#                 strat_rets[i] -= fee
#             else:
#                 in_position = 0
#                 strat_rets[i] -= fee
#             continue

#         if in_position == 1:
#             daily_pct = (prices[i] - prices[i - 1]) / prices[i - 1]
#             strat_rets[i] += -daily_pct
#             total_pnl = (entry_price - prices[i]) / entry_price
#             if total_pnl >= short_tp:
#                 strat_rets[i] -= fee
#                 in_position = 0

#     return strat_rets


# def get_trade_entry_and_tp_long_only(df, signal_col, long_tp):
#     """Most recent live long entry price and its take-profit target."""
#     prices = df["Close"].values
#     signal = df[signal_col].values
#     n = len(prices)
#     in_position = 0
#     entry_price = 0.0
#     current_signal = 0
#     last_entry_price = None
#     last_was_long = False

#     for i in range(1, n):
#         if signal[i] != current_signal:
#             current_signal = signal[i]
#             if current_signal == 1:
#                 in_position = 1
#                 entry_price = prices[i]
#                 last_entry_price = entry_price
#                 last_was_long = True
#             else:
#                 in_position = 0
#                 last_was_long = False
#             continue

#         if in_position == 1:
#             total_pnl = (prices[i] - entry_price) / entry_price
#             if total_pnl >= long_tp:
#                 in_position = 0
#                 last_was_long = False

#     if last_entry_price is not None and last_was_long:
#         tp_price = last_entry_price * (1 + long_tp)
#         return round(last_entry_price, 2), round(tp_price, 2)
#     return None, None


# def calculate_metrics(returns):
#     if len(returns) == 0 or np.std(returns) == 0:
#         return 0, 0, 0, 0
#     ann_ret = np.mean(returns) * 252
#     ann_vol = np.std(returns) * np.sqrt(252)
#     sharpe = ann_ret / ann_vol if ann_vol != 0 else 0

#     cum_ret = np.cumprod(1 + returns)
#     peak = np.maximum.accumulate(cum_ret)
#     mdd = np.max((peak - cum_ret) / peak) if len(cum_ret) else 0
#     calmar = ann_ret / mdd if mdd != 0 else 0
#     return ann_ret, ann_vol, sharpe, calmar


# def get_rsi(series, period=14):
#     delta = series.diff()
#     gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
#     loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
#     rs = gain / loss
#     return 100 - (100 / (1 + rs))


# def optimize_long_tp(df, signal_col, tp_ranges, fee):
#     best_sharpe = -np.inf
#     best_tp = tp_ranges[0]
#     for l_tp in tp_ranges:
#         rets = backtest_long_only(df, signal_col, l_tp, fee)
#         _, _, sharpe, _ = calculate_metrics(rets)
#         if sharpe > best_sharpe:
#             best_sharpe = sharpe
#             best_tp = l_tp
#     return best_tp


# def optimize_short_tp(df, signal_col, tp_ranges, fee):
#     best_sharpe = -np.inf
#     best_tp = tp_ranges[0]
#     for s_tp in tp_ranges:
#         rets = backtest_short_only(df, signal_col, s_tp, fee)
#         _, _, sharpe, _ = calculate_metrics(rets)
#         if sharpe > best_sharpe:
#             best_sharpe = sharpe
#             best_tp = s_tp
#     return best_tp


# def trend_status(df, signal_col, long_tp, short_tp):
#     """
#     Live status for a given signal column: current side, how many days
#     the current streak has run, the entry level, unrealized P&L, and
#     the take-profit / limit-order target for closing the position.
#     """
#     signal = df[signal_col].values
#     prices = df["Close"].values
#     dates = df.index
#     n = len(prices)

#     current_sig = signal[-1]

#     entry_idx = n - 1
#     for i in range(n - 2, -1, -1):
#         if signal[i] != current_sig:
#             break
#         entry_idx = i

#     days_in_trend = n - entry_idx
#     entry_price = float(prices[entry_idx])
#     entry_date = dates[entry_idx].strftime("%Y-%m-%d")
#     current_price = float(prices[-1])

#     if current_sig == 1:
#         side = "LONG"
#         pnl_pct = (current_price - entry_price) / entry_price * 100
#         target_price = entry_price * (1 + long_tp)
#         target_label = "Limit order (take-profit) to CLOSE the long"
#     else:
#         side = "DOWN / FLAT (long-only strategy takes no position)"
#         pnl_pct = (entry_price - current_price) / entry_price * 100
#         target_price = entry_price * (1 - short_tp)
#         target_label = "Informational SHORT target (not traded)"

#     return {
#         "side": side,
#         "days_in_trend": days_in_trend,
#         "entry_price": round(entry_price, 2),
#         "entry_date": entry_date,
#         "current_price": round(current_price, 2),
#         "pnl_pct": round(pnl_pct, 2),
#         "target_price": round(target_price, 2),
#         "target_label": target_label,
#     }


# @st.cache_data(ttl=3600, show_spinner=False)
# def load_data(ticker, start, end):
#     df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
#     if df is None or df.empty:
#         return df
#     if isinstance(df.columns, pd.MultiIndex):
#         df.columns = df.columns.get_level_values(0)
#     return df


# def build_signals(df, ema_length):
#     df = df.copy()
#     df["HA_Close"] = (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4
#     ha_open = np.zeros(len(df))
#     ha_open[0] = (df["Open"].iloc[0] + df["Close"].iloc[0]) / 2
#     for i in range(1, len(df)):
#         ha_open[i] = (ha_open[i - 1] + df["HA_Close"].iloc[i - 1]) / 2
#     df["HA_Open"] = ha_open
#     df["HA_Trend"] = np.where(df["HA_Close"] >= df["HA_Open"], 1, -1)

#     df["EMA_Val"] = df["Close"].ewm(span=ema_length, adjust=False).mean()
#     df["EMA_Trend"] = np.where(df["Close"] >= df["EMA_Val"], 1, -1)

#     df["RSI"] = get_rsi(df["Close"])
#     return df


# # ============================================================
# #  UI
# # ============================================================

# st.title("📈 Trend-Following Strategy Backtester")
# st.caption(
#     "Heikin-Ashi vs EMA trend, long-only, optimal take-profit search vs Buy & Hold "
#     "— same logic as the original script, run across as many tickers as you like."
# )

# START_DATE = datetime.date(2026, 1, 1)
# END_DATE = datetime.date.today() + datetime.timedelta(days=1)

# with st.sidebar:
#     st.header("Settings")
#     tickers_input = st.text_input("Tickers (comma-separated)", value=DEFAULT_TICKERS)
#     st.caption(f"Backtest window: **{START_DATE}** → **{END_DATE}** (fixed, start of 2026 to latest close)")
#     fee = st.number_input(
#         "Fee per trade (fraction)", value=0.0020, step=0.0005, format="%.4f"
#     )
#     ema_length = st.number_input("EMA length", value=9, min_value=2, max_value=200, step=1)

#     st.subheader("Take-profit grid search")
#     tp_min = st.number_input("Min TP", value=0.025, step=0.005, format="%.3f")
#     tp_max = st.number_input("Max TP", value=0.15, step=0.005, format="%.3f")
#     tp_step = st.number_input("Step", value=0.005, step=0.001, format="%.3f")

#     run_button = st.button("Run Backtest", type="primary", use_container_width=True)

# if not run_button:
#     st.info("Enter one or more tickers in the sidebar (comma-separated) and click **Run Backtest**.")
#     st.stop()

# tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
# if not tickers:
#     st.warning("Please enter at least one ticker.")
#     st.stop()

# tp_ranges = np.arange(tp_min, tp_max + tp_step / 2, tp_step)
# tp_ranges = tp_ranges[tp_ranges > 0]

# for ticker in tickers:
#     st.divider()
#     st.subheader(f"📊 {ticker}")

#     with st.spinner(f"Downloading & backtesting {ticker}..."):
#         try:
#             df = load_data(ticker, START_DATE.strftime("%Y-%m-%d"), END_DATE.strftime("%Y-%m-%d"))
#         except Exception as e:
#             st.error(f"Failed to download {ticker}: {e}")
#             continue

#         if df is None or df.empty or len(df) < 30:
#             st.warning(f"No / insufficient price history for **{ticker}** in this date range.")
#             continue

#         try:
#             df = build_signals(df, ema_length)

#             best_tp_ha = optimize_long_tp(df, "HA_Trend", tp_ranges, fee)
#             best_tp_ema = optimize_long_tp(df, "EMA_Trend", tp_ranges, fee)
#             best_short_tp_ha = optimize_short_tp(df, "HA_Trend", tp_ranges, fee)
#             best_short_tp_ema = optimize_short_tp(df, "EMA_Trend", tp_ranges, fee)

#             ha_rets = backtest_long_only(df, "HA_Trend", best_tp_ha, fee)
#             ema_rets = backtest_long_only(df, "EMA_Trend", best_tp_ema, fee)

#             ha_ann_r, ha_ann_v, ha_sha, ha_cal = calculate_metrics(ha_rets)
#             ema_ann_r, ema_ann_v, ema_sha, ema_cal = calculate_metrics(ema_rets)

#             bh_rets = df["Close"].pct_change().fillna(0).values
#             bh_ann_r, bh_ann_v, bh_sha, bh_cal = calculate_metrics(bh_rets)

#             ha_trade_price, ha_tp_price = get_trade_entry_and_tp_long_only(df, "HA_Trend", best_tp_ha)
#             ema_trade_price, ema_tp_price = get_trade_entry_and_tp_long_only(df, "EMA_Trend", best_tp_ema)

#             cum_ha = np.cumprod(1 + ha_rets)
#             cum_ema = np.cumprod(1 + ema_rets)
#             cum_bh = np.cumprod(1 + bh_rets)

#             last = df.iloc[-1]
#         except Exception as e:
#             st.error(f"Error while analyzing {ticker}: {e}")
#             continue

#     ha_status = trend_status(df, "HA_Trend", best_tp_ha, best_short_tp_ha)
#     ema_status = trend_status(df, "EMA_Trend", best_tp_ema, best_short_tp_ema)

#     current_price = float(last["Close"])

#     top1, top2 = st.columns(2)
#     top1.metric("Current Price", f"{current_price:.2f}")
#     top2.metric("RSI(14)", f"{float(last['RSI']):.1f}" if not np.isnan(last["RSI"]) else "n/a")

#     st.markdown("#### 🎯 Current Action & Target Levels (based on optimal TP thresholds)")
#     a1, a2 = st.columns(2)
#     for col, name, status, l_tp, s_tp in [
#         (a1, "Heikin-Ashi", ha_status, best_tp_ha, best_short_tp_ha),
#         (a2, f"EMA({ema_length})", ema_status, best_tp_ema, best_short_tp_ema),
#     ]:
#         with col:
#             is_long = status["side"] == "LONG"
#             action_label = "🟢 LONG — holding" if is_long else "⚪ FLAT — no position (long-only strategy)"
#             st.markdown(f"**{name}**")
#             st.write(f"Action: **{action_label}**")
#             gap_pct = (status["target_price"] - current_price) / current_price * 100
#             g1, g2 = st.columns(2)
#             g1.metric("Current Price", f"{current_price:.2f}")
#             g2.metric(
#                 "Target Price" if is_long else "Informational Target",
#                 f"{status['target_price']:.2f}",
#                 delta=f"{gap_pct:+.2f}% away",
#             )
#             st.caption(
#                 f"Entry {status['entry_date']} @ {status['entry_price']} · "
#                 f"day {status['days_in_trend']} of trend · "
#                 f"unrealized P&L {status['pnl_pct']:+.2f}% · "
#                 f"optimal {'long' if is_long else 'short'} TP = "
#                 f"{(l_tp if is_long else s_tp)*100:.1f}%"
#             )

#     with st.expander("Full performance summary (vs Buy & Hold)"):
#         summary_data = [
#             {
#                 "Strategy": "Heikin-Ashi (long-only)",
#                 "Opt Long TP": f"{best_tp_ha*100:.1f}%",
#                 "Ann Ret": f"{ha_ann_r*100:.1f}%",
#                 "Ann Vol": f"{ha_ann_v*100:.1f}%",
#                 "Sharpe": round(ha_sha, 2),
#                 "Calmar": round(ha_cal, 2),
#                 "Trade Price": ha_trade_price,
#                 "TP Price": ha_tp_price,
#                 "TP Diff": None if ha_trade_price is None else round(ha_tp_price - ha_trade_price, 2),
#             },
#             {
#                 "Strategy": f"EMA({ema_length}) (long-only)",
#                 "Opt Long TP": f"{best_tp_ema*100:.1f}%",
#                 "Ann Ret": f"{ema_ann_r*100:.1f}%",
#                 "Ann Vol": f"{ema_ann_v*100:.1f}%",
#                 "Sharpe": round(ema_sha, 2),
#                 "Calmar": round(ema_cal, 2),
#                 "Trade Price": ema_trade_price,
#                 "TP Price": ema_tp_price,
#                 "TP Diff": None if ema_trade_price is None else round(ema_tp_price - ema_trade_price, 2),
#             },
#             {
#                 "Strategy": "Buy & Hold",
#                 "Opt Long TP": "—",
#                 "Ann Ret": f"{bh_ann_r*100:.1f}%",
#                 "Ann Vol": f"{bh_ann_v*100:.1f}%",
#                 "Sharpe": round(bh_sha, 2),
#                 "Calmar": round(bh_cal, 2),
#                 "Trade Price": round(float(df["Close"].iloc[0]), 2),
#                 "TP Price": "—",
#                 "TP Diff": "—",
#             },
#         ]
#         results_df = pd.DataFrame(summary_data)
#         st.dataframe(results_df, use_container_width=True, hide_index=True)

#     fig, ax = plt.subplots(figsize=(12, 5))
#     ax.plot(df.index, cum_ha, label=f"Heikin-Ashi (TP={best_tp_ha*100:.1f}%)", linewidth=1.8)
#     ax.plot(df.index, cum_ema, label=f"EMA({ema_length}) (TP={best_tp_ema*100:.1f}%)", linewidth=1.8)
#     ax.plot(df.index, cum_bh, label="Buy & Hold", linewidth=1.8, linestyle="--", color="gray")
#     ax.set_title(f"{ticker} — Long-Only Strategy Comparison: Heikin-Ashi vs EMA vs Buy & Hold",
#                  fontsize=13, fontweight="bold")
#     ax.set_xlabel("Date")
#     ax.set_ylabel("Cumulative Return (Growth of $1)")
#     ax.legend(loc="best")
#     ax.grid(True, alpha=0.3)
#     fig.tight_layout()
#     st.pyplot(fig)
#     plt.close(fig)
