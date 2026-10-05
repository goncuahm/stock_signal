# ============================================================
#  REGIME SIGNALS — LIVE APP (Streamlit, light version)
#  Backtest window: from BACKTEST_START (2026-01-01) to the latest close, for
#  every strategy (buy & hold, EMA rule, HA rule, Logit) — it grows daily.
#  Logit [EMA + HA], two ways (sidebar):
#   (1) default — trained IN THE APP: initial training on TRAIN_START
#       (2025-09-01) → BACKTEST_START, then retrained on the first trading day
#       of every month on an expanding window (labels recomputed from past
#       prices only, as in the research code)
#   (2) a saved model trained offline on long history (export_regime_model.py)
#  Prices are downloaded from TRAIN_START minus a short warm-up for the EMA
#  and Heikin-Ashi inputs (the warm-up is never used for training).
#    • P(up-leg) from 4 inputs: EMA signal, distance to EMA,
#      Heikin-Ashi colour, Heikin-Ashi body
#    • signal: UP if P ≥ 0.60, DOWN if P ≤ 0.40, else keep
#    • compared with the EMA rule, the HA rule and buy & hold
#    • trading: long-only, enter / exit at the close of a switch,
#      cash earns the deposit rate, costs per side, no take-profit
#  Run:  streamlit run regime_live_app.py
# ============================================================
import os
import json
import time
import datetime
import warnings

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

BACKTEST_START = "2026-01-01" # every strategy is traded and reported from this date on
TRAIN_START    = "2025-09-01" # in-app training data start (expanding window)
MIN_LABELLED   = 30           # minimum labelled days for a fit
LEAD_DAYS    = 120            # extra calendar days downloaded for the EMA / HA warm-up
MODELS_FILE  = "regime_models.json"
P_LONG, P_SHORT = 0.60, 0.40   # defaults; adjustable in the sidebar
BIST_CLOSE   = datetime.time(18, 15)   # after the closing auction (Istanbul time)
PRICE_TTL    = 300            # seconds prices are cached (5 minutes)

# peak / trough labels — research settings
SMOOTH_SIGMA, MIN_DISTANCE, PROMINENCE_FACTOR = 3, 10, 0.5
EDGE_DAYS = 3 * SMOOTH_SIGMA + MIN_DISTANCE
FEATURES = ['EMA_signal', 'Dist_EMA', 'HA_trend', 'HA_body']


# ============================================================
#  DATA, INPUTS, MODEL
# ============================================================
def download_prices(ticker, start, end, retries=3):
    for attempt in range(retries):
        try:
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        except Exception:
            df = None
        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df[['Open', 'High', 'Low', 'Close']].apply(pd.to_numeric, errors='coerce').dropna()
            df = df[df['Close'] > 0]
            if len(df):
                return df
        time.sleep(1.5 * (attempt + 1))
    return None


def cash_series(dates, const_annual=0.40):
    """Daily cash return: a constant effective annual rate over 252 trading days."""
    return np.full(len(dates), (1 + const_annual) ** (1 / 252) - 1)


def unfinished_session(df, ticker, now=None):
    """True if the last bar is today's BIST session that has not closed yet (its 'close' is the
    latest delayed intraday price and will change until the closing auction). Used for labelling."""
    if not ticker.endswith('.IS') or df is None or not len(df):
        return False
    now = now or pd.Timestamp.now(tz='Europe/Istanbul')
    return df.index[-1].date() == now.date() and now.time() < BIST_CLOSE


def build_features(df, ema_len):
    c = df['Close']
    ema = c.ewm(span=ema_len, adjust=False).mean()
    ha_close = (df['Open'] + df['High'] + df['Low'] + df['Close']).values / 4
    ha_open = np.zeros(len(df))
    ha_open[0] = (df['Open'].iloc[0] + df['Close'].iloc[0]) / 2
    for i in range(1, len(df)):
        ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2
    F = pd.DataFrame(index=df.index)
    F['EMA_signal'] = np.where(c >= ema, 1.0, -1.0)
    F['Dist_EMA'] = c / ema - 1
    F['HA_trend'] = np.where(ha_close >= ha_open, 1.0, -1.0)
    F['HA_body'] = (ha_close - ha_open) / c.values
    return F, {'ema': ema.values, 'ha_open': ha_open, 'ha_close': ha_close}


def predict_proba(spec, X):
    """Saved standardised logistic regression: P(up) = 1 / (1 + exp(-(z·coef + b)))."""
    z = (np.asarray(X, float) - np.array(spec['mean'])) / np.array(spec['scale'])
    return 1.0 / (1.0 + np.exp(-(z @ np.array(spec['coef']) + spec['intercept'])))


def smooth_pivots(c):
    lp = np.log(c)
    sm_ = gaussian_filter1d(lp, sigma=SMOOTH_SIGMA)
    thr = np.std(np.diff(lp)) * PROMINENCE_FACTOR
    pk, _ = find_peaks(sm_, distance=MIN_DISTANCE, prominence=thr)
    tr, _ = find_peaks(-sm_, distance=MIN_DISTANCE, prominence=thr)
    last_ok = len(c) - 1 - EDGE_DAYS
    piv = sorted([(i, 'P') for i in pk if i <= last_ok] + [(i, 'T') for i in tr if i <= last_ok])
    alt = []
    for i, tp in piv:
        if not alt or alt[-1][1] != tp:
            alt.append((i, tp))
    return alt


def labels_from(pivots, m):
    y = np.full(m, np.nan)
    for (i0, t0), (i1, _) in zip(pivots[:-1], pivots[1:]):
        y[i0:i1] = 1.0 if t0 == 'T' else 0.0
    return y


def fit_spec(X, y):
    """Standardised, class-balanced logistic regression (research settings) → parameter dict."""
    sc = StandardScaler().fit(X)
    lr = LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000).fit(sc.transform(X), y.astype(int))
    sign = 1.0 if list(lr.classes_).index(1) == 1 else -1.0
    return {'mean': sc.mean_.tolist(), 'scale': sc.scale_.tolist(), 'coef': (sign * lr.coef_[0]).tolist(),
            'intercept': float(sign * lr.intercept_[0]), 'features': FEATURES}


def walk_forward_in_app(close, X, dates, t0, a):
    """Expanding-window training on data from index t0 (TRAIN_START). Refit on the first trading
    day of every month from index a (BACKTEST_START) on; each model predicts until the next refit.
    Labels at each refit are computed from prices t0..R only."""
    n = len(close)
    prob = np.full(n, np.nan)
    month = dates[a:].to_period('M')
    refits = [a + int(i) for i in np.r_[0, np.where(month[1:] != month[:-1])[0] + 1]]
    spec, n_refits, n_lab, first = None, 0, 0, None
    for k, R in enumerate(refits):
        seg = close[t0:R + 1]
        y = labels_from(smooth_pivots(seg), len(seg))
        tr = np.where(np.isfinite(y) & np.isfinite(X[t0:R + 1]).all(axis=1))[0]
        if len(tr) >= MIN_LABELLED and len(np.unique(y[tr])) == 2:
            spec = fit_spec(X[t0 + tr], y[tr])
            n_refits, n_lab = n_refits + 1, len(tr)
        if spec is None:
            continue                                   # not enough labelled history yet → stay in cash
        b = refits[k + 1] if k + 1 < len(refits) else n
        prob[R:b] = predict_proba(spec, X[R:b])
        first = R if first is None else first
    if spec is not None:
        spec.update({'n_train': n_lab, 'n_refits': n_refits, 'first_prediction': str(dates[first].date())})
    return prob, spec


def regime_signal(p, p_long=P_LONG, p_short=P_SHORT):
    s, cur = np.zeros(len(p)), 0.0
    for t in range(len(p)):
        if np.isfinite(p[t]):
            if p[t] >= p_long:
                cur = 1.0
            elif p[t] <= p_short:
                cur = -1.0
        s[t] = cur
    return s


# ============================================================
#  TRADING (research engine, long-only, no take-profit)
# ============================================================
def long_only(close, signal, cash, fee, a):
    """Enter at the close of a switch to UP, exit at the close of the switch away from UP;
    position set at close t earns the return of t+1; cash earns the deposit rate."""
    n = len(close)
    rets, expo, ent = np.zeros(n), np.zeros(n), np.zeros(n)
    pos, cur = 0, 0.0
    for i in range(a, n):
        if pos:
            expo[i] = 1.0
            rets[i] += close[i] / close[i - 1] - 1.0
        else:
            rets[i] += cash[i]
        if signal[i] != cur:
            cur = signal[i]
            if pos:
                rets[i] -= fee
                pos = 0
            if cur > 0:
                pos = 1
                rets[i] -= fee
                ent[i] = 1.0
    return rets, expo, ent


def perf(r, expo, ent, cash, a, b):
    r, c = r[a:b], cash[a:b]
    cum = np.cumprod(1 + r)
    sd = r.std()
    yrs = (b - a) / 252
    return {'Return % p.a.': r.mean() * 252 * 100,
            'Volatility % p.a.': sd * np.sqrt(252) * 100,
            'Sharpe (xs cash)': (r - c).mean() / sd * np.sqrt(252) if sd > 1e-10 else np.nan,
            'Max drawdown %': np.max((np.maximum.accumulate(cum) - cum) / np.maximum.accumulate(cum)) * 100,
            'Total return %': (cum[-1] - 1) * 100,
            'In market %': expo[a:b].mean() * 100,
            'Trades / yr': ent[a:b].sum() / yrs if yrs > 0 else np.nan}


def analyse(df, spec, ema_len, fee, cash, train_in_app=True, p_long=P_LONG, p_short=P_SHORT):
    close = df['Close'].values.astype(float)
    n = len(close)
    F, aux = build_features(df, ema_len)
    sigs = {}
    prob = None
    a = int(np.searchsorted(df.index, pd.Timestamp(BACKTEST_START)))
    if train_in_app:
        t0 = int(np.searchsorted(df.index, pd.Timestamp(TRAIN_START)))
        prob, spec = walk_forward_in_app(close, F[FEATURES].values, df.index, t0, a)
        if spec is None:
            return None                              # not enough labelled data yet
        spec.update({'train_start': str(df.index[t0].date()), 'cutoff': str(df.index[-1].date()),
                     'ema_len': ema_len})
        sigs['Logit [EMA + HA]'] = regime_signal(prob, p_long, p_short)
    elif spec is not None:
        prob = predict_proba(spec, F[spec['features']].values)
        prob[:a] = np.nan                            # the saved model is applied from BACKTEST_START on
        sigs['Logit [EMA + HA]'] = regime_signal(prob, p_long, p_short)
    sigs[f'EMA({ema_len}) rule'] = F['EMA_signal'].values
    sigs['HA rule'] = F['HA_trend'].values
    strats = {nm: long_only(close, sg, cash, fee, a) for nm, sg in sigs.items()}
    bh = np.zeros(n)
    bh[a + 1:] = close[a + 1:] / close[a:-1] - 1
    bh[a] = cash[a] - fee
    strats['Buy & hold'] = (bh, np.r_[np.zeros(a + 1), np.ones(n - a - 1)], np.r_[np.zeros(a), 1.0, np.zeros(n - a - 1)])
    return {'a': a, 'F': F, 'aux': aux, 'prob': prob, 'sigs': sigs, 'strats': strats, 'spec': spec,
            'p_long': p_long, 'p_short': p_short}


def position_status(df, sig, a):
    c = df['Close'].values
    n = len(c)
    entries = [t for t in range(max(a, 1), n) if sig[t] == 1 and (t == a or sig[t - 1] != 1)]
    exits = [t for t in range(max(a, 1), n) if sig[t] != 1 and sig[t - 1] == 1]
    if sig[-1] != 1 or not entries:
        return {'state': 'CASH', 'since': df.index[exits[-1]].date() if exits else None}
    e = entries[-1]
    return {'state': 'LONG', 'since': df.index[e].date(), 'entry': c[e], 'pnl': (c[-1] / c[e] - 1) * 100}


def live_preview(res, df, live_price, ema_len):
    """Provisional signals if today's session closed at the live price (not used in the backtest)."""
    if live_price is None:
        return None
    aux = res['aux']
    o, c = float(df['Close'].iloc[-1]), float(live_price)
    h, l = max(o, c), min(o, c)
    alpha = 2 / (ema_len + 1)
    ema_new = c * alpha + aux['ema'][-1] * (1 - alpha)
    ha_close = (o + h + l + c) / 4
    ha_open = (aux['ha_open'][-1] + aux['ha_close'][-1]) / 2
    x = {'EMA_signal': 1.0 if c >= ema_new else -1.0, 'Dist_EMA': c / ema_new - 1,
         'HA_trend': 1.0 if ha_close >= ha_open else -1.0, 'HA_body': (ha_close - ha_open) / c}
    spec = res['spec']
    out = {'ema': x['EMA_signal'], 'ha': x['HA_trend'], 'p': np.nan, 'logit': None}
    if spec is not None and 'Logit [EMA + HA]' in res['sigs']:
        p = float(predict_proba(spec, [[x[f] for f in spec['features']]])[0])
        cur = res['sigs']['Logit [EMA + HA]'][-1]
        out['p'], out['logit'] = p, (1.0 if p >= res['p_long'] else (-1.0 if p <= res['p_short'] else cur))
    return out


# ============================================================
#  STREAMLIT UI
# ============================================================
def main():
    import streamlit as st
    ver = tuple(int(x) for x in st.__version__.split('.')[:2])
    WIDE = {'width': 'stretch'} if ver >= (1, 46) else {'use_container_width': True}

    st.set_page_config(page_title="Regime Signals — Live", layout="wide")
    st.title("📈 Regime signals — live")
    st.caption(f"Backtest from {BACKTEST_START} to the latest close for every strategy. Logit [EMA + HA] on "
               f"peak/trough regimes: trained in the app on data from {TRAIN_START}, retrained every month on an "
               "expanding window (or loaded from a saved long-history model). Long-only.")

    @st.cache_data(ttl=PRICE_TTL, show_spinner=False)
    def cached_prices(ticker, start, end):
        return download_prices(ticker, start, end), pd.Timestamp.now(tz='Europe/Istanbul')

    @st.cache_data(ttl=120, show_spinner=False)
    def live_quote(ticker):
        try:
            fi = yf.Ticker(ticker).fast_info
            p = fi.get('last_price') if isinstance(fi, dict) else getattr(fi, 'last_price', None)
            return float(p) if p else None
        except Exception:
            return None

    with st.sidebar:
        st.header("Settings")
        tickers_in = st.text_input("Tickers (comma-separated)", "XU030.IS")
        mode = st.radio("Logit model", [f"Train in the app (data from {TRAIN_START}, monthly refits)",
                                        "Saved model (Colab file)"])
        train_in_app = mode.startswith("Train")
        ema_in = st.number_input("EMA length (in-app training)", 2, 200, 10, 1, disabled=not train_in_app)
        up_models = None
        if not train_in_app:
            up_models = st.file_uploader(f"Model file ({MODELS_FILE})", type=['json'],
                                         help="Created by export_regime_model.py in Colab. If omitted, the app "
                                              f"looks for {MODELS_FILE} next to the app.")
        st.subheader("Signal thresholds")
        p_long = st.slider("Go long when P(up) ≥", 0.50, 0.90, P_LONG, 0.01)
        p_short = st.slider("Go to cash when P(up) ≤", 0.10, 0.50, P_SHORT, 0.01,
                            help="Between the two thresholds the previous position is kept.")
        st.subheader("Costs and cash")
        fee = st.number_input("Cost per side", value=0.0010, step=0.0005, format="%.4f")
        const_rate = st.number_input("Cash rate % p.a.", value=40.0, step=1.0) / 100
        window = st.selectbox("Performance window", [f"Since {BACKTEST_START}", "Last 6 months",
                                                     "Last 3 months"])
        run = st.button("Run", type="primary", **WIDE)
        if st.button("🔄 Refresh prices", **WIDE, help=f"Prices are cached for {PRICE_TTL // 60} minutes; "
                                                       "this downloads them again now."):
            st.cache_data.clear()
            run = True

    models = {}
    try:
        if train_in_app:
            raise StopIteration
        if up_models is not None:
            models = json.load(up_models)
        elif os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), MODELS_FILE)):
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), MODELS_FILE)) as f:
                models = json.load(f)
    except StopIteration:
        pass
    except Exception as e:
        st.sidebar.error(f"Could not read the model file: {e}")
    if not train_in_app:
        if models:
            st.sidebar.success("Models: " + ", ".join(f"{k} (cutoff {v['cutoff']})" for k, v in models.items()))
        else:
            st.sidebar.warning("No model file — only the EMA and HA rules will be shown.")

    if not run:
        st.info("Choose tickers in the sidebar and click **Run**.")
        st.stop()

    if p_short > p_long:
        st.error("The cash threshold must not be above the long threshold.")
        st.stop()

    first_day = BACKTEST_START
    start = (pd.Timestamp(TRAIN_START) - pd.Timedelta(days=LEAD_DAYS)).strftime('%Y-%m-%d')
    end = (datetime.date.today() + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
    for k, ticker in enumerate([t.strip().upper() for t in tickers_in.split(',') if t.strip()]):
        if k:
            time.sleep(0.8)
        st.divider()
        st.subheader(f"📊 {ticker}")
        df, fetched = cached_prices(ticker, start, end)    # every bar Yahoo returns is used
        intraday_today = unfinished_session(df, ticker)
        if df is None or (df.index >= pd.Timestamp(first_day)).sum() < 20:
            st.warning(f"No / insufficient price data for **{ticker}**.")
            continue
        cash = cash_series(df.index, const_rate)
        dates_first = df.index[int(np.searchsorted(df.index, pd.Timestamp(BACKTEST_START)))]
        if train_in_app:
            ema_len = int(ema_in)
            res = analyse(df, None, ema_len, float(fee), cash, train_in_app=True, p_long=p_long, p_short=p_short)
            if res is None:
                st.warning(f"Not enough data since {TRAIN_START} to train on **{ticker}** "
                           "(needs at least one complete up- and down-leg).")
                continue
            sp = res['spec']
            late = pd.Timestamp(sp['first_prediction']) > dates_first
            st.caption(f"ℹ️ Logit trained in the app on {sp['train_start']} → {sp['cutoff']}, retrained monthly "
                       f"({sp['n_refits']} fits; {sp['n_train']} labelled days in the latest). Predictions are out of "
                       f"sample from {sp['first_prediction']}"
                       + (" — in cash before that, while the training sample was too short." if late else ".")
                       + " A small sample: noisier than the long-history research model.")
        else:
            spec = models.get(ticker)
            ema_len = int(spec['ema_len']) if spec else 10
            if spec is None:
                st.caption(f"No saved model for {ticker} — showing the rules only "
                           f"(add {ticker} to TICKERS in export_regime_model.py).")
            elif pd.Timestamp(spec['cutoff']) >= pd.Timestamp(BACKTEST_START):
                st.caption(f"ℹ️ Model trained up to {spec['cutoff']}: results before that date are in-sample.")
            res = analyse(df, spec, ema_len, float(fee), cash, train_in_app=False, p_long=p_long, p_short=p_short)
        spec = res['spec']
        a0, n, dates = res['a'], len(df), df.index

        # ---------- as-of stamp ----------
        last_px = float(df['Close'].iloc[-1])
        st.info(f"🕒 **All results below use every price Yahoo Finance returned, up to {dates[-1].date()} — "
                f"latest price {last_px:,.2f}** (downloaded {fetched.strftime('%d %b %Y %H:%M')} Istanbul time). "
                + ("This is **today's session in progress** (delayed intraday price): today's signals and returns "
                   "will change until the closing auction." if intraday_today else
                   ("For markets still trading, the last bar may be today's unfinished session."
                    if not ticker.endswith('.IS') else "This is a completed session's closing price.")))

        # ---------- current signals ----------
        last_close = float(df['Close'].iloc[-1])
        live = live_quote(ticker)
        c1, c2, c3 = st.columns(3)
        c1.metric(f"Latest price used ({dates[-1].date()}{', intraday' if intraday_today else ', close'})",
                  f"{last_close:,.2f}")
        c2.metric("Live quote", f"{live:,.2f}" if live else "n/a",
                  delta=f"{(live / last_close - 1) * 100:+.2f}%" if live else None)
        p_now = res['prob'][-1] if res['prob'] is not None else np.nan
        c3.metric("Logit P(up-leg)", f"{p_now:.2f}" if np.isfinite(p_now) else "n/a",
                  help=f"Long if ≥ {p_long:.2f}, cash if ≤ {p_short:.2f}, otherwise the previous position is kept.")

        st.markdown("#### 🎯 Current positions (long-only, after the last settled close)")
        cols = st.columns(len(res['sigs']))
        for col, (name, sg) in zip(cols, res['sigs'].items()):
            stt = position_status(df, sg, a0)
            with col:
                st.markdown(f"**{name}** — {'🟢 LONG' if stt['state'] == 'LONG' else '⚪ CASH'}")
                if stt['state'] == 'LONG':
                    st.caption(f"Bought {stt['since']} at {stt['entry']:,.2f} · P&L since entry {stt['pnl']:+.2f}%")
                else:
                    st.caption(f"In cash since {stt['since']}" if stt['since'] else "In cash")

        today_ist = pd.Timestamp.now(tz='Europe/Istanbul').date()
        pv = live_preview(res, df, live, ema_len) if dates[-1].date() < today_ist else None   # today not yet in data
        if pv:
            with st.expander("🔮 Provisional — signals if today closed at the live quote (not part of the backtest)"):
                side = lambda v: "🟢 LONG" if v == 1 else "⚪ CASH"
                txt = f"**EMA({ema_len}):** {side(pv['ema'])} · **HA:** {side(pv['ha'])}"
                if pv['logit'] is not None:
                    txt = f"**Logit [EMA + HA]:** P(up) {pv['p']:.2f} → {side(pv['logit'])} · " + txt
                st.write(txt)
                st.caption("Signals become final only at the official close; intraday they can flip back.")

        # ---------- performance ----------
        starts = {f"Since {BACKTEST_START}": dates[a0],
                  "Last 6 months": dates[-1] - pd.DateOffset(months=6),
                  "Last 3 months": dates[-1] - pd.DateOffset(months=3)}
        a = max(a0, int(np.searchsorted(dates, starts[window])))
        st.markdown(f"#### Performance — {window.lower()} ({dates[a].date()} → {dates[-1].date()})")
        rows = {nm: perf(*v, cash, a, n) for nm, v in res['strats'].items()}
        rows['Buy & hold']['Trades / yr'] = np.nan
        st.dataframe(pd.DataFrame(rows).T.style.format("{:,.2f}", na_rep="—"), **WIDE)
        st.caption(f"Net of {fee * 100:.2f}% per side; cash earns "
                   + f"{const_rate * 100:.0f}% p.a."
                   + "; Sharpe in excess of cash. Short windows are dominated by noise — "
                     "the long-run evidence is in the presentation.")

        # ---------- charts ----------
        fig, axes = plt.subplots(3 if res['prob'] is not None else 2, 1, figsize=(13, 10),
                                 gridspec_kw={'height_ratios': [3, 2, 1.3][:3 if res['prob'] is not None else 2]})
        colors = {'Logit [EMA + HA]': 'tab:red', f'EMA({ema_len}) rule': 'tab:blue', 'HA rule': 'tab:orange',
                  'Buy & hold': 'grey'}
        for nm, v in res['strats'].items():
            axes[0].plot(dates[a:], np.cumprod(1 + v[0][a:]), color=colors[nm], lw=2.2 if 'Logit' in nm else 1.4,
                         ls='--' if nm == 'Buy & hold' else '-', label=nm)
        axes[0].set_title(f"{ticker} — growth of 1 ({window.lower()}), using prices up to {dates[-1].date()} "
                          f"(latest price {last_px:,.2f}{', intraday' if intraday_today else ''})", fontweight='bold')
        for nm, v in res['strats'].items():
            g = np.prod(1 + v[0][a:])
            axes[0].annotate(f"{g:.3f}", (dates[-1], g), textcoords='offset points', xytext=(4, 0),
                             va='center', fontsize=8, color=colors[nm])
        axes[0].legend(loc='upper left', fontsize=9)
        axes[0].grid(alpha=0.3)

        key = 'Logit [EMA + HA]' if 'Logit [EMA + HA]' in res['sigs'] else f'EMA({ema_len}) rule'
        sg = res['sigs'][key]
        z = max(a0, n - 126)
        px = df['Close'].values
        axes[1].plot(dates[z:], px[z:], color='black', lw=1)
        long_ = (sg[z:] == 1).astype(int)
        cuts = np.r_[0, np.where(np.diff(long_) != 0)[0] + 1, len(long_)]
        for i0, i1 in zip(cuts[:-1], cuts[1:]):       # one block per regime: from its first day to the next switch
            x1 = dates[z + i1] if z + i1 < n else dates[-1]
            axes[1].axvspan(dates[z + i0], x1, color='green' if long_[i0] else 'red',
                            alpha=0.15 if long_[i0] else 0.10, lw=0)
        bots = [t for t in range(z + 1, n) if sg[t - 1] != 1 and sg[t] == 1]
        pks = [t for t in range(z + 1, n) if sg[t - 1] == 1 and sg[t] != 1]
        axes[1].scatter(dates[bots], px[bots], marker='^', color='darkgreen', s=60, zorder=5)
        axes[1].scatter(dates[pks], px[pks], marker='v', color='darkred', s=60, zorder=5)
        axes[1].scatter([dates[-1]], [px[-1]], color='black', s=35, zorder=6)
        axes[1].annotate(f"latest {px[-1]:,.2f}\n{dates[-1].strftime('%d %b %Y')}"
                         + (" (intraday)" if intraday_today else ""), (dates[-1], px[-1]),
                         textcoords='offset points', xytext=(-10, 10), ha='right', fontsize=9,
                         bbox=dict(facecolor='white', edgecolor='grey', alpha=0.85, pad=2))
        axes[1].set_title(f"{key} — last 6 months to {dates[-1].date()} (green = long, red = cash; ▲ buy, ▼ sell)",
                          fontweight='bold')
        axes[1].grid(alpha=0.3)
        if res['prob'] is not None:
            axes[2].plot(dates[z:], res['prob'][z:], color='tab:purple')
            axes[2].axhline(p_long, color='green', ls='--', lw=1)
            axes[2].axhline(p_short, color='red', ls='--', lw=1)
            axes[2].scatter([dates[-1]], [res['prob'][-1]], color='tab:purple', s=40, zorder=5)
            axes[2].annotate(f"{res['prob'][-1]:.2f} ({dates[-1].strftime('%d %b')})", (dates[-1], res['prob'][-1]),
                             textcoords='offset points', xytext=(-8, 8), ha='right', fontsize=9, color='tab:purple')
            axes[2].set_ylim(0, 1)
            axes[2].set_ylabel('P(up)')
            axes[2].grid(alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        if spec is not None:
            with st.expander("Model details"):
                st.write(f"Trained {spec['train_start']} → {spec['cutoff']} on {spec['n_train']:,} labelled days "
                         f"(latest fit); EMA length {spec['ema_len']}.")
                st.dataframe(pd.DataFrame({'standardised coefficient': spec['coef']}, index=spec['features'])
                             .style.format("{:+.3f}"))


if __name__ == '__main__' and not os.environ.get('REGIME_APP_TEST'):
    main()



# # ============================================================
# #  REGIME SIGNALS — LIVE APP (Streamlit, light version)
# #  Backtest window: from BACKTEST_START (2026-01-01) to the latest close, for
# #  every strategy (buy & hold, EMA rule, HA rule, Logit) — it grows daily.
# #  Logit [EMA + HA], two ways (sidebar):
# #   (1) default — trained IN THE APP: initial training on TRAIN_START
# #       (2025-09-01) → BACKTEST_START, then retrained on the first trading day
# #       of every month on an expanding window (labels recomputed from past
# #       prices only, as in the research code)
# #   (2) a saved model trained offline on long history (export_regime_model.py)
# #  Prices are downloaded from TRAIN_START minus a short warm-up for the EMA
# #  and Heikin-Ashi inputs (the warm-up is never used for training).
# #    • P(up-leg) from 4 inputs: EMA signal, distance to EMA,
# #      Heikin-Ashi colour, Heikin-Ashi body
# #    • signal: UP if P ≥ 0.60, DOWN if P ≤ 0.40, else keep
# #    • compared with the EMA rule, the HA rule and buy & hold
# #    • trading: long-only, enter / exit at the close of a switch,
# #      cash earns the deposit rate, costs per side, no take-profit
# #  Run:  streamlit run regime_live_app.py
# # ============================================================
# import os
# import json
# import time
# import datetime
# import warnings

# import numpy as np
# import pandas as pd
# import yfinance as yf
# import matplotlib.pyplot as plt
# from scipy.signal import find_peaks
# from scipy.ndimage import gaussian_filter1d
# from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import StandardScaler

# warnings.filterwarnings("ignore")

# BACKTEST_START = "2026-01-01" # every strategy is traded and reported from this date on
# TRAIN_START    = "2025-09-01" # in-app training data start (expanding window)
# MIN_LABELLED   = 30           # minimum labelled days for a fit
# LEAD_DAYS    = 120            # extra calendar days downloaded for the EMA / HA warm-up
# MODELS_FILE  = "regime_models.json"
# P_LONG, P_SHORT = 0.60, 0.40

# # peak / trough labels — research settings
# SMOOTH_SIGMA, MIN_DISTANCE, PROMINENCE_FACTOR = 3, 10, 0.5
# EDGE_DAYS = 3 * SMOOTH_SIGMA + MIN_DISTANCE
# FEATURES = ['EMA_signal', 'Dist_EMA', 'HA_trend', 'HA_body']


# # ============================================================
# #  DATA, INPUTS, MODEL
# # ============================================================
# def download_prices(ticker, start, end, retries=3):
#     for attempt in range(retries):
#         try:
#             df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
#         except Exception:
#             df = None
#         if df is not None and not df.empty:
#             if isinstance(df.columns, pd.MultiIndex):
#                 df.columns = df.columns.get_level_values(0)
            df = df[['Open', 'High', 'Low', 'Close']].apply(pd.to_numeric, errors='coerce').dropna()
            df = df[df['Close'] > 0]
            if len(df):
                return df
        time.sleep(1.5 * (attempt + 1))
    return None


def parse_evds(file_obj, column='TP_TRY_MT01'):
    """Weekly EVDS deposit rates (% p.a.) → Series indexed by date (note rows dropped)."""
    raw = pd.read_excel(file_obj)
    rate_col = column if column in raw.columns else raw.columns[1]
    d = pd.to_datetime(raw.iloc[:, 0].astype(str), format='%d-%m-%Y', errors='coerce')
    if d.notna().sum() == 0:
        d = pd.to_datetime(raw.iloc[:, 0], errors='coerce', dayfirst=True)
    rates = pd.Series(pd.to_numeric(raw[rate_col], errors='coerce').values, index=d)
    rates = rates[rates.index.notna()].dropna().sort_index()
    return rates[~rates.index.duplicated(keep='last')]


def cash_series(dates, rates=None, const_annual=0.40):
    """Daily cash return. EVDS: rate known at the close of t-1, rolled 1-month deposits accrued
    over calendar days (as in the research code). Otherwise a constant rate over 252 days."""
    if rates is None or len(rates) == 0:
        return np.full(len(dates), (1 + const_annual) ** (1 / 252) - 1)
    ann = rates.reindex(rates.index.union(dates)).ffill().reindex(dates).bfill().values / 100
    r_prev = np.r_[ann[0], ann[:-1]]
    gap = np.r_[1, np.diff(dates.values).astype('timedelta64[D]').astype(int)]
    return (1 + r_prev / 12) ** (12 * gap / 365) - 1


def build_features(df, ema_len):
    c = df['Close']
    ema = c.ewm(span=ema_len, adjust=False).mean()
    ha_close = (df['Open'] + df['High'] + df['Low'] + df['Close']).values / 4
    ha_open = np.zeros(len(df))
    ha_open[0] = (df['Open'].iloc[0] + df['Close'].iloc[0]) / 2
    for i in range(1, len(df)):
        ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2
    F = pd.DataFrame(index=df.index)
    F['EMA_signal'] = np.where(c >= ema, 1.0, -1.0)
    F['Dist_EMA'] = c / ema - 1
    F['HA_trend'] = np.where(ha_close >= ha_open, 1.0, -1.0)
    F['HA_body'] = (ha_close - ha_open) / c.values
    return F, {'ema': ema.values, 'ha_open': ha_open, 'ha_close': ha_close}


def predict_proba(spec, X):
    """Saved standardised logistic regression: P(up) = 1 / (1 + exp(-(z·coef + b)))."""
    z = (np.asarray(X, float) - np.array(spec['mean'])) / np.array(spec['scale'])
    return 1.0 / (1.0 + np.exp(-(z @ np.array(spec['coef']) + spec['intercept'])))


def smooth_pivots(c):
    lp = np.log(c)
    sm_ = gaussian_filter1d(lp, sigma=SMOOTH_SIGMA)
    thr = np.std(np.diff(lp)) * PROMINENCE_FACTOR
    pk, _ = find_peaks(sm_, distance=MIN_DISTANCE, prominence=thr)
    tr, _ = find_peaks(-sm_, distance=MIN_DISTANCE, prominence=thr)
    last_ok = len(c) - 1 - EDGE_DAYS
    piv = sorted([(i, 'P') for i in pk if i <= last_ok] + [(i, 'T') for i in tr if i <= last_ok])
    alt = []
    for i, tp in piv:
        if not alt or alt[-1][1] != tp:
            alt.append((i, tp))
    return alt


def labels_from(pivots, m):
    y = np.full(m, np.nan)
    for (i0, t0), (i1, _) in zip(pivots[:-1], pivots[1:]):
        y[i0:i1] = 1.0 if t0 == 'T' else 0.0
    return y


def fit_spec(X, y):
    """Standardised, class-balanced logistic regression (research settings) → parameter dict."""
    sc = StandardScaler().fit(X)
    lr = LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000).fit(sc.transform(X), y.astype(int))
    sign = 1.0 if list(lr.classes_).index(1) == 1 else -1.0
    return {'mean': sc.mean_.tolist(), 'scale': sc.scale_.tolist(), 'coef': (sign * lr.coef_[0]).tolist(),
            'intercept': float(sign * lr.intercept_[0]), 'features': FEATURES}


def walk_forward_in_app(close, X, dates, t0, a):
    """Expanding-window training on data from index t0 (TRAIN_START). Refit on the first trading
    day of every month from index a (BACKTEST_START) on; each model predicts until the next refit.
    Labels at each refit are computed from prices t0..R only."""
    n = len(close)
    prob = np.full(n, np.nan)
    month = dates[a:].to_period('M')
    refits = [a + int(i) for i in np.r_[0, np.where(month[1:] != month[:-1])[0] + 1]]
    spec, n_refits, n_lab, first = None, 0, 0, None
    for k, R in enumerate(refits):
        seg = close[t0:R + 1]
        y = labels_from(smooth_pivots(seg), len(seg))
        tr = np.where(np.isfinite(y) & np.isfinite(X[t0:R + 1]).all(axis=1))[0]
        if len(tr) >= MIN_LABELLED and len(np.unique(y[tr])) == 2:
            spec = fit_spec(X[t0 + tr], y[tr])
            n_refits, n_lab = n_refits + 1, len(tr)
        if spec is None:
            continue                                   # not enough labelled history yet → stay in cash
        b = refits[k + 1] if k + 1 < len(refits) else n
        prob[R:b] = predict_proba(spec, X[R:b])
        first = R if first is None else first
    if spec is not None:
        spec.update({'n_train': n_lab, 'n_refits': n_refits, 'first_prediction': str(dates[first].date())})
    return prob, spec


def regime_signal(p):
    s, cur = np.zeros(len(p)), 0.0
    for t in range(len(p)):
        if np.isfinite(p[t]):
            if p[t] >= P_LONG:
                cur = 1.0
            elif p[t] <= P_SHORT:
                cur = -1.0
        s[t] = cur
    return s


# ============================================================
#  TRADING (research engine, long-only, no take-profit)
# ============================================================
def long_only(close, signal, cash, fee, a):
    """Enter at the close of a switch to UP, exit at the close of the switch away from UP;
    position set at close t earns the return of t+1; cash earns the deposit rate."""
    n = len(close)
    rets, expo, ent = np.zeros(n), np.zeros(n), np.zeros(n)
    pos, cur = 0, 0.0
    for i in range(a, n):
        if pos:
            expo[i] = 1.0
            rets[i] += close[i] / close[i - 1] - 1.0
        else:
            rets[i] += cash[i]
        if signal[i] != cur:
            cur = signal[i]
            if pos:
                rets[i] -= fee
                pos = 0
            if cur > 0:
                pos = 1
                rets[i] -= fee
                ent[i] = 1.0
    return rets, expo, ent


def perf(r, expo, ent, cash, a, b):
    r, c = r[a:b], cash[a:b]
    cum = np.cumprod(1 + r)
    sd = r.std()
    yrs = (b - a) / 252
    return {'Return % p.a.': r.mean() * 252 * 100,
            'Volatility % p.a.': sd * np.sqrt(252) * 100,
            'Sharpe (xs cash)': (r - c).mean() / sd * np.sqrt(252) if sd > 1e-10 else np.nan,
            'Max drawdown %': np.max((np.maximum.accumulate(cum) - cum) / np.maximum.accumulate(cum)) * 100,
            'Total return %': (cum[-1] - 1) * 100,
            'In market %': expo[a:b].mean() * 100,
            'Trades / yr': ent[a:b].sum() / yrs if yrs > 0 else np.nan}


def analyse(df, spec, ema_len, fee, cash, train_in_app=True):
    close = df['Close'].values.astype(float)
    n = len(close)
    F, aux = build_features(df, ema_len)
    sigs = {}
    prob = None
    a = int(np.searchsorted(df.index, pd.Timestamp(BACKTEST_START)))
    if train_in_app:
        t0 = int(np.searchsorted(df.index, pd.Timestamp(TRAIN_START)))
        prob, spec = walk_forward_in_app(close, F[FEATURES].values, df.index, t0, a)
        if spec is None:
            return None                              # not enough labelled data yet
        spec.update({'train_start': str(df.index[t0].date()), 'cutoff': str(df.index[-1].date()),
                     'ema_len': ema_len})
        sigs['Logit [EMA + HA]'] = regime_signal(prob)
    elif spec is not None:
        prob = predict_proba(spec, F[spec['features']].values)
        prob[:a] = np.nan                            # the saved model is applied from BACKTEST_START on
        sigs['Logit [EMA + HA]'] = regime_signal(prob)
    sigs[f'EMA({ema_len}) rule'] = F['EMA_signal'].values
    sigs['HA rule'] = F['HA_trend'].values
    strats = {nm: long_only(close, sg, cash, fee, a) for nm, sg in sigs.items()}
    bh = np.zeros(n)
    bh[a + 1:] = close[a + 1:] / close[a:-1] - 1
    bh[a] = cash[a] - fee
    strats['Buy & hold'] = (bh, np.r_[np.zeros(a + 1), np.ones(n - a - 1)], np.r_[np.zeros(a), 1.0, np.zeros(n - a - 1)])
    return {'a': a, 'F': F, 'aux': aux, 'prob': prob, 'sigs': sigs, 'strats': strats, 'spec': spec}


def position_status(df, sig, a):
    c = df['Close'].values
    n = len(c)
    entries = [t for t in range(max(a, 1), n) if sig[t] == 1 and (t == a or sig[t - 1] != 1)]
    exits = [t for t in range(max(a, 1), n) if sig[t] != 1 and sig[t - 1] == 1]
    if sig[-1] != 1 or not entries:
        return {'state': 'CASH', 'since': df.index[exits[-1]].date() if exits else None}
    e = entries[-1]
    return {'state': 'LONG', 'since': df.index[e].date(), 'entry': c[e], 'pnl': (c[-1] / c[e] - 1) * 100}


def live_preview(res, df, live_price, ema_len):
    """Provisional signals if today's session closed at the live price (not used in the backtest)."""
    if live_price is None:
        return None
    aux = res['aux']
    o, c = float(df['Close'].iloc[-1]), float(live_price)
    h, l = max(o, c), min(o, c)
    alpha = 2 / (ema_len + 1)
    ema_new = c * alpha + aux['ema'][-1] * (1 - alpha)
    ha_close = (o + h + l + c) / 4
    ha_open = (aux['ha_open'][-1] + aux['ha_close'][-1]) / 2
    x = {'EMA_signal': 1.0 if c >= ema_new else -1.0, 'Dist_EMA': c / ema_new - 1,
         'HA_trend': 1.0 if ha_close >= ha_open else -1.0, 'HA_body': (ha_close - ha_open) / c}
    spec = res['spec']
    out = {'ema': x['EMA_signal'], 'ha': x['HA_trend'], 'p': np.nan, 'logit': None}
    if spec is not None and 'Logit [EMA + HA]' in res['sigs']:
        p = float(predict_proba(spec, [[x[f] for f in spec['features']]])[0])
        cur = res['sigs']['Logit [EMA + HA]'][-1]
        out['p'], out['logit'] = p, (1.0 if p >= P_LONG else (-1.0 if p <= P_SHORT else cur))
    return out


# ============================================================
#  STREAMLIT UI
# ============================================================
def main():
    import streamlit as st
    ver = tuple(int(x) for x in st.__version__.split('.')[:2])
    WIDE = {'width': 'stretch'} if ver >= (1, 46) else {'use_container_width': True}

    st.set_page_config(page_title="Regime Signals — Live", layout="wide")
    st.title("📈 Regime signals — live")
    st.caption(f"Backtest from {BACKTEST_START} to the latest close for every strategy. Logit [EMA + HA] on "
               f"peak/trough regimes: trained in the app on data from {TRAIN_START}, retrained every month on an "
               "expanding window (or loaded from a saved long-history model). Long-only.")

    @st.cache_data(ttl=3600, show_spinner=False)
    def cached_prices(ticker, start, end):
        return download_prices(ticker, start, end)

    @st.cache_data(ttl=120, show_spinner=False)
    def live_quote(ticker):
        try:
            fi = yf.Ticker(ticker).fast_info
            p = fi.get('last_price') if isinstance(fi, dict) else getattr(fi, 'last_price', None)
            return float(p) if p else None
        except Exception:
            return None

    with st.sidebar:
        st.header("Settings")
        tickers_in = st.text_input("Tickers (comma-separated)", "XU030.IS")
        mode = st.radio("Logit model", [f"Train in the app (data from {TRAIN_START}, monthly refits)",
                                        "Saved model (Colab file)"])
        train_in_app = mode.startswith("Train")
        ema_in = st.number_input("EMA length (in-app training)", 2, 200, 10, 1, disabled=not train_in_app)
        up_models = None
        if not train_in_app:
            up_models = st.file_uploader(f"Model file ({MODELS_FILE})", type=['json'],
                                         help="Created by export_regime_model.py in Colab. If omitted, the app "
                                              f"looks for {MODELS_FILE} next to the app.")
        fee = st.number_input("Cost per side", value=0.0010, step=0.0005, format="%.4f")
        st.subheader("Cash rate")
        evds = st.file_uploader("EVDS deposit-rate file (optional)", type=['xlsx'])
        const_rate = st.number_input("Otherwise: constant rate % p.a.", value=40.0, step=1.0) / 100
        window = st.selectbox("Performance window", [f"Since {BACKTEST_START}", "Last 6 months",
                                                     "Last 3 months"])
        run = st.button("Run", type="primary", **WIDE)

    models = {}
    try:
        if train_in_app:
            raise StopIteration
        if up_models is not None:
            models = json.load(up_models)
        elif os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), MODELS_FILE)):
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), MODELS_FILE)) as f:
                models = json.load(f)
    except StopIteration:
        pass
    except Exception as e:
        st.sidebar.error(f"Could not read the model file: {e}")
    if not train_in_app:
        if models:
            st.sidebar.success("Models: " + ", ".join(f"{k} (cutoff {v['cutoff']})" for k, v in models.items()))
        else:
            st.sidebar.warning("No model file — only the EMA and HA rules will be shown.")

    if not run:
        st.info("Choose tickers in the sidebar and click **Run**.")
        st.stop()

    rates = None
    if evds is not None:
        try:
            rates = parse_evds(evds)
        except Exception as e:
            st.sidebar.error(f"Could not read the EVDS file: {e}")

    first_day = BACKTEST_START
    start = (pd.Timestamp(TRAIN_START) - pd.Timedelta(days=LEAD_DAYS)).strftime('%Y-%m-%d')
    end = (datetime.date.today() + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
    for k, ticker in enumerate([t.strip().upper() for t in tickers_in.split(',') if t.strip()]):
        if k:
            time.sleep(0.8)
        st.divider()
        st.subheader(f"📊 {ticker}")
        df = cached_prices(ticker, start, end)
        if df is None or (df.index >= pd.Timestamp(first_day)).sum() < 20:
            st.warning(f"No / insufficient price data for **{ticker}**.")
            continue
        cash = cash_series(df.index, rates, const_rate)
        dates_first = df.index[int(np.searchsorted(df.index, pd.Timestamp(BACKTEST_START)))]
        if train_in_app:
            ema_len = int(ema_in)
            res = analyse(df, None, ema_len, float(fee), cash, train_in_app=True)
            if res is None:
                st.warning(f"Not enough data since {TRAIN_START} to train on **{ticker}** "
                           "(needs at least one complete up- and down-leg).")
                continue
            sp = res['spec']
            late = pd.Timestamp(sp['first_prediction']) > dates_first
            st.caption(f"ℹ️ Logit trained in the app on {sp['train_start']} → {sp['cutoff']}, retrained monthly "
                       f"({sp['n_refits']} fits; {sp['n_train']} labelled days in the latest). Predictions are out of "
                       f"sample from {sp['first_prediction']}"
                       + (" — in cash before that, while the training sample was too short." if late else ".")
                       + " A small sample: noisier than the long-history research model.")
        else:
            spec = models.get(ticker)
            ema_len = int(spec['ema_len']) if spec else 10
            if spec is None:
                st.caption(f"No saved model for {ticker} — showing the rules only "
                           f"(add {ticker} to TICKERS in export_regime_model.py).")
            elif pd.Timestamp(spec['cutoff']) >= pd.Timestamp(BACKTEST_START):
                st.caption(f"ℹ️ Model trained up to {spec['cutoff']}: results before that date are in-sample.")
            res = analyse(df, spec, ema_len, float(fee), cash, train_in_app=False)
        spec = res['spec']
        a0, n, dates = res['a'], len(df), df.index

        # ---------- current signals ----------
        last_close = float(df['Close'].iloc[-1])
        live = live_quote(ticker)
        c1, c2, c3 = st.columns(3)
        c1.metric(f"Last settled close ({dates[-1].date()})", f"{last_close:,.2f}")
        c2.metric("Live quote", f"{live:,.2f}" if live else "n/a",
                  delta=f"{(live / last_close - 1) * 100:+.2f}%" if live else None)
        p_now = res['prob'][-1] if res['prob'] is not None else np.nan
        c3.metric("Logit P(up-leg)", f"{p_now:.2f}" if np.isfinite(p_now) else "n/a",
                  help=f"UP if ≥ {P_LONG}, DOWN if ≤ {P_SHORT}, otherwise the previous state is kept.")

        st.markdown("#### 🎯 Current positions (long-only, after the last settled close)")
        cols = st.columns(len(res['sigs']))
        for col, (name, sg) in zip(cols, res['sigs'].items()):
            stt = position_status(df, sg, a0)
            with col:
                st.markdown(f"**{name}** — {'🟢 LONG' if stt['state'] == 'LONG' else '⚪ CASH'}")
                if stt['state'] == 'LONG':
                    st.caption(f"Bought {stt['since']} at {stt['entry']:,.2f} · P&L since entry {stt['pnl']:+.2f}%")
                else:
                    st.caption(f"In cash since {stt['since']}" if stt['since'] else "In cash")

        pv = live_preview(res, df, live, ema_len)
        if pv:
            with st.expander("🔮 Provisional — signals if today closed at the live quote (not part of the backtest)"):
                side = lambda v: "🟢 LONG" if v == 1 else "⚪ CASH"
                txt = f"**EMA({ema_len}):** {side(pv['ema'])} · **HA:** {side(pv['ha'])}"
                if pv['logit'] is not None:
                    txt = f"**Logit [EMA + HA]:** P(up) {pv['p']:.2f} → {side(pv['logit'])} · " + txt
                st.write(txt)
                st.caption("Signals become final only at the official close; intraday they can flip back.")

        # ---------- performance ----------
        starts = {f"Since {BACKTEST_START}": dates[a0],
                  "Last 6 months": dates[-1] - pd.DateOffset(months=6),
                  "Last 3 months": dates[-1] - pd.DateOffset(months=3)}
        a = max(a0, int(np.searchsorted(dates, starts[window])))
        st.markdown(f"#### Performance — {window.lower()} ({dates[a].date()} → {dates[-1].date()})")
        rows = {nm: perf(*v, cash, a, n) for nm, v in res['strats'].items()}
        rows['Buy & hold']['Trades / yr'] = np.nan
        st.dataframe(pd.DataFrame(rows).T.style.format("{:,.2f}", na_rep="—"), **WIDE)
        st.caption(f"Net of {fee * 100:.2f}% per side; cash earns "
                   + ("the EVDS deposit rate" if rates is not None else f"{const_rate * 100:.0f}% p.a.")
                   + "; Sharpe in excess of cash. Short windows are dominated by noise — "
                     "the long-run evidence is in the presentation.")

        # ---------- charts ----------
        fig, axes = plt.subplots(3 if res['prob'] is not None else 2, 1, figsize=(13, 10),
                                 gridspec_kw={'height_ratios': [3, 2, 1.3][:3 if res['prob'] is not None else 2]})
        colors = {'Logit [EMA + HA]': 'tab:red', f'EMA({ema_len}) rule': 'tab:blue', 'HA rule': 'tab:orange',
                  'Buy & hold': 'grey'}
        for nm, v in res['strats'].items():
            axes[0].plot(dates[a:], np.cumprod(1 + v[0][a:]), color=colors[nm], lw=2.2 if 'Logit' in nm else 1.4,
                         ls='--' if nm == 'Buy & hold' else '-', label=nm)
        axes[0].set_title(f"{ticker} — growth of 1 ({window.lower()})", fontweight='bold')
        axes[0].legend(loc='upper left', fontsize=9)
        axes[0].grid(alpha=0.3)

        key = 'Logit [EMA + HA]' if 'Logit [EMA + HA]' in res['sigs'] else f'EMA({ema_len}) rule'
        sg = res['sigs'][key]
        z = max(a0, n - 126)
        px = df['Close'].values
        axes[1].plot(dates[z:], px[z:], color='black', lw=1)
        axes[1].fill_between(dates[z:], 0, 1, where=sg[z:] == 1, step='post', color='green', alpha=0.15,
                             transform=axes[1].get_xaxis_transform(), lw=0)
        axes[1].fill_between(dates[z:], 0, 1, where=sg[z:] != 1, step='post', color='red', alpha=0.10,
                             transform=axes[1].get_xaxis_transform(), lw=0)
        bots = [t for t in range(z + 1, n) if sg[t - 1] != 1 and sg[t] == 1]
        pks = [t for t in range(z + 1, n) if sg[t - 1] == 1 and sg[t] != 1]
        axes[1].scatter(dates[bots], px[bots], marker='^', color='darkgreen', s=60, zorder=5)
        axes[1].scatter(dates[pks], px[pks], marker='v', color='darkred', s=60, zorder=5)
        axes[1].set_title(f"{key} — last 6 months (green = long, red = cash; ▲ buy, ▼ sell)", fontweight='bold')
        axes[1].grid(alpha=0.3)
        if res['prob'] is not None:
            axes[2].plot(dates[z:], res['prob'][z:], color='tab:purple')
            axes[2].axhline(P_LONG, color='green', ls='--', lw=1)
            axes[2].axhline(P_SHORT, color='red', ls='--', lw=1)
            axes[2].set_ylim(0, 1)
            axes[2].set_ylabel('P(up)')
            axes[2].grid(alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        if spec is not None:
            with st.expander("Model details"):
                st.write(f"Trained {spec['train_start']} → {spec['cutoff']} on {spec['n_train']:,} labelled days "
                         f"(latest fit); EMA length {spec['ema_len']}.")
                st.dataframe(pd.DataFrame({'standardised coefficient': spec['coef']}, index=spec['features'])
                             .style.format("{:+.3f}"))


if __name__ == '__main__' and not os.environ.get('REGIME_APP_TEST'):
    main()


# import streamlit as st
# import pandas as pd
# import numpy as np
# import yfinance as yf
# import matplotlib.pyplot as plt
# import datetime
# import time
# import warnings

# warnings.filterwarnings("ignore")

# st.set_page_config(page_title="Trend Strategy Backtester", layout="wide")

# DEFAULT_TICKERS = "SI=F, XU030.IS, EREGL.IS, SASA.IS, ENJSA.IS"


# # ============================================================
# #  CORE STRATEGY LOGIC
# #  Corrections vs. the previous version (same as the script):
# #   [FIX 1] Exit-day return is booked (hold from yesterday's close to
# #           today's close, then sell at today's close).
# #   [FIX 2] Exit fee only charged if a position is actually open.
# #   [FIX 3] Current Action panel detects a take-profit already reached
# #           during the current trend (position closed → flat).
# # ============================================================

# def backtest_long_only(df, signal_col, long_tp, fee):
#     """
#     Long-only strategy driven by a generic +1/-1 trend/signal column.
#     Enter long when signal flips to +1 at that bar's close; exit either
#     when the signal flips to -1 (at that bar's close), or when the running
#     total P&L on the open position reaches long_tp — whichever comes first.
#     """
#     prices = df["Close"].values
#     signal = df[signal_col].values
#     n = len(prices)
#     strat_rets = np.zeros(n)
#     in_position = 0
#     entry_price = 0.0
#     current_signal = 0

#     for i in range(1, n):
#         # [FIX 1] book today's move first if the position was held into today
#         if in_position == 1:
#             strat_rets[i] += (prices[i] - prices[i - 1]) / prices[i - 1]

#         if signal[i] != current_signal:
#             current_signal = signal[i]
#             if current_signal == 1:
#                 in_position = 1
#                 entry_price = prices[i]
#                 strat_rets[i] -= fee
#             else:
#                 # [FIX 2] exit fee only if a position is open
#                 if in_position == 1:
#                     strat_rets[i] -= fee
#                     in_position = 0
#             continue

#         if in_position == 1:
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
#         # [FIX 1]
#         if in_position == 1:
#             strat_rets[i] += -(prices[i] - prices[i - 1]) / prices[i - 1]

#         if signal[i] != current_signal:
#             current_signal = signal[i]
#             if current_signal == -1:
#                 in_position = 1
#                 entry_price = prices[i]
#                 strat_rets[i] -= fee
#             else:
#                 # [FIX 2]
#                 if in_position == 1:
#                     strat_rets[i] -= fee
#                     in_position = 0
#             continue

#         if in_position == 1:
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
#     the current streak has run, the entry level, unrealized P&L, the
#     take-profit / limit-order target, and whether that target was
#     already reached during the current trend.
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

#     after = prices[entry_idx + 1:]   # bars on which the backtest checks the TP

#     if current_sig == 1:
#         side = "LONG"
#         pnl_pct = (current_price - entry_price) / entry_price * 100
#         target_price = entry_price * (1 + long_tp)
#         target_label = "Limit order (take-profit) to CLOSE the long"
#         hit_mask = (after - entry_price) / entry_price >= long_tp
#     else:
#         side = "DOWN / FLAT (long-only strategy takes no position)"
#         pnl_pct = (entry_price - current_price) / entry_price * 100
#         target_price = entry_price * (1 - short_tp)
#         target_label = "Informational SHORT target (not traded)"
#         hit_mask = (entry_price - after) / entry_price >= short_tp

#     # [FIX 3] take-profit already reached in this trend?
#     tp_hit_date = None
#     if hit_mask.any():
#         tp_hit_date = dates[entry_idx + 1 + int(np.argmax(hit_mask))].strftime("%Y-%m-%d")

#     return {
#         "side": side,
#         "days_in_trend": days_in_trend,
#         "entry_price": round(entry_price, 2),
#         "entry_date": entry_date,
#         "current_price": round(current_price, 2),
#         "pnl_pct": round(pnl_pct, 2),
#         "target_price": round(target_price, 2),
#         "target_label": target_label,
#         "tp_hit_date": tp_hit_date,
#     }


# @st.cache_data(ttl=3600, show_spinner=False)
# def load_data(ticker, start, end, max_retries=3):
#     """Fetches with retry+backoff. Yahoo Finance rate-limits repeated
#     sequential requests (more likely the more tickers you fetch in one
#     run, especially from a shared/cloud IP) -- when that happens,
#     yfinance often doesn't raise a clean error, it just returns an
#     empty frame or one whose Close values are all NaN. Retrying after a
#     short backoff, rather than accepting the bad response immediately,
#     resolves most of these transient cases."""
#     last_df = None
#     for attempt in range(max_retries):
#         try:
#             df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
#         except Exception:
#             df = None

#         if df is not None and not df.empty:
#             if isinstance(df.columns, pd.MultiIndex):
#                 df.columns = df.columns.get_level_values(0)
#             last_df = df
#             if "Close" in df.columns and df["Close"].notna().any():
#                 return df  # usable data -- done

#         if attempt < max_retries - 1:
#             time.sleep(1.5 * (attempt + 1))  # backoff before retrying

#     return last_df  # best available result after retries, even if still unusable -- caller checks it


# @st.cache_data(ttl=120, show_spinner=False)
# def get_live_quote(ticker):
#     """Best-effort real-time/delayed quote -- separate from the daily
#     historical series used for signals/backtesting. This is what
#     should match the price shown on the Yahoo Finance website, since
#     that reflects live/intraday (and sometimes pre/after-market)
#     trading, whereas the daily bar used elsewhere in this app only
#     updates once a session is fully settled. Short TTL (2 min) since
#     the whole point of this value is to be current."""
#     try:
#         fi = yf.Ticker(ticker).fast_info
#         price = fi.get("last_price") if isinstance(fi, dict) else getattr(fi, "last_price", None)
#         if price:
#             return float(price)
#     except Exception:
#         pass
#     try:
#         info = yf.Ticker(ticker).info
#         price = info.get("regularMarketPrice") or info.get("currentPrice")
#         if price:
#             return float(price)
#     except Exception:
#         pass
#     return None


# def preview_signal_with_live_price(df, live_price, ema_length):
#     """Forward-looking, PROVISIONAL preview only -- does not touch df,
#     the backtest returns, or any stat shown elsewhere. Treats the live
#     quote as a stand-in for "today's close" and recomputes just the
#     HA/EMA trend classification for that one hypothetical bar, so you
#     can see what the signal would become IF the session settled right
#     now. This is intentionally NOT fed back into df/backtest_long_only/
#     optimize_*_tp -- doing so would let a still-moving intraday price
#     flip the trend classification back and forth before the real close
#     prints, making the backtest and 'Current Action' non-reproducible
#     within the same day. Keep this strictly a preview."""
#     if live_price is None or len(df) == 0:
#         return None

#     last_close = float(df["Close"].iloc[-1])
#     prev_ha_close = float(df["HA_Close"].iloc[-1])
#     prev_ha_open = float(df["HA_Open"].iloc[-1])
#     prev_ema = float(df["EMA_Val"].iloc[-1])

#     # Synthetic hypothetical bar: last settled close -> live price.
#     o, c = last_close, live_price
#     h, l = max(o, c), min(o, c)

#     ha_close_new = (o + h + l + c) / 4
#     ha_open_new = (prev_ha_open + prev_ha_close) / 2
#     ha_trend_new = 1 if ha_close_new >= ha_open_new else -1

#     alpha = 2 / (ema_length + 1)
#     ema_new = c * alpha + prev_ema * (1 - alpha)
#     ema_trend_new = 1 if c >= ema_new else -1

#     return {"ha_trend": ha_trend_new, "ema_trend": ema_trend_new}


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

# for idx, ticker in enumerate(tickers):
#     if idx > 0:
#         time.sleep(0.8)  # brief pacing between sequential Yahoo Finance requests

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

#         # Drop trailing rows with NaN Close (unsettled/incomplete bars).
#         n_before = len(df)
#         df = df.dropna(subset=["Close"])
#         if len(df) < n_before:
#             st.caption(f"ℹ️ Dropped {n_before - len(df)} trailing row(s) with no settled Close price for {ticker}.")

#         if df.empty or len(df) < 30:
#             st.warning(
#                 f"No / insufficient *settled* price history for **{ticker}** after removing "
#                 f"incomplete rows and retrying the fetch. This is often Yahoo Finance "
#                 f"rate-limiting rather than the ticker itself -- try clicking **Run Backtest** "
#                 f"again in a few seconds, or fetch fewer tickers at once."
#             )
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
#     live_price = get_live_quote(ticker)

#     top1, top2, top3 = st.columns(3)
#     settled_date_str = df.index[-1].strftime("%Y-%m-%d")
#     top1.metric("Last Settled Close (used in backtest)", f"{current_price:.2f}")
#     top1.caption(f"As of {settled_date_str}")
#     if live_price is not None:
#         gap_vs_settled = (live_price - current_price) / current_price * 100
#         top2.metric("Live Quote (from Yahoo Finance)", f"{live_price:.2f}",
#                     delta=f"{gap_vs_settled:+.2f}% vs settled close",
#                     help="Real-time/delayed quote -- this is what the Yahoo Finance website shows, "
#                          "and can differ from the settled daily close, especially for near-"
#                          "continuously-traded tickers like futures, during/after market hours, or "
#                          "for exchanges (like BIST) where the official close comes from a separate "
#                          "closing auction rather than the last continuous trade.")
#     else:
#         top2.metric("Live Quote (from Yahoo Finance)", "n/a")
#     top3.metric("RSI(14)", f"{float(last['RSI']):.1f}" if not np.isnan(last["RSI"]) else "n/a")

#     st.caption("ℹ️ **Last Settled Close** is the completed daily bar the backtest and target levels "
#                "below are calculated from. **Live Quote** is a separate, real-time lookup meant to "
#                "match what you'd see on the Yahoo Finance website right now -- the two can genuinely "
#                "differ until the current session settles.")

#     if live_price is not None:
#         preview = preview_signal_with_live_price(df, live_price, ema_length)
#         if preview is not None:
#             ha_now_long = ha_status["side"] == "LONG"
#             ema_now_long = ema_status["side"] == "LONG"
#             ha_preview_long = preview["ha_trend"] == 1
#             ema_preview_long = preview["ema_trend"] == 1

#             def _side_str(is_long):
#                 return "🟢 LONG" if is_long else "⚪ FLAT"

#             def _flip_note(now_long, preview_long):
#                 return " *(would flip)*" if now_long != preview_long else ""

#             with st.expander("🔮 Live preview — what the signal would be if today settled right now (provisional)"):
#                 st.caption(
#                     "This is NOT part of the backtest, the table below, or the plot -- it's a "
#                     "what-if using the live quote as a stand-in for today's close. It will keep "
#                     "changing until the session actually settles, and can flip back before it does."
#                 )
#                 p1, p2 = st.columns(2)
#                 p1.write(f"**Heikin-Ashi:** {_side_str(ha_now_long)} (settled) → "
#                           f"{_side_str(ha_preview_long)} (if settled at {live_price:.2f})"
#                           f"{_flip_note(ha_now_long, ha_preview_long)}")
#                 p2.write(f"**EMA({ema_length}):** {_side_str(ema_now_long)} (settled) → "
#                           f"{_side_str(ema_preview_long)} (if settled at {live_price:.2f})"
#                           f"{_flip_note(ema_now_long, ema_preview_long)}")

#     st.markdown("#### 🎯 Current Action & Target Levels (based on optimal TP thresholds)")
#     a1, a2 = st.columns(2)
#     for col, name, status, l_tp, s_tp in [
#         (a1, "Heikin-Ashi", ha_status, best_tp_ha, best_short_tp_ha),
#         (a2, f"EMA({ema_length})", ema_status, best_tp_ema, best_short_tp_ema),
#     ]:
#         with col:
#             is_long = status["side"] == "LONG"
#             tp_hit = status["tp_hit_date"] is not None
#             # [FIX 3] show that the position was already closed at the take-profit
#             if is_long and tp_hit:
#                 action_label = (f"✅ Target reached on {status['tp_hit_date']} — long closed, "
#                                 f"FLAT until the next buy signal")
#             elif is_long:
#                 action_label = "🟢 LONG — holding"
#             elif tp_hit:
#                 action_label = (f"🔴 SHORT (informational) — short target already reached on "
#                                 f"{status['tp_hit_date']}")
#             else:
#                 action_label = "🔴 SHORT (informational only, no position — long-only strategy)"
#             st.markdown(f"**{name}**")
#             st.write(f"Action: **{action_label}**")
#             gap_pct = (status["target_price"] - current_price) / current_price * 100
#             g1, g2 = st.columns(2)
#             g1.metric("Settled Close", f"{current_price:.2f}")
#             g2.metric(
#                 ("Target Price (reached)" if tp_hit else "Target Price") if is_long
#                 else "Informational Target",
#                 f"{status['target_price']:.2f}",
#                 delta=f"{gap_pct:+.2f}% away",
#             )
#             st.caption(
#                 f"Entry {status['entry_date']} @ {status['entry_price']} · "
#                 f"day {status['days_in_trend']} of trend · "
#                 f"{'P&L since entry' if tp_hit else 'unrealized P&L'} {status['pnl_pct']:+.2f}% · "
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
