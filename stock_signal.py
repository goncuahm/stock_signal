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
GRID_SHORT   = np.round(np.arange(0.29, 0.4901, 0.02), 2)   # candidate cash thresholds (optimisation)
GRID_LONG    = np.round(np.arange(0.51, 0.7101, 0.02), 2)   # candidate long thresholds (optimisation)
MIN_EXPOSURE = 0.20           # a threshold pair must keep the strategy in the market ≥ 20% of training days
OPT_AFTER_MONTHS = 6          # with optimisation on: fixed thresholds for the first 6 months of the backtest
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


def choose_thresholds(p_tr, close_tr, cash_tr, fee):
    """Best (cash, long) threshold pair on the training window: long-only strategy net of costs,
    Sharpe ratio in excess of cash, at least MIN_EXPOSURE of days in the market."""
    best, best_s = None, -np.inf
    for ps in GRID_SHORT:
        for pl in GRID_LONG:
            r, e, _ = long_only(close_tr, regime_signal(p_tr, pl, ps), cash_tr, fee, 0)
            if e.mean() < MIN_EXPOSURE or r.std() < 1e-10:
                continue
            sh = (r - cash_tr).mean() / r.std() * np.sqrt(252)
            if sh > best_s:
                best, best_s = (float(ps), float(pl)), sh
    return best, best_s


def walk_forward_in_app(close, X, dates, t0, a, cash=None, fee=0.0, optimise=False,
                        p_long=P_LONG, p_short=P_SHORT):
    """Expanding-window training on data from index t0 (TRAIN_START). Refit on the first trading
    day of every month from index a (BACKTEST_START) on; each model predicts until the next refit.
    Labels at each refit are computed from prices t0..R only. With optimise=True the thresholds are
    also chosen at each refit, on the training window t0..R only, and used until the next refit — starting
    OPT_AFTER_MONTHS after BACKTEST_START; before that the given (fixed) thresholds are used."""
    n = len(close)
    prob = np.full(n, np.nan)
    thl, ths = np.full(n, np.nan), np.full(n, np.nan)
    month = dates[a:].to_period('M')
    refits = [a + int(i) for i in np.r_[0, np.where(month[1:] != month[:-1])[0] + 1]]
    spec, n_refits, n_lab, first, table = None, 0, 0, None, []
    cur_thr = (p_short, p_long)
    opt_from = dates[a] + pd.DateOffset(months=OPT_AFTER_MONTHS)
    for k, R in enumerate(refits):
        seg = close[t0:R + 1]
        y = labels_from(smooth_pivots(seg), len(seg))
        tr = np.where(np.isfinite(y) & np.isfinite(X[t0:R + 1]).all(axis=1))[0]
        refit = len(tr) >= MIN_LABELLED and len(np.unique(y[tr])) == 2
        if refit:
            spec = fit_spec(X[t0 + tr], y[tr])
            n_refits, n_lab = n_refits + 1, len(tr)
        if spec is None:
            continue                                   # not enough labelled history yet → stay in cash
        if optimise and refit:
            if dates[R] >= opt_from:
                best, best_s = choose_thresholds(predict_proba(spec, X[t0:R + 1]), close[t0:R + 1],
                                                 cash[t0:R + 1], fee)
                cur_thr = best if best is not None else (p_short, p_long)
                method = 'optimised' if best is not None else 'fixed (no pair met the 20% rule)'
            else:
                cur_thr, best_s, method = (p_short, p_long), np.nan, f'fixed (first {OPT_AFTER_MONTHS} months)'
            table.append({'Refit': str(dates[R].date()), 'Training days': R - t0 + 1, 'Method': method,
                          'Cash if P ≤': cur_thr[0], 'Long if P ≥': cur_thr[1],
                          'Training Sharpe (xs)': best_s})
        b = refits[k + 1] if k + 1 < len(refits) else n
        prob[R:b] = predict_proba(spec, X[R:b])
        ths[R:b], thl[R:b] = cur_thr
        first = R if first is None else first
    if spec is not None:
        spec.update({'n_train': n_lab, 'n_refits': n_refits, 'first_prediction': str(dates[first].date())})
    return prob, spec, thl, ths, table


def regime_signal(p, p_long=P_LONG, p_short=P_SHORT):
    """+1 long / -1 cash with a hysteresis band; thresholds may be numbers or per-day arrays."""
    pl = np.broadcast_to(np.asarray(p_long, float), len(p))
    ps = np.broadcast_to(np.asarray(p_short, float), len(p))
    s, cur = np.zeros(len(p)), 0.0
    for t in range(len(p)):
        if np.isfinite(p[t]):
            if p[t] >= pl[t]:
                cur = 1.0
            elif p[t] <= ps[t]:
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


def analyse(df, spec, ema_len, fee, cash, train_in_app=True, p_long=P_LONG, p_short=P_SHORT, optimise=False):
    close = df['Close'].values.astype(float)
    n = len(close)
    F, aux = build_features(df, ema_len)
    sigs = {}
    prob = None
    a = int(np.searchsorted(df.index, pd.Timestamp(BACKTEST_START)))
    thl, ths, table = np.full(n, p_long), np.full(n, p_short), []
    if train_in_app:
        t0 = int(np.searchsorted(df.index, pd.Timestamp(TRAIN_START)))
        prob, spec, thl, ths, table = walk_forward_in_app(close, F[FEATURES].values, df.index, t0, a, cash, fee,
                                                          optimise, p_long, p_short)
        if spec is None:
            return None                              # not enough labelled data yet
        spec.update({'train_start': str(df.index[t0].date()), 'cutoff': str(df.index[-1].date()),
                     'ema_len': ema_len})
        sigs['Logit [EMA + HA]'] = regime_signal(prob, thl, ths)
    elif spec is not None:
        prob = predict_proba(spec, F[spec['features']].values)
        prob[:a] = np.nan                            # the saved model is applied from BACKTEST_START on
        sigs['Logit [EMA + HA]'] = regime_signal(prob, thl, ths)
    sigs[f'EMA({ema_len}) rule'] = F['EMA_signal'].values
    sigs['HA rule'] = F['HA_trend'].values
    strats = {nm: long_only(close, sg, cash, fee, a) for nm, sg in sigs.items()}
    bh = np.zeros(n)
    bh[a + 1:] = close[a + 1:] / close[a:-1] - 1
    bh[a] = cash[a] - fee
    strats['Buy & hold'] = (bh, np.r_[np.zeros(a + 1), np.ones(n - a - 1)], np.r_[np.zeros(a), 1.0, np.zeros(n - a - 1)])
    return {'a': a, 'F': F, 'aux': aux, 'prob': prob, 'sigs': sigs, 'strats': strats, 'spec': spec,
            'thl': thl, 'ths': ths, 'thr_table': table, 'optimised': bool(optimise and train_in_app),
            'opt_from': next((pd.Timestamp(r['Refit']) for r in table if r['Method'] == 'optimised'), None)}


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
        pl, ps = res['thl'][-1], res['ths'][-1]
        out['p'], out['logit'] = p, (1.0 if p >= pl else (-1.0 if p <= ps else cur))
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
        optimise = st.checkbox("Choose thresholds optimally from the training data", value=False,
                               disabled=not train_in_app,
                               help=f"The thresholds below are used for the first {OPT_AFTER_MONTHS} months of "
                                    "the backtest. After that, at every monthly refit, cash thresholds 0.29–0.49 and "
                                    "long thresholds 0.51–0.71 (steps of 0.02) are tested on the training data only; "
                                    "the pair with the best Sharpe ratio in excess of cash is used until the next "
                                    "refit. Available with in-app training.") and train_in_app
        lbl = f" (first {OPT_AFTER_MONTHS} months)" if optimise else ""
        p_long = st.slider("Go long when P(up) ≥" + lbl, 0.50, 0.90, P_LONG, 0.01)
        p_short = st.slider("Go to cash when P(up) ≤" + lbl, 0.10, 0.50, P_SHORT, 0.01,
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
            res = analyse(df, None, ema_len, float(fee), cash, train_in_app=True, p_long=p_long, p_short=p_short,
                          optimise=optimise)
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
                  help=f"Long if ≥ {res['thl'][-1]:.2f}, cash if ≤ {res['ths'][-1]:.2f}, otherwise the previous "
                       "position is kept" + (" (thresholds chosen on the training data)." if res['optimised'] else "."))
        if res['optimised'] and res['thr_table']:
            last = res['thr_table'][-1]
            st.caption(f"🎚️ Thresholds in use since the latest refit ({last['Refit']}, {last['Method']}): "
                       f"**long if P ≥ {last['Long if P ≥']:.2f}, cash if P ≤ {last['Cash if P ≤']:.2f}**. "
                       + (f"Optimisation started {res['opt_from'].date()}." if res['opt_from'] is not None else
                          f"Optimisation starts {OPT_AFTER_MONTHS} months after {BACKTEST_START}."))
            with st.expander("Thresholds chosen at each monthly refit (training data only)"):
                st.dataframe(pd.DataFrame(res['thr_table']).set_index('Refit')
                             .style.format({'Cash if P ≤': '{:.2f}', 'Long if P ≥': '{:.2f}',
                                            'Training Sharpe (xs)': '{:.2f}'}, na_rep='—'), **WIDE)
                st.caption("Each pair is chosen using only data up to its refit date and applied to the "
                           "following month, so the backtest stays out of sample.")

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
        z = a0                                       # whole backtest window (from BACKTEST_START)
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
        axes[1].set_title(f"{key} — regimes {dates[z].date()} → {dates[-1].date()} (green = long, red = cash; "
                          "▲ buy, ▼ sell)", fontweight='bold')
        axes[1].grid(alpha=0.3)
        if res['prob'] is not None:
            axes[2].plot(dates[z:], res['prob'][z:], color='tab:purple')
            axes[2].step(dates[z:], res['thl'][z:], where='post', color='green', ls='--', lw=1.6, label='long threshold')
            axes[2].step(dates[z:], res['ths'][z:], where='post', color='red', ls='--', lw=1.6, label='cash threshold')
            if res['opt_from'] is not None:
                axes[2].axvline(res['opt_from'], color='black', ls=':', lw=1.2)
                axes[2].annotate('thresholds optimised from here →', (res['opt_from'], 0.97), ha='right', va='top',
                                 fontsize=8, textcoords='offset points', xytext=(-4, 0))
            axes[2].set_title("P(up-leg) with the long (green) and cash (red) thresholds in use", fontsize=10)
            axes[2].legend(loc='lower left', fontsize=8, ncol=2)
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
