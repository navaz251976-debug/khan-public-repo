"""Render FinViz-style annotated daily candlestick charts for all tickers.

Instead of scraping the FinViz PNG (whose pixel/price mapping can't be
calibrated reliably), we render the chart ourselves from yfinance OHLCV in
the same dark style, then draw analyst-style annotations in data coordinates:
support/resistance dashed lines, callout labels with dashed leaders, a spike
circle + fade arrow, a volume-surge box, the after-hours level, and a READ
summary box.

Output: docs/charts/<TICKER>.png — overwritten on every run, never committed
to git (served to GitHub Pages via the deploy artifact). Each image carries a
"Generated <time> ET" stamp so staleness is visible.
"""
import datetime as dt
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans
from matplotlib.patches import Ellipse, Rectangle
import mplfinance as mpf
import numpy as np
import pandas as pd
import yfinance as yf

CHARTS_DIR = Path("docs") / "charts"
FETCH_PERIOD = "15mo"   # enough history for a valid SMA200 at the window start
DISPLAY_BARS = 185      # ~9 months of trading days shown

BG       = "#11151c"
GRID     = "#232a35"
SMA50_C  = "#5082ff"
SMA200_C = "#ff9800"
RED      = "#ff5050"
GREEN    = "#3fa651"
YELLOW   = "#ffd23c"
ORANGE   = "#ff9632"
CYAN     = "#50c8ff"
PURPLE   = "#be8cff"
WHITE    = "#ebf0f5"
GRAY     = "#8b949e"
BOX_BG   = (0.04, 0.055, 0.10, 0.88)


def fetch_ohlcv(symbols: list) -> dict:
    """One batched download -> {sym: OHLCV DataFrame}."""
    data = yf.download(symbols, period=FETCH_PERIOD, group_by="ticker",
                       auto_adjust=True, progress=False)
    out = {}
    for sym in symbols:
        try:
            df = data[sym] if isinstance(data.columns, pd.MultiIndex) else data
        except KeyError:
            continue
        df = df.dropna(subset=["Close"])
        if len(df) >= 30:
            out[sym] = df
    return out


def analyze(df: pd.DataFrame, sma50: pd.Series, sma200: pd.Series, ah: tuple) -> dict:
    """Rule-based chart read: trend leg, spike/fade, SMA200 interaction,
    support/resistance, volume surge, after-hours context.

    df / sma series are the display window; the SMAs were computed on the
    full fetched history so they are valid from the first displayed bar."""
    n = len(df)
    highs, lows, closes = df["High"], df["Low"], df["Close"]
    close = float(closes.iloc[-1])
    s50 = float(sma50.iloc[-1]) if pd.notna(sma50.iloc[-1]) else None
    s200 = float(sma200.iloc[-1]) if pd.notna(sma200.iloc[-1]) else None

    a = {"close": close, "sma50": sma50, "sma200": sma200,
         "s50": s50, "s200": s200}

    # Major leg: biggest move between the window's extreme high and low
    i_hi, i_lo = int(np.argmax(highs.values)), int(np.argmin(lows.values))
    hi, lo = float(highs.iloc[i_hi]), float(lows.iloc[i_lo])
    a["leg"] = None
    if i_hi < i_lo and lo / hi - 1 < -0.25:
        a["leg"] = ("bear", i_hi, i_lo, lo / hi - 1)
    elif i_lo < i_hi and hi / lo - 1 > 0.30:
        a["leg"] = ("bull", i_lo, i_hi, hi / lo - 1)

    # Recent blow-off spike that faded hard
    a["spike"] = None
    if n >= 126:
        rec = highs.iloc[-20:]
        i_rhi = n - 20 + int(np.argmax(rec.values))
        rhi = float(rec.max())
        if rhi >= float(highs.iloc[-126:].max()) and close <= rhi * 0.92:
            a["spike"] = (i_rhi, rhi, close / rhi - 1)

    # SMA200 interaction
    a["sma200_note"] = None
    if s200 is not None:
        recently_above = bool((closes.iloc[-20:] > sma200.iloc[-20:]).any())
        if close < s200 and recently_above:
            cross = np.where(closes.iloc[-20:].values > sma200.iloc[-20:].values)[0]
            a["sma200_note"] = ("failed", n - 20 + int(cross[-1]),
                                f"SMA200 ~{s200:,.0f} reclaim FAILED -> resistance")
        elif close > s200:
            a["sma200_note"] = ("above", n - 1,
                                f"Above SMA200 ~{s200:,.0f} -> support")
        else:
            a["sma200_note"] = ("below", n - 1,
                                f"Below SMA200 ~{s200:,.0f} -> resistance")

    # Support / resistance levels: recent base vs older major low
    near_lows = lows.iloc[-25:]
    a["sup_near"] = float(near_lows.min())
    a["i_sup_near"] = n - 25 + int(np.argmin(near_lows.values))
    older = lows.iloc[:-25]
    a["sup_major"] = None
    if len(older) and float(older.min()) < a["sup_near"] * 0.97:
        a["sup_major"] = float(older.min())
        a["i_sup_major"] = int(np.argmin(older.values))
    a["res"] = float(highs.iloc[-63:].max())

    # Volume surge
    vol = df["Volume"]
    a["vol_surge"] = (n >= 90 and
                      float(vol.iloc[-5:].max()) > 2.5 * float(vol.iloc[-90:].mean()))

    # After-hours
    ah_pct, _, ah_price = (ah or (0.0, 0, 0))
    a["ah"] = (ah_pct, ah_price) if abs(ah_pct) >= 1.0 and ah_price else None

    # Sentiment badge (simple positional read; detailed logic lives in READ box)
    if s200 is None or s50 is None:
        a["sentiment"] = "NEUTRAL"
    elif close > s50 and close > s200:
        a["sentiment"] = "BULLISH"
    elif close > s200:
        a["sentiment"] = "LEANS BULLISH"
    elif close < s50 and close < s200:
        a["sentiment"] = "BEARISH"
    else:
        a["sentiment"] = "LEANS BEARISH"

    # READ summary
    lines = []
    if a["sma200_note"] and a["sma200_note"][0] == "failed":
        lines.append("Recovery rally rejected at the 200-day.")
    elif s200 is not None and close > s200:
        lines.append("Trend constructive: price above the 200-day.")
    elif s200 is not None:
        lines.append("Downtrend: price below the 200-day.")
    if a["ah"] and a["ah"][0] <= -2:
        lines.append(f"AH drop {a['ah'][0]:+.1f}% lands near support ~{a['sup_near']:,.0f}.")
    elif a["ah"] and a["ah"][0] >= 2:
        lines.append(f"AH pop {a['ah'][0]:+.1f}% — gap toward ~{a['res']:,.0f} resistance.")
    if a["sup_major"]:
        lines.append(f"Hold ~{a['sup_near']:,.0f} -> range repair;  "
                     f"lose it -> {a['sup_major']:,.0f} next.")
    else:
        lines.append(f"Key support ~{a['sup_near']:,.0f}.")
    if s200 is not None and close < s200:
        lines.append(f"Reclaim {s200:,.0f}+ (SMA200) to turn the trend bullish.")
    else:
        lines.append(f"Next resistance ~{a['res']:,.0f}.")
    a["read"] = "READ:  " + "\n".join(lines)
    return a


def _style():
    mc = mpf.make_marketcolors(up=GREEN, down="#e25149", edge="inherit",
                               wick="#8a8f98",
                               volume={"up": "#1e5c2f", "down": "#7c2a26"})
    return mpf.make_mpf_style(base_mpf_style="nightclouds", marketcolors=mc,
                              facecolor=BG, figcolor=BG, edgecolor=GRID,
                              gridcolor=GRID, gridstyle=":",
                              rc={"font.size": 11,
                                  "axes.labelcolor": GRAY,
                                  "xtick.color": GRAY, "ytick.color": GRAY})


def render(sym: str, df: pd.DataFrame, a: dict, stamp: str, out: Path):
    n = len(df)
    aps = []
    if a["s50"] is not None:
        aps.append(mpf.make_addplot(a["sma50"], color=SMA50_C, width=1.3))
    if a["s200"] is not None:
        aps.append(mpf.make_addplot(a["sma200"], color=SMA200_C, width=1.3))

    fig, axes = mpf.plot(df, type="candle", volume=True, addplot=aps or None,
                         style=_style(), figsize=(18, 6.8),
                         panel_ratios=(4, 1), datetime_format="%b",
                         xrotation=0, returnfig=True,
                         scale_padding={"top": 2.2, "right": 0.4})
    ax, axv = axes[0], axes[2]
    # mplfinance places axes absolutely; reposition them to fill the figure
    for axx in axes[:2]:
        axx.set_position([0.030, 0.265, 0.930, 0.655])
    for axx in axes[2:4]:
        axx.set_position([0.030, 0.070, 0.930, 0.165])
    ax.yaxis.tick_right()
    ax.set_ylabel("")
    axv.set_ylabel("Volume  x10^6", color=GRAY, fontsize=9)

    # one tick at each month boundary (mpf's auto ticks can skip months)
    month_idx = [i for i in range(1, n) if df.index[i].month != df.index[i - 1].month]
    for axx in (ax, axv):
        axx.set_xticks(month_idx)
        axx.set_xticklabels([df.index[i].strftime("%b") for i in month_idx])

    dashed = (0, (6, 4))
    trans_tag = mtrans.blended_transform_factory(ax.transAxes, ax.transData)
    last = df.iloc[-1]

    # ---------- header (finviz-like) ----------
    fig.text(0.012, 0.965, sym, fontsize=22, fontweight="bold", color=WHITE)
    fig.text(0.075, 0.965,
             f"{df.index[-1]:%b %d}   O:{last['Open']:,.2f}  H:{last['High']:,.2f}  "
             f"L:{last['Low']:,.2f}  C:{last['Close']:,.2f}   Vol:{last['Volume']/1e6:,.2f}M",
             fontsize=12, color=WHITE, va="bottom")
    prev = float(df["Close"].iloc[-2]) if n > 1 else float(last["Close"])
    chg = float(last["Close"]) - prev
    chg_txt = f"{chg:+,.2f} ({chg/prev:+.2%})"
    if a["ah"]:
        ah_pct, ah_price = a["ah"]
        chg_txt += f"   AH: {ah_price - a['close']:+,.2f} ({ah_pct:+.2f}%)"
    fig.text(0.79, 0.965, chg_txt, fontsize=13, fontweight="bold",
             color=RED if chg < 0 else GREEN, ha="right", va="bottom")
    fig.text(0.955, 0.965, f"Generated {stamp}", fontsize=10, color=GRAY,
             ha="right", va="bottom")

    # ---------- price tags on the right axis ----------
    ax.text(1.003, a["close"], f"{a['close']:,.2f}", transform=trans_tag,
            fontsize=10, fontweight="bold", color="black", va="center",
            clip_on=False, bbox=dict(boxstyle="square,pad=0.25", fc=YELLOW, ec="none"))
    if a["ah"]:
        ah_pct, ah_price = a["ah"]
        ax.text(1.003, ah_price, f"{ah_price:,.2f}", transform=trans_tag,
                fontsize=10, fontweight="bold", color="black", va="center",
                clip_on=False, bbox=dict(boxstyle="square,pad=0.25", fc=PURPLE, ec="none"))
        ax.hlines(ah_price, n * 0.62, n - 1, colors=PURPLE,
                  linestyles=dashed, lw=1.6)

    # ---------- support / resistance dashed lines ----------
    ax.hlines(a["sup_near"], max(a["i_sup_near"] - 25, 0), n - 1,
              colors=YELLOW, linestyles=dashed, lw=2)
    if a["sup_major"]:
        ax.hlines(a["sup_major"], max(a["i_sup_major"] - 25, 0),
                  min(a["i_sup_major"] + 90, n - 1),
                  colors=ORANGE, linestyles=dashed, lw=2)

    # ---------- stacked callout labels with dashed leaders ----------
    def callout(row, text, color, target=None):
        pos = (0.60, 0.975 - row * 0.088)
        box = dict(boxstyle="square,pad=0.35", fc=BOX_BG, ec=color, lw=1.2)
        if target is None:
            ax.text(*pos, text, transform=ax.transAxes, fontsize=11.5,
                    fontweight="bold", color=color, ha="left", va="top", bbox=box)
        else:
            ax.annotate(text, xytext=pos, textcoords="axes fraction",
                        xy=target, xycoords="data", fontsize=11.5,
                        fontweight="bold", color=color, ha="left", va="top",
                        annotation_clip=False, bbox=box,
                        arrowprops=dict(arrowstyle="-", color=color,
                                        linestyle="--", lw=1.1, alpha=0.85))

    row = 0
    if a["spike"]:
        i_s, p_s, fade = a["spike"]
        callout(row, f"Spike to ~{p_s:,.0f} faded hard: {fade:+.0%}", RED, (i_s, p_s))
        row += 1
        # circle the spike + fade arrow into the latest close
        ylo, yhi = ax.get_ylim()
        ax.add_patch(Ellipse((i_s, p_s), width=n * 0.045, height=(yhi - ylo) * 0.09,
                             fill=False, color=RED, lw=2.5))
        ax.annotate("", xy=(n - 1, a["close"]), xycoords="data",
                    xytext=(i_s, p_s * 0.99), textcoords="data",
                    arrowprops=dict(arrowstyle="-|>", color=RED, lw=2.5))
    if a["sma200_note"]:
        kind, i_n, text = a["sma200_note"]
        callout(row, text, CYAN, (i_n, float(a["sma200"].iloc[i_n])))
        row += 1
    if a["ah"]:
        ah_pct, ah_price = a["ah"]
        ref = (f" (< SMA50 ~{a['s50']:,.0f})" if a["s50"] and ah_price < a["s50"] < a["close"]
               else "")
        callout(row, f"AFTER-HOURS {ah_pct:+.1f}% to {ah_price:,.2f}{ref}",
                PURPLE, (n - 1, ah_price))
        row += 1
    sup_txt = f"Support ~{a['sup_near']:,.0f}"
    if a["ah"] and abs(a["ah"][1] - a["sup_near"]) / a["sup_near"] < 0.03:
        sup_txt += " - AH sits right on it"
    callout(row, sup_txt, YELLOW, (a["i_sup_near"], a["sup_near"]))
    row += 1
    if a["sup_major"]:
        callout(row, f"Next support ~{a['sup_major']:,.0f}", ORANGE,
                (a["i_sup_major"], a["sup_major"]))
        row += 1
    if a["vol_surge"]:
        callout(row, "Volume surge", RED)
        vmax = float(df["Volume"].iloc[-12:].max())
        axv.add_patch(Rectangle((n - 12.5, 0), 12.5, vmax * 1.08,
                                fill=False, color=RED, lw=2))
        row += 1

    # ---------- major leg label (left side) ----------
    if a["leg"]:
        kind, i0, i1, pct = a["leg"]
        txt = (f"{df.index[i0]:%b} -> {df.index[i1]:%b}: "
               f"{pct:+.0%} {'bear' if kind == 'bear' else 'bull'} leg")
        mid = (i0 + i1) // 2
        y_mid = float(df["High"].iloc[i0]) if kind == "bear" else float(df["Low"].iloc[i0])
        ax.annotate(txt, xytext=(0.06, 0.975), textcoords="axes fraction",
                    xy=(i0, y_mid), xycoords="data",
                    fontsize=11.5, fontweight="bold",
                    color=RED if kind == "bear" else GREEN, ha="left", va="top",
                    bbox=dict(boxstyle="square,pad=0.35", fc=BOX_BG,
                              ec=RED if kind == "bear" else GREEN, lw=1.2),
                    arrowprops=dict(arrowstyle="-", linestyle="--", lw=1.1,
                                    color=RED if kind == "bear" else GREEN, alpha=0.85))

    # ---------- READ box ----------
    ax.text(0.235, 0.975, a["read"], transform=ax.transAxes, fontsize=10.5,
            fontweight="bold", color=WHITE, ha="left", va="top",
            bbox=dict(boxstyle="square,pad=0.5", fc=(0.04, 0.055, 0.1, 0.92),
                      ec=WHITE, lw=1.2))

    fig.savefig(out, dpi=100, facecolor=BG)
    plt.close(fig)


def build_charts(symbols: list, ah_stats: dict) -> dict:
    """Render an annotated chart per symbol. Returns {sym: sentiment_label}."""
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        import zoneinfo
        now = dt.datetime.now(zoneinfo.ZoneInfo("America/New_York"))
    except Exception:
        now = dt.datetime.utcnow() - dt.timedelta(hours=5)
    stamp = now.strftime("%m-%d-%y %H:%M ET")

    ohlcv = fetch_ohlcv(symbols)
    sentiments = {}
    for sym in symbols:
        if sym not in ohlcv:
            print(f"  {sym}: no OHLCV data — skipped")
            continue
        try:
            full = ohlcv[sym]
            sma50 = full["Close"].rolling(50).mean()
            sma200 = full["Close"].rolling(200).mean()
            disp = full.iloc[-DISPLAY_BARS:]
            a = analyze(disp, sma50.loc[disp.index], sma200.loc[disp.index],
                        ah_stats.get(sym))
            sentiments[sym] = a["sentiment"]
            render(sym, disp, a, stamp, CHARTS_DIR / f"{sym}.png")
        except Exception as exc:
            print(f"  {sym}: chart render failed ({exc}) — keeping previous image")
    print(f"  Charts written: {len(list(CHARTS_DIR.glob('*.png')))} in {CHARTS_DIR}")
    return sentiments
