"""Fetch FinViz daily charts for all tickers, compute sentiment from cached
prices, and annotate each image with a sentiment banner and key levels.

Output: docs/charts/<TICKER>.png — overwritten on every run, never committed
to git (served to GitHub Pages via the deploy artifact). Each image carries a
"Generated <time> ET" stamp so staleness is visible.
"""
import datetime as dt
import io
import urllib.request
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

CHART_URL = ("https://charts-node.finviz.com/chart.ashx?cs=l&t={ticker}&tf=d"
             "&s=linear&ct=candle_stick&tm=d"
             "&o[0][ot]=sma&o[0][op]=50&o[1][ot]=sma&o[1][op]=200")

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/125.0.0.0 Safari/537.36"),
    "Referer": "https://finviz.com/",
    "Accept": "image/avif,image/webp,image/png,image/*,*/*;q=0.8",
}

CHARTS_DIR = Path("docs") / "charts"

SENTIMENT_COLORS = {
    "BULLISH":       (0, 160, 60),
    "LEANS BULLISH": (0, 110, 45),
    "NEUTRAL":       (90, 90, 90),
    "LEANS BEARISH": (160, 35, 35),
    "BEARISH":       (210, 20, 20),
}

_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\arialbd.ttf",                                  # Windows
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",           # Ubuntu CI
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]


def _font(size: int):
    for path in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def analyze(series: pd.Series) -> dict:
    """Sentiment + key levels from a daily close series (~400 calendar days)."""
    s = series.dropna()
    close = float(s.iloc[-1])
    sma50 = float(s.rolling(50).mean().iloc[-1]) if len(s) >= 50 else close
    sma200 = float(s.rolling(200).mean().iloc[-1]) if len(s) >= 200 else sma50
    r21 = close / float(s.iloc[-22]) - 1 if len(s) >= 22 else 0.0
    r63 = close / float(s.iloc[-64]) - 1 if len(s) >= 64 else 0.0
    recent = s.iloc[-63:]
    support, resistance = float(recent.min()), float(recent.max())

    score = (1 if close > sma50 else -1) \
          + (1 if close > sma200 else -1) \
          + (1 if sma50 > sma200 else -1) \
          + (1 if r21 > 0.02 else -1 if r21 < -0.02 else 0) \
          + (1 if r63 > 0.05 else -1 if r63 < -0.05 else 0)

    if score >= 3:
        label = "BULLISH"
    elif score >= 1:
        label = "LEANS BULLISH"
    elif score <= -3:
        label = "BEARISH"
    elif score <= -1:
        label = "LEANS BEARISH"
    else:
        label = "NEUTRAL"

    trend = (f"{'above' if close > sma50 else 'below'} SMA50 "
             f"{sma50:,.0f} / {'above' if close > sma200 else 'below'} "
             f"SMA200 {sma200:,.0f}")
    return {
        "label": label,
        "close": close,
        "trend": trend,
        "r21": r21,
        "r63": r63,
        "support": support,
        "resistance": resistance,
    }


def _fetch_png(ticker: str) -> Image.Image:
    url = CHART_URL.format(ticker=ticker)
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = resp.read()
        ctype = resp.headers.get("Content-Type", "")
    if not data or "image" not in ctype:
        raise RuntimeError(f"unexpected response: {ctype}, {len(data)} bytes")
    return Image.open(io.BytesIO(data)).convert("RGB")


def annotate(chart: Image.Image, ticker: str, info: dict, stamp: str) -> Image.Image:
    """Extend the canvas with a banner strip above the chart (never covers it)."""
    banner_h = 56
    out = Image.new("RGB", (chart.width, chart.height + banner_h), (13, 17, 23))
    out.paste(chart, (0, banner_h))
    d = ImageDraw.Draw(out)

    color = SENTIMENT_COLORS[info["label"]]
    f_big, f_small = _font(20), _font(13)

    d.rectangle([0, 0, chart.width, banner_h], fill=(13, 17, 23))
    d.text((12, 6), ticker, font=f_big, fill=(235, 240, 245))
    name_w = d.textlength(ticker, font=f_big)

    badge_text = info["label"]
    bx = 12 + name_w + 14
    bw = d.textlength(badge_text, font=f_small)
    d.rounded_rectangle([bx, 8, bx + bw + 16, 28], radius=10, fill=color)
    d.text((bx + 8, 11), badge_text, font=f_small, fill=(255, 255, 255))

    stamp_text = f"Generated {stamp}"
    d.text((chart.width - 12 - d.textlength(stamp_text, font=f_small), 11),
           stamp_text, font=f_small, fill=(139, 148, 158))

    detail = (f"Close {info['close']:,.0f}   {info['trend']}   "
              f"1M {info['r21']:+.1%}  3M {info['r63']:+.1%}   "
              f"Support ~{info['support']:,.0f}  Resistance ~{info['resistance']:,.0f}")
    d.text((12, 34), detail, font=f_small, fill=(200, 208, 215))
    return out


def build_charts(prices: pd.DataFrame, symbols: list) -> dict:
    """Fetch + annotate a chart per symbol. Returns {sym: sentiment_label}."""
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        import zoneinfo
        now = dt.datetime.now(zoneinfo.ZoneInfo("America/New_York"))
    except Exception:
        now = dt.datetime.utcnow() - dt.timedelta(hours=5)
    stamp = now.strftime("%m-%d-%y %H:%M ET")

    sentiments = {}
    for sym in symbols:
        if sym not in prices.columns:
            continue
        info = analyze(prices[sym])
        sentiments[sym] = info["label"]
        try:
            chart = _fetch_png(sym)
        except Exception as exc:
            print(f"  {sym}: chart fetch failed ({exc}) — keeping previous image")
            continue
        annotate(chart, sym, info, stamp).save(CHARTS_DIR / f"{sym}.png")
    print(f"  Charts written: {len(list(CHARTS_DIR.glob('*.png')))} in {CHARTS_DIR}")
    return sentiments
