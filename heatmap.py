import json as _json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf
import plotly.graph_objects as go

from charts import build_charts
from tickers import tickers

PERIODS = [1, 7, 15, 30, 60, 90, 180, 270, 365]
PERIOD_LABELS = ["1D", "7D", "15D", "1M", "2M", "3M", "6M", "9M", "12M"]
CACHE_PATH = Path("prices_cache.parquet")
HISTORY_DAYS = 400


def _download(symbols: list, **kwargs) -> pd.DataFrame:
    data = yf.download(symbols, auto_adjust=True, progress=False, **kwargs)
    prices = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data
    prices.columns = [str(c) for c in prices.columns]
    return prices


def load_prices(symbols: list) -> pd.DataFrame:
    cache = pd.read_parquet(CACHE_PATH) if CACHE_PATH.exists() else pd.DataFrame()
    today = pd.Timestamp.today().normalize()

    # Symbols not yet in cache need a full history pull
    new_syms = [s for s in symbols if s not in cache.columns]
    if new_syms:
        print(f"  New symbols (fetching {HISTORY_DAYS}d): {new_syms}")
        fresh = _download(new_syms, period=f"{HISTORY_DAYS}d")
        cache = cache.join(fresh, how="outer") if not cache.empty else fresh

    # Bring existing symbols up to date if cache is stale
    last_date = cache.index[-1] if not cache.empty else None
    if last_date is None or last_date.date() < today.date():
        if last_date is not None:
            fetch_days = (today - last_date).days + 3
            print(f"  Cache last updated {last_date.date()} — fetching {fetch_days}d delta ...")
            delta = _download(symbols, period=f"{fetch_days}d")
            before = cache[cache.index < delta.index[0]]
            cache = pd.concat([before, delta])
            cache = cache[~cache.index.duplicated(keep="last")].sort_index()
        else:
            print(f"  No cache — fetching {HISTORY_DAYS}d ...")
            cache = _download(symbols, period=f"{HISTORY_DAYS}d")
    else:
        print(f"  Cache is current ({last_date.date()}) — skipping download.")

    # Trim to keep file lean
    cache = cache[cache.index >= today - pd.Timedelta(days=HISTORY_DAYS)]
    cache.to_parquet(CACHE_PATH)
    print(f"  Cache saved: {len(cache)} rows x {len(cache.columns)} symbols")
    return cache[[s for s in symbols if s in cache.columns]]


def period_stats(prices: pd.DataFrame, calendar_days: int) -> dict:
    """Returns {sym: (pct, start_price, end_price)} for each symbol."""
    result = {}
    for sym in prices.columns:
        series = prices[sym].dropna()
        if len(series) < 2:
            result[sym] = (0.0, 0, 0)
            continue
        cutoff = series.index[-1] - pd.Timedelta(days=calendar_days)
        past = series[series.index <= cutoff]
        if past.empty:
            result[sym] = (0.0, 0, 0)
        else:
            start = int(round(past.iloc[-1]))
            end = int(round(series.iloc[-1]))
            pct = round((end - start) / start * 100, 2)
            result[sym] = (pct, start, end)
    return result


def afterhours_stats(symbols: list) -> dict:
    """Returns {sym: (pct, reg_close, latest_price)} — last regular-session price
    vs the latest extended-hours (pre-market or after-hours) price.

    Fetches 2 days so that during pre-market (e.g. 6AM PT) the baseline is the
    previous session's close rather than today's still-empty regular session."""
    try:
        data = yf.download(symbols, period="2d", interval="1m",
                           prepost=True, auto_adjust=True, progress=False)
        prices = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data
        prices.columns = [str(c) for c in prices.columns]
    except Exception:
        return {s: (0.0, 0, 0) for s in symbols}

    result = {}
    for sym in symbols:
        if sym not in prices.columns:
            result[sym] = (0.0, 0, 0)
            continue
        series = prices[sym].dropna()
        if series.empty:
            result[sym] = (0.0, 0, 0)
            continue

        try:
            idx_et = series.index.tz_convert("America/New_York")
            open_min, close_min = 9 * 60 + 30, 16 * 60
        except Exception:
            idx_et = series.index.tz_convert("UTC")
            open_min, close_min = 14 * 60 + 30, 21 * 60  # 9:30/16:00 ET in UTC (EDT)
        minutes = idx_et.hour * 60 + idx_et.minute
        regular = series[(minutes >= open_min) & (minutes <= close_min)]

        if regular.empty:
            result[sym] = (0.0, 0, 0)
            continue

        reg_close = regular.iloc[-1]
        ah_price = series.iloc[-1]

        if reg_close == 0:
            result[sym] = (0.0, 0, 0)
            continue

        pct = round((ah_price - reg_close) / reg_close * 100, 2)
        result[sym] = (pct, int(round(reg_close)), int(round(ah_price)))

    return result


def fetch_news(symbols: list, max_articles: int = 3) -> dict:
    """Returns {sym: [{"title", "url", "publisher"}, ...]} — top news per ticker."""
    news = {}
    for sym in symbols:
        articles = []
        try:
            items = yf.Ticker(sym).news or []
        except Exception:
            items = []
        for item in items:
            # yfinance >= 0.2.50 nests fields under "content"; older versions are flat
            content = item.get("content", item)
            title = content.get("title")
            url = (
                (content.get("canonicalUrl") or {}).get("url")
                or (content.get("clickThroughUrl") or {}).get("url")
                or item.get("link")
            )
            publisher = (
                (content.get("provider") or {}).get("displayName")
                or item.get("publisher")
                or ""
            )
            if title and url:
                articles.append({"title": title, "url": url, "publisher": publisher})
            if len(articles) >= max_articles:
                break
        news[sym] = articles
    return news


# Each list index = 5% bracket: [0-5%), [5-10%), [10-15%), [15-20%), [20%+)
# Small moves = light, large moves = dark
_GREEN_STEPS = [
    "rgb(100,220,100)",
    "rgb(0,185,65)",
    "rgb(0,140,45)",
    "rgb(0,100,30)",
    "rgb(0,65,18)",
]
_RED_STEPS = [
    "rgb(235,90,90)",
    "rgb(210,30,30)",
    "rgb(165,10,10)",
    "rgb(115,0,0)",
    "rgb(70,0,0)",
]


def pct_to_color(pct: float) -> str:
    """Color intensity steps every 5%; up to 5 steps each direction."""
    if pct == 0:
        return "rgb(60,60,60)"
    step = min(int(abs(pct) / 5), len(_GREEN_STEPS) - 1)
    return _GREEN_STEPS[step] if pct > 0 else _RED_STEPS[step]


# Clusters ordered best → worst; each entry: (display name, threshold test, header color)
CLUSTERS = [
    ("Strong Gain",  lambda p: p >  5,        "rgb(0,160,60)"),
    ("Gain",         lambda p: 1 < p <= 5,    "rgb(0,110,45)"),
    ("Flat",         lambda p: -1 <= p <= 1,  "rgb(55,55,55)"),
    ("Loss",         lambda p: -5 <= p < -1,  "rgb(160,35,35)"),
    ("Strong Loss",  lambda p: p < -5,        "rgb(210,20,20)"),
]


def _cluster_name(pct: float) -> str:
    for name, test, _ in CLUSTERS:
        if test(pct):
            return name
    return "Flat"


HEADER_WEIGHT = 0.6  # area reserved for each cluster header bar


def build_trace(syms: list, stats: dict, label: str, visible: bool) -> go.Treemap:
    pcts   = {s: stats.get(s, (0.0, 0, 0))[0] for s in syms}
    starts = {s: stats.get(s, (0.0, 0, 0))[1] for s in syms}
    ends   = {s: stats.get(s, (0.0, 0, 0))[2] for s in syms}

    active = [c for c, _, _ in CLUSTERS if any(_cluster_name(pcts[s]) == c for s in syms)]
    cluster_color_map = {c: col for c, _, col in CLUSTERS}
    counts = {c: sum(1 for s in syms if _cluster_name(pcts[s]) == c) for c in active}

    # branchvalues="remainder": cluster visible header area = value - sum(children)
    #   → set cluster value = n_tickers + HEADER_WEIGHT so header gets HEADER_WEIGHT area
    node_labels  = active + syms
    node_parents = [""] * len(active) + [_cluster_name(pcts[s]) for s in syms]
    node_values  = [counts[c] + HEADER_WEIGHT for c in active] + [1] * len(syms)
    node_colors  = [cluster_color_map[c] for c in active] + \
                   [pct_to_color(pcts[s]) for s in syms]
    node_text    = [f"<b>{c}</b>  ({counts[c]})" for c in active] + \
                   [f"<b>{s}</b><br>{pcts[s]:+.2f}%" for s in syms]
    node_custom  = [[0, 0, 0]] * len(active) + \
                   [[pcts[s], starts[s], ends[s]] for s in syms]

    return go.Treemap(
        labels=node_labels,
        parents=node_parents,
        values=node_values,
        branchvalues="remainder",
        text=node_text,
        texttemplate="%{text}",
        textfont=dict(color="white", size=12),
        customdata=node_custom,
        marker=dict(
            colors=node_colors,
            line=dict(width=2, color="#0d1117"),
            pad=dict(t=18, l=2, r=2, b=2),
        ),
        hovertemplate=(
            "<b>%{label}</b><br>"
            f"Period: {label}<br>"
            "Start: $%{customdata[1]}<br>"
            "End:   $%{customdata[2]}<br>"
            "Return: %{customdata[0]:+.2f}%"
            "<extra></extra>"
        ),
        visible=visible,
        name=label,
    )


def build_heatmap(prices: pd.DataFrame) -> str:
    syms = [s for s in tickers if s in prices.columns]
    all_stats = {days: period_stats(prices, days) for days in PERIODS}

    print("Fetching pre/post-market data ...")
    ah_stats = afterhours_stats(syms)

    print("Fetching news ...")
    news_data = fetch_news(syms)

    print("Building annotated charts ...")
    sentiments = build_charts(prices, syms)

    all_labels = ["Pre/Post Market"] + PERIOD_LABELS
    all_stat_list = [ah_stats] + [all_stats[d] for d in PERIODS]

    # Serialize all stats for JS: {period: {sym: [pct, start, end]}}
    data_js = {
        lbl: {s: list(stat.get(s, (0.0, 0, 0))) for s in syms}
        for lbl, stat in zip(all_labels, all_stat_list)
    }

    # Initial Plotly figure (Pre/Post Market — first in dropdown)
    initial_trace = build_trace(syms, ah_stats, "Pre/Post Market", True)

    legend_shapes, legend_annotations = [], []
    stops = [-20, -15, -10, -5, 0, 5, 10, 15, 20]
    bar_x, bar_w, bar_y, bar_h = 0.35, 0.035, -0.04, 0.018
    for k, v in enumerate(stops):
        c = pct_to_color(v)
        legend_shapes.append(dict(
            type="rect", xref="paper", yref="paper",
            x0=bar_x + k * bar_w, x1=bar_x + (k + 1) * bar_w,
            y0=bar_y, y1=bar_y + bar_h,
            fillcolor=c, line_width=0,
        ))
        legend_annotations.append(dict(
            xref="paper", yref="paper",
            x=bar_x + k * bar_w + bar_w / 2, y=bar_y - 0.025,
            text=f"{v:+d}%", showarrow=False,
            font=dict(color="white", size=10),
        ))

    fig = go.Figure(data=[initial_trace])
    fig.update_layout(
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font=dict(color="white"),
        margin=dict(t=20, l=5, r=5, b=60),
        shapes=legend_shapes,
        annotations=legend_annotations,
    )

    chart_html = fig.to_html(full_html=False, include_plotlyjs="cdn",
                             config={"responsive": True})

    try:
        import zoneinfo
        et = zoneinfo.ZoneInfo("America/New_York")
        now_et_str = datetime.now(et).strftime("%m-%d-%y:%H-%M")
    except Exception:
        from datetime import timedelta
        now_et_str = (datetime.now(timezone.utc) - timedelta(hours=5)).strftime("%m-%d-%y:%H-%M")

    period_options = "\n      ".join(
        f'<option value="{lbl}">{lbl}</option>' for lbl in all_labels
    )
    ticker_options = (
        '<option value="__ALL__">All Tickers</option>\n      '
        + "\n      ".join(f'<option value="{s}">{s}</option>' for s in sorted(syms))
    )

    data_json  = _json.dumps(data_js)
    news_json  = _json.dumps(news_data)
    sent_json  = _json.dumps(sentiments)
    syms_json  = _json.dumps(syms)
    green_json = _json.dumps(_GREEN_STEPS)
    red_json   = _json.dumps(_RED_STEPS)

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Portfolio Heatmap</title>
<style>
* {{ box-sizing: border-box; }}
html, body {{ height: 100%; }}
body {{ background: #0d1117; color: #fff; margin: 0; display: flex; flex-direction: column;
       font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }}
h1 {{ text-align: center; font-size: 20px; font-weight: 600; padding: 12px 0 0; flex-shrink: 0; }}
.toolbar {{ display: flex; align-items: flex-end; justify-content: center;
            gap: 14px; padding: 8px 0 4px; flex-shrink: 0; }}
.toolbar label {{ display: block; font-size: 11px; color: #8b949e; margin-bottom: 4px; }}
select {{ background: #21262d; color: #fff; border: 1px solid #30363d;
          padding: 6px 14px; font-size: 13px; border-radius: 4px; cursor: pointer; outline: none; }}
.plotly-graph-div {{ flex: 1 1 auto; min-height: 0; }}
#news-panel {{ display: none; position: fixed; bottom: 14px; left: 50%;
               transform: translateX(-50%); z-index: 10;
               width: calc(100% - 32px); max-width: 720px;
               background: rgba(22, 27, 34, 0.97); border: 1px solid #30363d;
               border-radius: 8px; padding: 10px 16px 12px;
               box-shadow: 0 4px 16px rgba(0,0,0,0.5); }}
#news-panel b {{ display: block; font-size: 13px; margin-bottom: 6px; color: #8b949e; }}
#news-panel a {{ display: block; color: #58a6ff; font-size: 13px; text-decoration: none;
                 padding: 3px 0; }}
#news-panel a:hover {{ text-decoration: underline; }}
#news-panel .pub {{ color: #8b949e; font-size: 11px; }}
#news-panel {{ max-height: 85vh; overflow-y: auto; }}
#chart-section {{ margin-top: 10px; border-top: 1px solid #30363d; padding-top: 8px; }}
#chart-section .badge {{ float: right; color: #fff; padding: 1px 10px;
                         border-radius: 10px; font-size: 11px; }}
#chart-section img {{ width: 100%; border-radius: 6px; display: block; }}
</style>
</head>
<body>
<h1>Portfolio Heatmap</h1>
<div class="toolbar">
  <div>
    <label for="period-select">Duration</label>
    <select id="period-select" onchange="updateChart(this)">
      {period_options}
    </select>
  </div>
  <div>
    <label for="ticker-select">Ticker</label>
    <select id="ticker-select" onchange="updateChart(this)">
      {ticker_options}
    </select>
  </div>
  <div style="font-size:11px; color:#8b949e; padding-bottom:6px;">
    Updated<br>{now_et_str} EST
  </div>
</div>
{chart_html}
<div id="news-panel"></div>
<script>
const HEATMAP_DATA = {data_json};
const NEWS_DATA    = {news_json};
const SENTIMENTS   = {sent_json};
const CHART_VER    = "{now_et_str}";
const ALL_SYMS     = {syms_json};
const SENT_COLORS  = {{
  'BULLISH': 'rgb(0,160,60)', 'LEANS BULLISH': 'rgb(0,110,45)',
  'NEUTRAL': 'rgb(90,90,90)',
  'LEANS BEARISH': 'rgb(160,35,35)', 'BEARISH': 'rgb(210,20,20)',
}};
const GREEN_STEPS  = {green_json};
const RED_STEPS    = {red_json};

function pctToColor(pct) {{
  if (pct === 0) return 'rgb(60,60,60)';
  const step = Math.min(Math.floor(Math.abs(pct) / 5), GREEN_STEPS.length - 1);
  return pct > 0 ? GREEN_STEPS[step] : RED_STEPS[step];
}}

const CLUSTERS = [
  {{name:'Strong Gain', test: p => p > 5,             color:'rgb(0,160,60)'}},
  {{name:'Gain',        test: p => p > 1 && p <= 5,   color:'rgb(0,110,45)'}},
  {{name:'Flat',        test: p => p >= -1 && p <= 1, color:'rgb(55,55,55)'}},
  {{name:'Loss',        test: p => p >= -5 && p < -1, color:'rgb(160,35,35)'}},
  {{name:'Strong Loss', test: p => p < -5,            color:'rgb(210,20,20)'}},
];
const HEADER_WEIGHT = 0.6;

function clusterName(pct) {{
  for (const c of CLUSTERS) if (c.test(pct)) return c.name;
  return 'Flat';
}}

function buildTraceData(syms, periodData, period) {{
  const pcts = {{}}, starts = {{}}, ends = {{}};
  for (const s of syms) {{
    const d = periodData[s] || [0, 0, 0];
    pcts[s] = d[0]; starts[s] = d[1]; ends[s] = d[2];
  }}
  const active = CLUSTERS.filter(c => syms.some(s => clusterName(pcts[s]) === c.name));
  const counts = Object.fromEntries(
    active.map(c => [c.name, syms.filter(s => clusterName(pcts[s]) === c.name).length])
  );
  const single = syms.length === 1;
  const symText = s => {{
    const ret = (pcts[s] >= 0 ? '+' : '') + pcts[s].toFixed(2) + '%';
    if (!single) return '<b>' + s + '</b><br>' + ret;
    return '<b>' + s + '</b><br><br>' +
      'Period: ' + period + '<br>' +
      'Start: $' + starts[s] + '<br>' +
      'End: $' + ends[s] + '<br>' +
      'Return: ' + ret;
  }};
  return {{
    labels:     [...active.map(c => c.name), ...syms],
    parents:    [...active.map(() => ''),    ...syms.map(s => clusterName(pcts[s]))],
    values:     [...active.map(c => counts[c.name] + HEADER_WEIGHT), ...syms.map(() => 1)],
    colors:     [...active.map(c => c.color), ...syms.map(s => pctToColor(pcts[s]))],
    text: [
      ...active.map(c => '<b>' + c.name + '</b>  (' + counts[c.name] + ')'),
      ...syms.map(symText),
    ],
    customdata: [
      ...active.map(() => [0, 0, 0]),
      ...syms.map(s => [pcts[s], starts[s], ends[s]]),
    ],
  }};
}}

function updateNews(ticker) {{
  const panel = document.getElementById('news-panel');
  if (ticker === '__ALL__') {{
    panel.style.display = 'none';
    return;
  }}
  const articles = NEWS_DATA[ticker] || [];
  let html = '<b>Latest News — ' + ticker + '</b>';
  if (articles.length === 0) {{
    html += '<span class="pub">No recent articles found.</span>';
  }} else {{
    for (const a of articles.slice(0, 3)) {{
      html += '<a href="' + a.url + '" target="_blank" rel="noopener">' + a.title +
              (a.publisher ? ' <span class="pub">— ' + a.publisher + '</span>' : '') + '</a>';
    }}
  }}
  const sent = SENTIMENTS[ticker];
  if (sent) {{
    html += '<div id="chart-section">' +
      '<b>Chart Analysis — ' + ticker +
      '<span class="badge" style="background:' + SENT_COLORS[sent] + '">' + sent + '</span></b>' +
      '<img src="charts/' + ticker + '.png?v=' + encodeURIComponent(CHART_VER) + '"' +
      ' alt="' + ticker + ' chart" onerror="this.parentElement.style.display=\\'none\\'">' +
      '</div>';
  }}
  panel.innerHTML = html;
  panel.style.display = 'block';
}}

function updateChart(trigger) {{
  const period = document.getElementById('period-select').value;
  const ticker = document.getElementById('ticker-select').value;
  const symsToShow = ticker === '__ALL__' ? ALL_SYMS : [ticker];
  updateNews(ticker);
  const td = buildTraceData(symsToShow, HEATMAP_DATA[period], period);
  const div = document.querySelector('.plotly-graph-div');
  Plotly.react(div, [{{
    type: 'treemap',
    labels: td.labels,
    parents: td.parents,
    values: td.values,
    branchvalues: 'remainder',
    text: td.text,
    texttemplate: '%{{text}}',
    textfont: {{color: 'white', size: 12}},
    customdata: td.customdata,
    marker: {{
      colors: td.colors,
      line: {{width: 2, color: '#0d1117'}},
      pad: {{t: 18, l: 2, r: 2, b: 2}},
    }},
    hovertemplate: '<b>%{{label}}</b><br>Period: ' + period +
      '<br>Start: $%{{customdata[1]}}<br>End: $%{{customdata[2]}}<br>' +
      'Return: %{{customdata[0]:+.2f}}%<extra></extra>',
  }}], div.layout);
  if (trigger) trigger.focus();
}}

setTimeout(function() {{
  document.querySelector('.plotly-graph-div').on('plotly_click', function(data) {{
    if (!data.points.length) return;
    const label = data.points[0].label;
    if (!ALL_SYMS.includes(label)) return;
    const tickerSelect = document.getElementById('ticker-select');
    tickerSelect.value = label;
    updateChart(tickerSelect);
  }});
}}, 0);
</script>
</body>
</html>"""


if __name__ == "__main__":
    print(f"Loading prices for {len(tickers)} tickers ...")
    prices = load_prices(tickers)

    print("Building figure ...")
    html = build_heatmap(prices)
    Path("heatmap.html").write_text(html, encoding="utf-8")
    docs_index = Path("docs") / "index.html"
    docs_index.parent.mkdir(exist_ok=True)
    docs_index.write_text(html, encoding="utf-8")
    print("Saved -> heatmap.html and docs/index.html")
