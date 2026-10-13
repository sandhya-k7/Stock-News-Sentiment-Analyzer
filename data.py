"""News + price ingestion and news-to-trading-day alignment."""
import os
import pandas as pd
import requests
import yfinance as yf


def fetch_news(query: str, max_pages: int = 5, archive: bool = False,
               from_date: str | None = None, to_date: str | None = None) -> pd.DataFrame:
    """Fetch articles from NewsData.io. The archive endpoint needs a paid plan."""
    url = "https://newsdata.io/api/1/" + ("archive" if archive else "latest")
    params = {"apikey": os.environ["NEWSDATA_API_KEY"], "q": query, "language": "en"}
    if archive:
        params.update(from_date=from_date, to_date=to_date)
    rows, page = [], None
    for _ in range(max_pages):
        if page:
            params["page"] = page
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        d = r.json()
        rows += d.get("results", [])
        page = d.get("nextPage")
        if not page:
            break
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.rename(columns={"pubDate": "published", "link": "url"})
    df["published"] = pd.to_datetime(df["published"], utc=True)
    df["text"] = df["title"].fillna("") + ". " + df["description"].fillna("")
    return df[["title", "description", "text", "source_id", "published", "url"]].drop_duplicates("title")


def fetch_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    df.columns = df.columns.get_level_values(0)
    return df


def assign_trading_day(news: pd.DataFrame, trading_days) -> pd.DataFrame:
    """Map each article to a trading day. After-close (>=16:00 ET) and weekend news go to the next session."""
    t = news["published"].dt.tz_convert("US/Eastern")
    d = t.dt.normalize().dt.tz_localize(None)
    d = d.where(t.dt.hour < 16, d + pd.Timedelta(days=1))
    idx = pd.DatetimeIndex(trading_days)
    pos = idx.searchsorted(d)
    out = news.assign(day=[idx[p] if p < len(idx) else pd.NaT for p in pos])
    return out.dropna(subset=["day"])


def fetch_news_finnhub(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Finnhub company news (free tier ~1 year of history). Fetched in 7-day chunks to avoid result caps."""
    key = os.environ["FINNHUB_API_KEY"]
    rows = []
    for a in pd.date_range(start, end, freq="7D"):
        b = min(a + pd.Timedelta(days=6), pd.Timestamp(end))
        r = requests.get("https://finnhub.io/api/v1/company-news", timeout=30,
                         params={"symbol": ticker, "from": a.date(), "to": b.date(), "token": key})
        r.raise_for_status()
        rows += r.json()
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["published"] = pd.to_datetime(df["datetime"], unit="s", utc=True)
    df = df.rename(columns={"headline": "title", "summary": "description", "source": "source_id"})
    df["text"] = df["title"].fillna("") + ". " + df["description"].fillna("")
    return df[["title", "description", "text", "source_id", "published", "url"]].drop_duplicates("title")
