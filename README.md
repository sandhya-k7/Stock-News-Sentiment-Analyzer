# Stock News Sentiment Analyzer
Educational project, not financial advice.

    pip install -r requirements.txt
    export NEWSDATA_API_KEY=...        # see .env.example
    streamlit run app.py

Pipeline: NewsData.io news -> VADER/FinBERT score -> align to US trading day (after 16:00 ET rolls forward)
-> daily features -> lag correlations, Granger tests -> walk-forward models (price-only vs price+sentiment).

Notes
- NewsData.io's free tier only covers recent news; the archive endpoint is paid. For 1-2 years of history,
  swap fetch_news for Finnhub, Alpha Vantage News, or the FNSPID Kaggle dataset (keep the same columns).
- Features at day t predict return at t+1 (no look-ahead); splits are chronological.

## Added modules
- `src/sentiment_finetune.py` - `make finetune` (GPU/Colab); saves `models/finbert-ft`, auto-listed in the Model comparison tab
- `src/backtest.py`, `src/event_study.py`, SHAP importance (`src/model.py`) - Prediction tab
- `src/cache.py` - SQLite score cache; `tests/` - `make test`
- `src/data.py::fetch_news_finnhub` - ~1 year of history on Finnhub's free tier (`FINNHUB_API_KEY`)
- `configs/config.yaml` - default ticker, query and news source
