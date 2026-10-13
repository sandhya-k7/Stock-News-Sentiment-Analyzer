"""Run: streamlit run app.py   (needs NEWSDATA_API_KEY in the environment)"""
import datetime as dt
import yaml
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.analysis import granger, lag_correlations
from src.backtest import backtest
from src.event_study import event_study
from src.data import assign_trading_day, fetch_news, fetch_news_finnhub, fetch_prices
from src.features import build_features
from src.model import compare_models, shap_importance
from src.sentiment import FinBert, Vader, score_articles

CFG = yaml.safe_load(open("configs/config.yaml"))
st.set_page_config(page_title="Stock News Sentiment Analyzer", layout="wide")
st.title("Stock News Sentiment Analyzer")
st.caption("Educational project. Not financial advice.")

with st.sidebar:
    ticker = st.text_input("Ticker", CFG["ticker"]).upper()
    query = st.text_input("News query (NewsData only)", CFG["query"])
    start = st.date_input("Start", dt.date.today() - dt.timedelta(days=365))
    end = st.date_input("End", dt.date.today())
    engine_name = st.selectbox("Sentiment model", ["FinBERT", "VADER"])
    source = st.selectbox("News source", ["Finnhub", "NewsData.io"], index=["Finnhub", "NewsData.io"].index(CFG["news_source"]))
    archive = st.checkbox("NewsData archive endpoint (paid plan)")
    max_lag = st.slider("Max lag (days)", 1, 10, 5)
    run = st.button("Run analysis", type="primary")


@st.cache_resource
def get_engine(name):
    return FinBert() if name == "FinBERT" else Vader()


@st.cache_data(show_spinner=False)
def load(ticker, query, start, end, engine_name, archive, source):
    if source == "Finnhub":
        news = fetch_news_finnhub(ticker, str(start), str(end))
    else:
        news = fetch_news(query, archive=archive, from_date=str(start), to_date=str(end))
    prices = fetch_prices(ticker, str(start), str(end))
    if news.empty or prices.empty:
        return news, prices, None
    news = assign_trading_day(score_articles(news, get_engine(engine_name)), prices.index)
    return news, prices, build_features(prices, news)


if run:
    try:
        with st.spinner("Fetching news, scoring sentiment, loading prices..."):
            news, prices, feats = load(ticker, query, start, end, engine_name, archive, source)
    except Exception as e:
        st.error(f"Failed: {e}")
        st.stop()
    if feats is None:
        st.warning("No data returned. Check the ticker, query, date range or API plan.")
        st.stop()
    days_with_news = int((feats["n_articles"] > 0).sum())
    if days_with_news < 60:
        st.warning(f"Only {days_with_news} days have news. Statistics below will be unreliable.")

    t1, t2, t3, t4, t5, t6 = st.tabs(["News", "Sentiment vs price", "Lag analysis", "Prediction", "Export", "Model comparison"])
    with t1:
        st.dataframe(news[["published", "title", "label", "score", "source_id", "url"]]
                     .sort_values("published", ascending=False), use_container_width=True)
    with t2:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Candlestick(x=prices.index, open=prices["Open"], high=prices["High"],
                                     low=prices["Low"], close=prices["Close"], name="Price"))
        fig.add_trace(go.Bar(x=feats.index, y=feats["sent_mean"], name="Daily sentiment", opacity=0.4),
                      secondary_y=True)
        fig.update_layout(xaxis_rangeslider_visible=False, height=550)
        st.plotly_chart(fig, use_container_width=True)
    with t3:
        lc = lag_correlations(feats, max_lag)
        st.plotly_chart(go.Figure(go.Bar(x=lc["lag"], y=lc["spearman"])).update_layout(
            title="Spearman corr(sentiment_t, return_t+lag)"), use_container_width=True)
        st.dataframe(lc)
        st.write("Granger causality (min p-value across lags):", granger(feats))
        st.caption("p < 0.05 is suggestive only; many lags are tested, so expect false positives.")
    with t4:
        st.write("Walk-forward (TimeSeriesSplit) next-day direction results")
        st.dataframe(compare_models(feats).round(3))
        st.caption("Compare 'price + sentiment' with 'price only'. A gain of 1-2 points is within noise.")
    with t4:
        st.subheader("Backtest (long/flat, 0.1% cost) vs buy & hold")
        for flag, nm in [(True, "with sentiment"), (False, "price only")]:
            st.write(nm); st.dataframe(backtest(feats, flag).round(3))
        st.subheader("Event study: avg abnormal return around extreme-sentiment days")
        st.line_chart(event_study(feats))
        st.subheader("Feature importance (mean |SHAP|, random forest)")
        try:
            st.bar_chart(shap_importance(feats))
        except Exception as e:
            st.info(f"SHAP unavailable: {e}")
    with t5:
        st.download_button("Download daily features (CSV)", feats.to_csv().encode(), f"{ticker}_features.csv")
else:
    st.info("Set options in the sidebar and click **Run analysis**.")

    with t6:
        st.write("Evaluates engines on labeled financial tweets (first run downloads data/models).")
        if st.button("Run comparison"):
            import os
            from src.evaluate_sentiment import evaluate
            engines = {"VADER": Vader(), "FinBERT": FinBert()}
            if os.path.isdir("models/finbert-ft"):
                engines["FinBERT fine-tuned"] = FinBert("models/finbert-ft")
            tbl, cms = evaluate(engines)
            st.dataframe(tbl.round(3))
            for k, v in cms.items():
                st.write(k, "(rows=true, cols=predicted)"); st.dataframe(v)
