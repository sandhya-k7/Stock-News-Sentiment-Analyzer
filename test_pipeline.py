import numpy as np
import pandas as pd
from src.data import assign_trading_day
from src.features import build_features

DAYS = pd.bdate_range("2024-01-02", "2024-03-29")


def _news(ts):
    return pd.DataFrame({"published": pd.to_datetime(ts, utc=True), "score": [0.5] * len(ts)})


def test_after_close_goes_to_next_day():
    out = assign_trading_day(_news(["2024-01-03 22:00"]), DAYS)  # 17:00 ET Wed
    assert out["day"].iloc[0] == pd.Timestamp("2024-01-04")


def test_weekend_goes_to_monday():
    out = assign_trading_day(_news(["2024-01-06 15:00"]), DAYS)  # Saturday
    assert out["day"].iloc[0] == pd.Timestamp("2024-01-08")


def _prices(seed=0, tail_scale=1.0):
    rng = np.random.default_rng(seed)
    c = 100 * np.cumprod(1 + rng.normal(0, .01, len(DAYS)))
    c[60:] *= tail_scale
    return pd.DataFrame({"Close": c, "Volume": 1e6 + rng.normal(0, 1e4, len(DAYS))}, index=DAYS)


def test_features_do_not_leak_future():
    news = assign_trading_day(_news(["2024-01-10 15:00", "2024-02-01 15:00"]), DAYS)
    a, b = build_features(_prices(), news), build_features(_prices(tail_scale=1.5), news)
    cols = ["ret", "vol5", "rsi", "ma_gap", "sent_mean", "sent_5d"]
    pd.testing.assert_frame_equal(a.iloc[:59][cols], b.iloc[:59][cols])  # past rows unaffected by future prices


def test_target_is_next_day_return():
    f = build_features(_prices(), assign_trading_day(_news(["2024-01-10 15:00"]), DAYS))
    assert np.isclose(f["next_ret"].iloc[10], f["ret"].iloc[11])
