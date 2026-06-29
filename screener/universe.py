"""Fetch and cache the S&P 500 ticker universe from Wikipedia."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_CACHE_PATH = Path.home() / ".tradingagents" / "cache" / "sp500_tickers.csv"
_CACHE_TTL_HOURS = 24  # refresh universe once per day


def get_sp500_tickers(force_refresh: bool = False) -> list[str]:
    """Return S&P 500 ticker symbols, using a daily local cache."""
    if not force_refresh and _cache_is_fresh():
        tickers = pd.read_csv(_CACHE_PATH)["ticker"].tolist()
        logger.info("Loaded %d tickers from cache", len(tickers))
        return tickers

    tickers = _fetch_from_wikipedia()
    _save_cache(tickers)
    return tickers


def _cache_is_fresh() -> bool:
    if not _CACHE_PATH.exists():
        return False
    age_hours = (time.time() - _CACHE_PATH.stat().st_mtime) / 3600
    return age_hours < _CACHE_TTL_HOURS


def _fetch_from_wikipedia() -> list[str]:
    logger.info("Fetching S&P 500 constituents from Wikipedia…")
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    # Wikipedia requires a User-Agent header — anonymous requests are blocked
    headers = {"User-Agent": "TradingAgents/1.0 (research project; python-requests)"}
    try:
        html = requests.get(url, timeout=15, headers=headers).text
        tables = pd.read_html(html)
        tickers = tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
        logger.info("Fetched %d tickers from Wikipedia", len(tickers))
        return tickers
    except Exception as exc:
        logger.warning("Wikipedia fetch failed (%s); falling back to hardcoded list", exc)
        return _fallback_tickers()


def _save_cache(tickers: list[str]) -> None:
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ticker": tickers}).to_csv(_CACHE_PATH, index=False)


def _fallback_tickers() -> list[str]:
    """Minimal hardcoded list used only when Wikipedia is unreachable."""
    return [
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "BRK-B",
        "UNH", "LLY", "JPM", "V", "XOM", "MA", "AVGO", "PG", "HD", "COST",
        "MRK", "ABBV", "CVX", "KO", "PEP", "ADBE", "WMT", "BAC", "CRM",
        "TMO", "ACN", "MCD", "CSCO", "ABT", "LIN", "DHR", "AMD", "TXN",
        "NEE", "PM", "NFLX", "RTX", "QCOM", "HON", "AMGN", "IBM", "GE",
        "LOW", "CAT", "SPGI", "BLK", "INTU",
    ]
