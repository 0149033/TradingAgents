"""
Two-pass screener for S&P 500 tickers.

Pass 1 — Fundamentals (Yahoo Finance cheat-sheet thresholds):
  P/E, revenue growth, profit margin, debt/equity, free cash flow.
  AVOID tickers (majorBreakdown OR below EMA200) are dropped early.

Pass 2 — Pine Script technicals (Stock Decision Engine v4.2):
  Trend (daily EMA + ADX + weekly EMA), breakout, EMA bounce,
  anti-chase, stop loss, 2R/3R targets.

Final ranking:
  BUY tickers (by fundamental score) → ACCUMULATE → WATCH
  AVOID tickers never reach the output.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import yfinance as yf

from .technicals import TechnicalSignal, TechParams, classify, DEFAULT_PARAMS
from .universe import get_sp500_tickers

logger = logging.getLogger(__name__)

# Signal priority for sorting (lower = higher priority)
_SIGNAL_PRIORITY = {"BUY": 0, "ACCUMULATE": 1, "WATCH": 2, "NEUTRAL": 3, "AVOID": 99}


@dataclass
class ScreenerResult:
    ticker: str
    fundamental_score: float
    signal: str                        # BUY | ACCUMULATE | WATCH | AVOID | NEUTRAL
    tech: TechnicalSignal

    # ── Fundamental fields ─────────────────────────────────────────────
    pe_ratio: Optional[float] = None
    revenue_growth: Optional[float] = None
    profit_margin: Optional[float] = None
    debt_to_equity: Optional[float] = None
    free_cash_flow: Optional[float] = None
    fcf_growing: Optional[bool] = None
    eps: Optional[float] = None

    reasons: list[str] = field(default_factory=list)

    # ── Risk levels (from Pine Script) ────────────────────────────────
    @property
    def entry(self) -> Optional[float]:
        return self.tech.entry

    @property
    def stop_loss(self) -> Optional[float]:
        return self.tech.stop_loss

    @property
    def target1(self) -> Optional[float]:
        return self.tech.target1

    @property
    def target2(self) -> Optional[float]:
        return self.tech.target2

    @property
    def risk_pct(self) -> Optional[float]:
        return self.tech.risk_pct

    def summary(self) -> str:
        parts = [f"signal={self.signal}", f"fscore={self.fundamental_score:.0f}"]
        if self.pe_ratio is not None:
            parts.append(f"P/E={self.pe_ratio:.0f}")
        if self.profit_margin is not None:
            parts.append(f"margin={self.profit_margin*100:.0f}%")
        if self.tech.rsi is not None:
            parts.append(f"RSI={self.tech.rsi:.0f}")
        if self.tech.adx is not None:
            parts.append(f"ADX={self.tech.adx:.0f}")
        return "  ".join(parts)


class Screener:
    """Screen the S&P 500 universe, return top N actionable tickers."""

    def __init__(
        self,
        top_n: int = 10,
        batch_size: int = 50,
        tech_params: TechParams = DEFAULT_PARAMS,
        include_watch: bool = True,
    ):
        self.top_n = top_n
        self.batch_size = batch_size
        self.tech_params = tech_params
        self.include_watch = include_watch

    def run(self, force_universe_refresh: bool = False) -> list[ScreenerResult]:
        tickers = get_sp500_tickers(force_refresh=force_universe_refresh)
        logger.info("Screening %d tickers (top_n=%d)…", len(tickers), self.top_n)

        results: list[ScreenerResult] = []
        for i in range(0, len(tickers), self.batch_size):
            batch = tickers[i : i + self.batch_size]
            results.extend(self._process_batch(batch))
            logger.info(
                "Processed %d/%d tickers",
                min(i + self.batch_size, len(tickers)),
                len(tickers),
            )

        # Drop AVOID; optionally drop WATCH
        keep_signals = {"BUY", "ACCUMULATE"}
        if self.include_watch:
            keep_signals.add("WATCH")
        results = [r for r in results if r.signal in keep_signals]

        # Sort: signal priority first, then fundamental score descending
        results.sort(
            key=lambda r: (_SIGNAL_PRIORITY.get(r.signal, 99), -r.fundamental_score)
        )

        top = results[: self.top_n]
        logger.info(
            "Top %d: %s",
            len(top),
            ", ".join(f"{r.ticker}[{r.signal}]({r.fundamental_score:.0f})" for r in top),
        )
        return top

    # ------------------------------------------------------------------ #

    def _process_batch(self, tickers: list[str]) -> list[ScreenerResult]:
        # Download 5 years of daily OHLCV — needed for weekly EMA200 (200 weeks ≈ 4 years)
        space = " ".join(tickers)
        try:
            data = yf.download(
                space,
                period="5y",
                interval="1d",
                progress=False,
                auto_adjust=True,
            )
        except Exception as exc:
            logger.warning("yfinance download failed for batch: %s", exc)
            return []

        # Normalise MultiIndex (multi-ticker) vs single-ticker download
        is_multi = isinstance(data.columns, pd.MultiIndex)

        results = []
        for ticker in tickers:
            try:
                ohlcv = _extract_ohlcv(data, ticker, is_multi)
                if ohlcv is None or len(ohlcv) < 60:
                    continue
                result = self._score_ticker(ticker, ohlcv)
                if result is not None:
                    results.append(result)
            except Exception as exc:
                logger.debug("Skipping %s: %s", ticker, exc)
        return results

    def _score_ticker(self, ticker: str, ohlcv: pd.DataFrame) -> Optional[ScreenerResult]:
        # ── Pass 2: Pine Script signal ────────────────────────────────
        tech = classify(ohlcv, self.tech_params)

        # Hard-skip AVOID here — no point fetching fundamentals
        if tech.signal == "AVOID":
            return None

        # ── Pass 1: Fundamentals ──────────────────────────────────────
        fund = _fetch_fundamentals(ticker)
        fscore, reasons = _score_fundamentals(fund)

        return ScreenerResult(
            ticker=ticker,
            fundamental_score=fscore,
            signal=tech.signal,
            tech=tech,
            pe_ratio=fund["pe_ratio"],
            revenue_growth=fund["revenue_growth"],
            profit_margin=fund["profit_margin"],
            debt_to_equity=fund["debt_to_equity"],
            free_cash_flow=fund["free_cash_flow"],
            fcf_growing=fund["fcf_growing"],
            eps=fund["eps"],
            reasons=reasons,
        )


# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #

def _extract_ohlcv(data: pd.DataFrame, ticker: str, is_multi: bool) -> Optional[pd.DataFrame]:
    """Pull OHLCV for one ticker from a potentially multi-ticker download."""
    if is_multi:
        if ticker not in data.columns.get_level_values(1):
            return None
        ohlcv = data.xs(ticker, axis=1, level=1)[["Open", "High", "Low", "Close", "Volume"]]
    else:
        if not all(c in data.columns for c in ["Open", "High", "Low", "Close", "Volume"]):
            return None
        ohlcv = data[["Open", "High", "Low", "Close", "Volume"]]

    return ohlcv.dropna(subset=["Close"]).sort_index()


def _fetch_fundamentals(ticker: str) -> dict:
    empty = {
        "eps": None, "revenue_growth": None, "profit_margin": None,
        "pe_ratio": None, "debt_to_equity": None,
        "free_cash_flow": None, "fcf_growing": None,
    }
    try:
        tk = yf.Ticker(ticker)
        info = tk.info
    except Exception:
        return empty

    eps        = _f(info.get("trailingEps"))
    pe         = _f(info.get("trailingPE"))
    margin     = _f(info.get("profitMargins"))
    rev_growth = _f(info.get("revenueGrowth"))
    de         = _f(info.get("debtToEquity"))
    if de is not None and de > 20:          # yfinance sometimes returns ×100
        de = de / 100.0

    fcf: Optional[float] = None
    fcf_growing: Optional[bool] = None
    try:
        cf = tk.cashflow
        if cf is not None and not cf.empty and "Free Cash Flow" in cf.index:
            row = cf.loc["Free Cash Flow"].dropna()
            if len(row) >= 1:
                fcf = float(row.iloc[0])
            if len(row) >= 2:
                fcf_growing = bool(row.iloc[0] > row.iloc[1])
    except Exception:
        pass
    if fcf is None:
        fcf = _f(info.get("freeCashflow"))

    return {
        "eps": eps, "revenue_growth": rev_growth, "profit_margin": margin,
        "pe_ratio": pe, "debt_to_equity": de,
        "free_cash_flow": fcf, "fcf_growing": fcf_growing,
    }


def _score_fundamentals(f: dict) -> tuple[float, list[str]]:
    """Yahoo Finance cheat-sheet thresholds → numeric score + reasons."""
    score = 0.0
    reasons: list[str] = []

    pe, rev, margin, de, fcf, fcf_growing = (
        f["pe_ratio"], f["revenue_growth"], f["profit_margin"],
        f["debt_to_equity"], f["free_cash_flow"], f["fcf_growing"],
    )

    # P/E ratio
    if pe is not None and pe > 0:
        if pe < 25:
            score += 20; reasons.append(f"P/E {pe:.0f} ✓")
        elif pe <= 40:
            score += 8;  reasons.append(f"P/E {pe:.0f} ok")
        else:
            score -= 10; reasons.append(f"P/E {pe:.0f} expensive")

    # Revenue growth
    if rev is not None:
        if rev >= 0.10:
            score += 20; reasons.append(f"rev +{rev*100:.0f}% strong")
        elif rev >= 0.05:
            score += 10; reasons.append(f"rev +{rev*100:.0f}% decent")
        elif rev >= 0:
            score += 3
        else:
            score -= 10; reasons.append(f"rev {rev*100:.0f}% shrinking")

    # Profit margin
    if margin is not None:
        if margin >= 0.15:
            score += 20; reasons.append(f"margin {margin*100:.0f}% strong")
        elif margin >= 0.05:
            score += 10; reasons.append(f"margin {margin*100:.0f}% decent")
        else:
            score -= 10; reasons.append(f"margin {margin*100:.0f}% thin")

    # Debt-to-equity
    if de is not None and de >= 0:
        if de < 1.0:
            score += 15; reasons.append(f"D/E {de:.2f} healthy")
        elif de <= 2.0:
            score += 5;  reasons.append(f"D/E {de:.2f} watch")
        else:
            score -= 15; reasons.append(f"D/E {de:.2f} risky")

    # Free cash flow
    if fcf is not None:
        if fcf > 0 and fcf_growing:
            score += 20; reasons.append("FCF +growing")
        elif fcf > 0:
            score += 10; reasons.append("FCF positive")
        else:
            score -= 20; reasons.append("FCF negative")

    return score, reasons


def _f(val) -> Optional[float]:
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None
