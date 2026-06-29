"""Run the TradingAgents pipeline for a list of screened tickers and dispatch alerts."""

from __future__ import annotations

import logging
import os
from datetime import datetime

import pytz

from integrations.telegram import SignalAlert, TelegramAlerter
from screener.screener import Screener, ScreenerResult
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph

logger = logging.getLogger(__name__)

ET = pytz.timezone("America/New_York")


def run_screening_and_analysis() -> None:
    """Full pipeline: screen → analyse → alert. Called by the scheduler."""
    trade_date = _today_et()
    logger.info("=== TradingAgents run starting  date=%s ===", trade_date)

    # ── 1. Screen ────────────────────────────────────────────────────
    top_n = int(os.environ.get("SCREENER_TOP_N", "10"))
    screener = Screener(top_n=top_n)
    try:
        top_tickers: list[ScreenerResult] = screener.run()
    except Exception as exc:
        logger.exception("Screener failed: %s", exc)
        return

    if not top_tickers:
        logger.warning("Screener returned no tickers — skipping analysis.")
        return

    # ── 2. Init shared resources ──────────────────────────────────────
    alerter = _build_alerter()
    config  = _build_config()
    # Use only market + fundamentals analysts to cut LLM calls by ~half
    analysts = tuple(
        os.environ.get("TRADINGAGENTS_ANALYSTS", "market,fundamentals").split(",")
    )
    ta      = TradingAgentsGraph(config=config, selected_analysts=analysts)
    alerts: list[SignalAlert] = []

    # ── 3. Analyse each ticker ────────────────────────────────────────
    for screened in top_tickers:
        ticker = screened.ticker
        logger.info(
            "Analysing %s  screener=%s  fscore=%.0f",
            ticker, screened.signal, screened.fundamental_score,
        )
        try:
            final_state, pipeline_signal = ta.propagate(ticker, trade_date)
        except Exception as exc:
            logger.exception("Pipeline failed for %s: %s", ticker, exc)
            if alerter:
                alerter.send_error(ticker, str(exc))
            continue

        alert = SignalAlert(
            ticker=ticker,
            screener_signal=screened.signal,
            pipeline_signal=pipeline_signal,
            trade_date=trade_date,
            fundamental_score=screened.fundamental_score,
            screener_reasons=screened.reasons,
            # Risk levels from Pine Script
            entry=screened.entry,
            stop_loss=screened.stop_loss,
            target1=screened.target1,
            target2=screened.target2,
            risk_pct=screened.risk_pct,
            # Technical context
            setup_score=screened.tech.setup_score,
            trigger_score=screened.tech.trigger_score,
            adx=screened.tech.adx,
            rsi=screened.tech.rsi,
            weekly_uptrend=screened.tech.weekly_uptrend,
            # TradingAgents pipeline
            investment_plan=final_state.get("investment_plan"),
            final_trade_decision=final_state.get("final_trade_decision"),
        )
        alerts.append(alert)

        if alerter:
            alerter.send_signal(alert)
        logger.info("%s  screener=%s  pipeline=%s", ticker, screened.signal, pipeline_signal)

    # ── 4. Send end-of-run summary ────────────────────────────────────
    if alerter:
        alerter.send_run_summary(alerts, trade_date)

    logger.info("=== Run complete — %d signals generated ===", len(alerts))


# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #

def _today_et() -> str:
    return datetime.now(ET).strftime("%Y-%m-%d")


def _build_alerter() -> TelegramAlerter | None:
    token   = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        logger.warning("Telegram not configured — alerts disabled.")
        return None
    return TelegramAlerter(bot_token=token, chat_id=chat_id)


def _build_config() -> dict:
    config = DEFAULT_CONFIG.copy()
    overrides = {
        "llm_provider":          os.environ.get("TRADINGAGENTS_LLM_PROVIDER"),
        "deep_think_llm":        os.environ.get("TRADINGAGENTS_DEEP_THINK_LLM"),
        "quick_think_llm":       os.environ.get("TRADINGAGENTS_QUICK_THINK_LLM"),
        "max_debate_rounds":     os.environ.get("TRADINGAGENTS_MAX_DEBATE_ROUNDS"),
        "max_risk_discuss_rounds": os.environ.get("TRADINGAGENTS_MAX_RISK_ROUNDS"),
    }
    for key, val in overrides.items():
        if val is not None:
            default = DEFAULT_CONFIG.get(key)
            config[key] = type(default)(val) if default is not None else val
    return config
