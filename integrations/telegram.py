"""Send trading signal alerts to a Telegram chat."""

from __future__ import annotations

import logging
import os
import textwrap
from dataclasses import dataclass, field
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# Pine Script signal emojis
_SCREENER_EMOJI = {
    "BUY":        "🟢",
    "ACCUMULATE": "🔵",
    "WATCH":      "🟡",
    "AVOID":      "🔴",
    "NEUTRAL":    "⚪",
}

# TradingAgents pipeline signal emojis
_PIPELINE_EMOJI = {
    "Buy":        "🟢",
    "Overweight": "🟩",
    "Hold":       "🟡",
    "Underweight":"🟥",
    "Sell":       "🔴",
}

_MAX_MESSAGE_LENGTH = 4096

# ── Confidence table ───────────────────────────────────────────────────────
# Maps (screener_signal, pipeline_signal) → (label, position_multiplier)
# position_multiplier: 1.0 = full size, 0.67 = two-thirds, 0.33 = one-third, 0 = skip
_CONFIDENCE: dict[tuple[str, str], tuple[str, float]] = {
    ("BUY",        "Buy"):         ("MAX — Full size",      1.00),
    ("BUY",        "Overweight"):  ("STRONG — Full size",   1.00),
    ("BUY",        "Hold"):        ("MODERATE — Half size", 0.50),
    ("ACCUMULATE", "Buy"):         ("STRONG — 2/3 size",    0.67),
    ("ACCUMULATE", "Overweight"):  ("STRONG — 2/3 size",    0.67),
    ("ACCUMULATE", "Hold"):        ("WEAK — 1/3 size",      0.33),
    ("WATCH",      "Buy"):         ("MODERATE — 1/3 size",  0.33),
    ("WATCH",      "Overweight"):  ("MODERATE — 1/3 size",  0.33),
    ("WATCH",      "Hold"):        ("SKIP — No action",     0.00),
    ("WATCH",      "Underweight"): ("SKIP — No action",     0.00),
    ("WATCH",      "Sell"):        ("SKIP — No action",     0.00),
    ("BUY",        "Underweight"): ("CONFLICTED — Skip",    0.00),
    ("BUY",        "Sell"):        ("CONFLICTED — Skip",    0.00),
    ("ACCUMULATE", "Underweight"): ("SKIP — No action",     0.00),
    ("ACCUMULATE", "Sell"):        ("SKIP — No action",     0.00),
}

def _confidence(screener: str, pipeline: str) -> tuple[str, float]:
    """Return (label, position_multiplier) for a screener + pipeline pair."""
    return _CONFIDENCE.get((screener, pipeline), ("SKIP — No action", 0.00))


@dataclass
class SignalAlert:
    ticker: str
    screener_signal: str     # BUY | ACCUMULATE | WATCH from Pine Script
    pipeline_signal: str     # Buy | Overweight | Hold | Underweight | Sell from TradingAgents
    trade_date: str
    fundamental_score: float
    screener_reasons: list[str] = field(default_factory=list)

    # Risk levels from Pine Script
    entry: Optional[float] = None
    stop_loss: Optional[float] = None
    target1: Optional[float] = None   # 2R
    target2: Optional[float] = None   # 3R
    risk_pct: Optional[float] = None

    # Technical conditions
    setup_score: int = 0
    trigger_score: int = 0
    adx: Optional[float] = None
    rsi: Optional[float] = None
    weekly_uptrend: Optional[bool] = None

    # TradingAgents pipeline output
    investment_plan: Optional[str] = None
    final_trade_decision: Optional[str] = None


class TelegramAlerter:
    """Send alerts via the Telegram Bot API (synchronous HTTP)."""

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
    ):
        self.bot_token = bot_token or os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self.chat_id   = chat_id   or os.environ.get("TELEGRAM_CHAT_ID", "")
        if not self.bot_token or not self.chat_id:
            raise ValueError(
                "TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set "
                "(env vars or constructor args)."
            )

    def send_signal(self, alert: SignalAlert) -> bool:
        return self._send(_format_signal(alert))

    def send_run_summary(self, alerts: list[SignalAlert], run_date: str) -> bool:
        if not alerts:
            return self._send(
                f"📊 *TradingAgents — {run_date}*\nScreener found no qualifying tickers."
            )
        lines = [f"📊 *TradingAgents run — {run_date}*\n"]
        for a in alerts:
            se = _SCREENER_EMOJI.get(a.screener_signal, "⚪")
            pe = _PIPELINE_EMOJI.get(a.pipeline_signal, "⚪")
            risk_str = f"  Risk {a.risk_pct:.1f}%" if a.risk_pct else ""
            lines.append(
                f"{se} `{a.ticker}`  screener: *{a.screener_signal}*  "
                f"→  {pe} pipeline: *{a.pipeline_signal}*{risk_str}"
            )
        lines.append("\n_Full per-ticker reports follow…_")
        return self._send("\n".join(lines))

    def send_error(self, ticker: str, error: str) -> bool:
        return self._send(
            f"⚠️ *TradingAgents error*\nTicker: `{ticker}`\n`{error[:300]}`"
        )

    def _send(self, text: str) -> bool:
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        if len(text) > _MAX_MESSAGE_LENGTH:
            text = text[: _MAX_MESSAGE_LENGTH - 20] + "\n…_(truncated)_"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True,
        }
        try:
            resp = requests.post(url, json=payload, timeout=15)
            resp.raise_for_status()
            return True
        except Exception as exc:
            logger.error("Telegram send failed: %s", exc)
            return False


# ------------------------------------------------------------------ #
# Message formatter                                                   #
# ------------------------------------------------------------------ #

def _format_signal(a: SignalAlert) -> str:
    se = _SCREENER_EMOJI.get(a.screener_signal, "⚪")
    pe = _PIPELINE_EMOJI.get(a.pipeline_signal, "⚪")
    conf_label, pos_mult = _confidence(a.screener_signal, a.pipeline_signal)

    # ── Header ────────────────────────────────────────────────────────
    lines = [
        f"{se} *{a.ticker}*  —  Screener: *{a.screener_signal}*  |  {pe} Pipeline: *{a.pipeline_signal}*",
        f"📅 {a.trade_date}   📊 Fund. score: {a.fundamental_score:.0f}",
    ]

    # ── Confidence & action ───────────────────────────────────────────
    conf_emoji = "🔥" if pos_mult == 1.0 else "✅" if pos_mult >= 0.67 else "⚠️" if pos_mult > 0 else "⛔"
    lines.append(f"{conf_emoji} *Confidence: {conf_label}*")

    # ── Scores & technicals ───────────────────────────────────────────
    tech_parts = []
    if a.setup_score or a.trigger_score:
        tech_parts.append(f"Setup {a.setup_score}/5  Trigger {a.trigger_score}/5")
    if a.adx is not None:
        tech_parts.append(f"ADX {a.adx:.0f}")
    if a.rsi is not None:
        tech_parts.append(f"RSI {a.rsi:.0f}")
    if a.weekly_uptrend is not None:
        tech_parts.append(f"WeeklyTrend {'✓' if a.weekly_uptrend else '✗'}")
    if tech_parts:
        lines.append("_" + "  ·  ".join(tech_parts) + "_")

    # ── Risk + position sizing table ──────────────────────────────────
    if a.entry is not None and a.stop_loss is not None and pos_mult > 0:
        portfolio  = float(os.environ.get("PORTFOLIO_SIZE", "10000"))
        max_risk_pct = float(os.environ.get("MAX_RISK_PCT", "1.0"))

        # Dollar risk scales with confidence multiplier
        # e.g. 1% of $10k = $100 full size → $67 at 2/3 → $33 at 1/3
        dollar_risk    = portfolio * (max_risk_pct / 100) * pos_mult
        risk_per_share = a.entry - a.stop_loss
        shares         = int(dollar_risk / risk_per_share) if risk_per_share > 0 else 0
        position_value = shares * a.entry

        lines.append("")
        lines.append("```")
        lines.append(f"Entry   : ${a.entry:.2f}")
        lines.append(f"Stop    : ${a.stop_loss:.2f}  ({a.risk_pct:.1f}% away)")
        if a.target1:
            lines.append(f"Target1 : ${a.target1:.2f}  (2R)")
        if a.target2:
            lines.append(f"Target2 : ${a.target2:.2f}  (3R)")
        lines.append("─────────────────────")
        lines.append(f"Portfolio : ${portfolio:,.0f}")
        lines.append(f"Risk/trade: ${dollar_risk:.0f}  ({max_risk_pct * pos_mult:.1f}% of portfolio)")
        lines.append(f"Shares    : {shares}")
        lines.append(f"Position  : ${position_value:,.0f}")
        lines.append("```")
    elif a.entry is not None and pos_mult == 0:
        lines.append(f"\n_Entry ${a.entry:.2f} — no position recommended_")

    # ── Screener reasons ─────────────────────────────────────────────
    if a.screener_reasons:
        lines.append("*Why screened:* " + "  ·  ".join(a.screener_reasons[:6]))

    # ── Investment plan summary ───────────────────────────────────────
    if a.investment_plan:
        plan_text = "\n".join(
            ln for ln in a.investment_plan.strip().splitlines()
            if ln.strip() and not ln.startswith("#")
        )
        summary = textwrap.shorten(plan_text, width=400, placeholder="…")
        lines.append(f"\n*Investment thesis:*\n{summary}")

    # ── Pipeline decision ─────────────────────────────────────────────
    if a.final_trade_decision:
        dec = textwrap.shorten(a.final_trade_decision.strip(), width=280, placeholder="…")
        lines.append(f"\n*Portfolio Manager:*\n{dec}")

    lines.append("\n_Full report → ~/.tradingagents/logs/_")
    return "\n".join(lines)
