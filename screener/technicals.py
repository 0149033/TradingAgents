"""
Pine Script "Stock Decision Engine v4.2" ported to Python.

Replicates all conditions exactly:
  - Trend  : EMA50 > EMA200 (daily) + ADX > 20 + weekly EMA50 > weekly EMA200
  - Signals: BUY / ACCUMULATE / WATCH / AVOID
  - Risk   : stop loss (ATR-based), Target 1 (2R), Target 2 (3R), Risk %

Weekly EMA is computed by resampling the daily OHLCV to weekly — no extra
network call needed as long as the caller supplies ≥ 4 years of daily data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


# ------------------------------------------------------------------ #
# Parameters (mirror Pine Script defaults)                            #
# ------------------------------------------------------------------ #

@dataclass
class TechParams:
    ema_fast: int = 50
    ema_slow: int = 200
    rsi_len: int = 14
    rsi_buy_min: float = 55.0
    rsi_acc_min: float = 50.0
    vol_len: int = 20
    buy_vol_mult: float = 1.5
    acc_vol_mult: float = 1.0
    break_len: int = 20
    adx_len: int = 14
    adx_smooth: int = 14
    adx_min: float = 20.0
    atr_len: int = 14
    atr_break_mult: float = 0.5
    ema_touch_pct: float = 3.0
    max_ext_pct: float = 12.0
    setup_score_min: int = 3
    trigger_score_min: int = 5


DEFAULT_PARAMS = TechParams()


# ------------------------------------------------------------------ #
# Output                                                               #
# ------------------------------------------------------------------ #

@dataclass
class TechnicalSignal:
    signal: str                   # BUY | ACCUMULATE | WATCH | AVOID | NEUTRAL
    setup_score: int              # 0–5
    trigger_score: int            # 0–5
    entry: Optional[float] = None
    stop_loss: Optional[float] = None
    target1: Optional[float] = None   # 2R
    target2: Optional[float] = None   # 3R
    risk_pct: Optional[float] = None
    # individual conditions (useful for Telegram debug and TradingAgents prompt)
    trend: bool = False
    above_ema200: bool = False
    momentum: bool = False            # RSI > buy_min
    acc_momentum: bool = False        # RSI > acc_min
    volume_spike: bool = False
    acc_volume_ok: bool = False
    bull_candle: bool = False
    breakout: bool = False
    over_extended: bool = False
    bounce_from_ema: bool = False
    major_breakdown: bool = False
    weekly_uptrend: Optional[bool] = None
    adx: Optional[float] = None
    rsi: Optional[float] = None
    ema_fast_val: Optional[float] = None
    ema_slow_val: Optional[float] = None
    missing: list[str] = field(default_factory=list)


# ------------------------------------------------------------------ #
# Main entry point                                                     #
# ------------------------------------------------------------------ #

def classify(
    ohlcv: pd.DataFrame,
    params: TechParams = DEFAULT_PARAMS,
) -> TechnicalSignal:
    """
    Classify a ticker from its daily OHLCV DataFrame.

    ohlcv must have columns: Open, High, Low, Close, Volume
    Index must be a DatetimeIndex, sorted ascending.
    Needs at least 4 years (~1000 rows) for weekly EMA200 to be meaningful;
    signals are still produced with fewer rows but weekly_uptrend may be None.
    """
    if ohlcv is None or len(ohlcv) < max(params.ema_slow + 5, 60):
        return TechnicalSignal(signal="NEUTRAL", setup_score=0, trigger_score=0)

    close   = ohlcv["Close"]
    high    = ohlcv["High"]
    low     = ohlcv["Low"]
    volume  = ohlcv["Volume"]
    open_   = ohlcv["Open"]

    # ── Indicators ────────────────────────────────────────────────────
    ema_fast = _ema(close, params.ema_fast)
    ema_slow = _ema(close, params.ema_slow)
    rsi      = _rsi(close, params.rsi_len)
    vol_avg  = close.rolling(params.vol_len).mean()   # Pine uses SMA on volume
    vol_avg  = volume.rolling(params.vol_len).mean()
    atr      = _atr(high, low, close, params.atr_len)
    adx_val, di_plus, di_minus = _adx(high, low, close, params.adx_len, params.adx_smooth)

    # Weekly EMA via resample
    weekly_uptrend = _weekly_uptrend(ohlcv, params.ema_fast, params.ema_slow)

    # Grab last-bar scalars
    c   = float(close.iloc[-1])
    o   = float(open_.iloc[-1])
    h   = float(high.iloc[-1])
    lo  = float(low.iloc[-1])
    vol = float(volume.iloc[-1])

    ef  = float(ema_fast.iloc[-1])
    es  = float(ema_slow.iloc[-1])
    rsi_val = float(rsi.iloc[-1]) if pd.notna(rsi.iloc[-1]) else None
    vol_avg_val = float(vol_avg.iloc[-1]) if pd.notna(vol_avg.iloc[-1]) else None
    atr_val = float(atr.iloc[-1]) if pd.notna(atr.iloc[-1]) else None
    adx_scalar = float(adx_val.iloc[-1]) if pd.notna(adx_val.iloc[-1]) else None

    # 20-bar highest high (excluding current bar, like Pine's [1])
    prev_high = float(high.iloc[-params.break_len - 1 : -1].max()) if len(high) > params.break_len else None

    # ── Core conditions (mirrors Pine Script exactly) ─────────────────
    strong_trend = (adx_scalar is not None and adx_scalar > params.adx_min)
    trend        = (ef > es) and strong_trend and (weekly_uptrend is True)
    above_ema200 = c > es
    momentum     = (rsi_val is not None and rsi_val > params.rsi_buy_min)
    acc_momentum = (rsi_val is not None and rsi_val > params.rsi_acc_min)
    volume_spike = (vol_avg_val is not None and vol > vol_avg_val * params.buy_vol_mult)
    acc_vol_ok   = (vol_avg_val is not None and vol > vol_avg_val * params.acc_vol_mult)
    bull_candle  = c > o

    breakout_raw = (prev_high is not None and c > prev_high)
    breakout     = (
        breakout_raw and
        atr_val is not None and
        (c - prev_high) > atr_val * params.atr_break_mult
    ) if breakout_raw else False

    above_ema_pct = ((c - ef) / ef * 100) if ef != 0 else None
    over_extended = (
        above_ema_pct is not None and
        c > ef and
        above_ema_pct > params.max_ext_pct
    )

    # EMA bounce
    touched_ema_zone = lo <= ef * (1 + params.ema_touch_pct / 100)
    reclaimed_ema    = c > ef
    bullish_reclaim  = c > o and c > float(high.iloc[-2]) if len(high) >= 2 else False
    no_heavy_sell    = not (c < o and vol_avg_val is not None and vol > vol_avg_val * 2)

    bounce_from_ema = (
        touched_ema_zone and
        reclaimed_ema and
        bullish_reclaim and
        acc_momentum and
        acc_vol_ok and
        no_heavy_sell
    )

    major_breakdown = (
        c < ef and
        (rsi_val is not None and rsi_val < 50) and
        (vol_avg_val is not None and vol > vol_avg_val * 1.5)
    )

    # ── Scores ────────────────────────────────────────────────────────
    setup_score = sum([
        trend,
        above_ema200,
        momentum,
        volume_spike,
        bull_candle,
    ])

    trigger_score = sum([
        trend,
        above_ema200,
        momentum,
        volume_spike,
        breakout,
    ])

    setup_ready   = setup_score   >= params.setup_score_min
    trigger_ready = trigger_score >= params.trigger_score_min

    # ── Signal classification ─────────────────────────────────────────
    strong_buy = (
        setup_ready and
        trigger_ready and
        breakout and
        bull_candle and
        not over_extended and
        c > ef and
        c > es
    )

    accumulate = (
        trend and
        bounce_from_ema and
        not major_breakdown and
        not over_extended and
        not breakout
    )

    avoid = (
        major_breakdown or
        not trend or
        c < es or
        (rsi_val is not None and rsi_val < 45)
    )

    watch = (
        not strong_buy and
        not accumulate and
        setup_ready and
        not over_extended and
        not major_breakdown
    )

    if strong_buy:
        signal = "BUY"
    elif accumulate:
        signal = "ACCUMULATE"
    elif avoid:
        signal = "AVOID"
    elif watch:
        signal = "WATCH"
    else:
        signal = "NEUTRAL"

    # ── Risk management ───────────────────────────────────────────────
    entry = stop = t1 = t2 = risk_pct_val = None
    active = strong_buy or accumulate
    if active and atr_val is not None:
        entry = c
        breakout_stop = lo
        atr_stop      = c - atr_val * 2
        acc_stop      = ef - atr_val

        if strong_buy:
            stop = min(breakout_stop, atr_stop)
        else:
            stop = min(acc_stop, atr_stop)

        risk = entry - stop
        if risk > 0:
            t1          = entry + risk * 2
            t2          = entry + risk * 3
            risk_pct_val = (risk / entry) * 100

    # ── Missing conditions (debug) ────────────────────────────────────
    missing: list[str] = []
    if not trend:        missing.append("Trend")
    if not above_ema200: missing.append("AboveEMA200")
    if not momentum:     missing.append("RSI")
    if not volume_spike: missing.append("Volume")
    if not breakout:     missing.append("Breakout")
    if not bull_candle:  missing.append("BullCandle")
    if not bounce_from_ema: missing.append("EMABounce")
    if over_extended:    missing.append("OverExtended")

    return TechnicalSignal(
        signal=signal,
        setup_score=setup_score,
        trigger_score=trigger_score,
        entry=entry,
        stop_loss=stop,
        target1=t1,
        target2=t2,
        risk_pct=risk_pct_val,
        trend=trend,
        above_ema200=above_ema200,
        momentum=momentum,
        acc_momentum=acc_momentum,
        volume_spike=volume_spike,
        acc_volume_ok=acc_vol_ok,
        bull_candle=bull_candle,
        breakout=breakout,
        over_extended=over_extended,
        bounce_from_ema=bounce_from_ema,
        major_breakdown=major_breakdown,
        weekly_uptrend=weekly_uptrend,
        adx=adx_scalar,
        rsi=rsi_val,
        ema_fast_val=ef,
        ema_slow_val=es,
        missing=missing,
    )


# ------------------------------------------------------------------ #
# Indicator implementations                                           #
# ------------------------------------------------------------------ #

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def _adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    di_len: int = 14,
    adx_smooth: int = 14,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Wilder-smoothed ADX, +DI, -DI — matches Pine Script ta.dmi()."""
    prev_high  = high.shift(1)
    prev_low   = low.shift(1)
    prev_close = close.shift(1)

    up_move   = high - prev_high
    down_move = prev_low - low

    plus_dm  = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

    alpha = 1 / di_len
    atr_w     = pd.Series(tr).ewm(alpha=alpha, adjust=False).mean()
    plus_dm_s = pd.Series(plus_dm,  index=high.index).ewm(alpha=alpha, adjust=False).mean()
    minus_dm_s= pd.Series(minus_dm, index=high.index).ewm(alpha=alpha, adjust=False).mean()

    di_plus  = 100 * plus_dm_s  / atr_w.replace(0, np.nan)
    di_minus = 100 * minus_dm_s / atr_w.replace(0, np.nan)

    dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / adx_smooth, adjust=False).mean()

    return adx, di_plus, di_minus


def _weekly_uptrend(
    ohlcv: pd.DataFrame,
    fast_span: int,
    slow_span: int,
) -> Optional[bool]:
    """Resample daily OHLCV to weekly and check EMA fast > EMA slow."""
    try:
        weekly_close = ohlcv["Close"].resample("W").last().dropna()
        if len(weekly_close) < slow_span:
            return None
        wef = _ema(weekly_close, fast_span).iloc[-1]
        wes = _ema(weekly_close, slow_span).iloc[-1]
        return bool(wef > wes)
    except Exception:
        return None
