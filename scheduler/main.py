"""
APScheduler entry point for the TradingAgents automated pipeline.

Default schedule:
  - Pre-market:  08:30 ET  Mon–Fri   (screen + analyse overnight developments)
  - Post-market: 16:30 ET  Mon–Fri   (screen + analyse end-of-day data)

Override with env vars:
  SCHEDULE_PRE_MARKET_HOUR   (default: 8)
  SCHEDULE_PRE_MARKET_MINUTE (default: 30)
  SCHEDULE_POST_MARKET_HOUR  (default: 16)
  SCHEDULE_POST_MARKET_MINUTE(default: 30)
  SCHEDULE_DISABLE_PRE=true  — skip pre-market run
  SCHEDULE_DISABLE_POST=true — skip post-market run

Run once immediately (no scheduler):
  python -m scheduler.main --now
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root before anything else
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from scheduler.runner import run_screening_and_analysis


def _int_env(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _bool_env(name: str) -> bool:
    return os.environ.get(name, "").lower() in ("1", "true", "yes")


def main() -> None:
    if "--now" in sys.argv:
        logger.info("--now flag: running pipeline immediately (no scheduler).")
        run_screening_and_analysis()
        return

    scheduler = BlockingScheduler(timezone="America/New_York")

    pre_hour = _int_env("SCHEDULE_PRE_MARKET_HOUR", 8)
    pre_min = _int_env("SCHEDULE_PRE_MARKET_MINUTE", 30)
    post_hour = _int_env("SCHEDULE_POST_MARKET_HOUR", 16)
    post_min = _int_env("SCHEDULE_POST_MARKET_MINUTE", 30)

    if not _bool_env("SCHEDULE_DISABLE_PRE"):
        scheduler.add_job(
            run_screening_and_analysis,
            CronTrigger(day_of_week="mon-fri", hour=pre_hour, minute=pre_min),
            id="pre_market",
            name=f"Pre-market run ({pre_hour:02d}:{pre_min:02d} ET)",
            misfire_grace_time=300,
        )
        logger.info("Scheduled pre-market run at %02d:%02d ET (Mon–Fri)", pre_hour, pre_min)

    if not _bool_env("SCHEDULE_DISABLE_POST"):
        scheduler.add_job(
            run_screening_and_analysis,
            CronTrigger(day_of_week="mon-fri", hour=post_hour, minute=post_min),
            id="post_market",
            name=f"Post-market run ({post_hour:02d}:{post_min:02d} ET)",
            misfire_grace_time=300,
        )
        logger.info("Scheduled post-market run at %02d:%02d ET (Mon–Fri)", post_hour, post_min)

    logger.info("Scheduler started — press Ctrl+C to stop.")
    try:
        scheduler.start()
    except KeyboardInterrupt:
        logger.info("Scheduler stopped.")


if __name__ == "__main__":
    main()
