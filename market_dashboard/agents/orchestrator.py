"""
Orchestrator — schedules and coordinates all agents.
Uses APScheduler for background execution.

Schedule (default):
  - indicators_agent:  every 5 minutes (always)
  - signal_agent:      every 5 minutes, after indicators (only if composite_score ≥ 65)
  - analysis_agent:    every 60 minutes
  - config_optimizer:  daily at 07:00 Israel time
  - daily_pipeline:    daily at 07:30 Israel time (runs final_score.py backend)

Thread-safety: all state writes use atomic JSON overwrites.
The Streamlit UI reads agent_status.json (written by each agent).
"""
from __future__ import annotations

import json
import logging
import sys
import threading
import traceback
from datetime import datetime
from pathlib import Path

import pytz

_MD   = Path(__file__).resolve().parents[1]
_ROOT = _MD.parent
for p in (str(_MD), str(_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger
    from apscheduler.triggers.interval import IntervalTrigger
    _APScheduler_available = True
except ImportError:
    _APScheduler_available = False

# Agent modules imported lazily inside job functions so that missing
# optional dependencies (anthropic, apscheduler) don't crash the UI on load.

IL_TZ  = pytz.timezone("Asia/Jerusalem")
_LOGS  = _MD / "logs"
_STATUS_PATH = _LOGS / "agent_status.json"
_ORCH_LOG    = _LOGS / "orchestrator_log.json"

logger = logging.getLogger("orchestrator")

# ─── Global scheduler (singleton) ─────────────────────────────────────────────
_scheduler: "BackgroundScheduler | None" = None
_lock = threading.Lock()


# ─── Status helpers ────────────────────────────────────────────────────────────

def _load_status() -> dict:
    if _STATUS_PATH.exists():
        try:
            return json.loads(_STATUS_PATH.read_text())
        except Exception:
            pass
    return {}


def _update_orch_status(status: str, info: str = "") -> None:
    data = _load_status()
    data["orchestrator"] = {
        **data.get("orchestrator", {}),
        "status":    status,
        "last_run":  datetime.now(IL_TZ).isoformat(),
        "info":      info,
        "enabled":   data.get("orchestrator", {}).get("enabled", False),
        "runs":      data.get("orchestrator", {}).get("runs", 0) + (1 if status == "done" else 0),
    }
    _LOGS.mkdir(parents=True, exist_ok=True)
    _STATUS_PATH.write_text(json.dumps(data, indent=2))


def _append_orch_log(entry: dict) -> None:
    entries = []
    if _ORCH_LOG.exists():
        try:
            entries = json.loads(_ORCH_LOG.read_text())
        except Exception:
            pass
    entry["timestamp"] = datetime.now(IL_TZ).isoformat()
    entries.append(entry)
    _ORCH_LOG.write_text(json.dumps(entries[-500:], indent=2))


# ─── Individual agent jobs ─────────────────────────────────────────────────────

def _job_indicators() -> None:
    """Scheduled job: run indicators_agent."""
    try:
        from agents import indicators_agent
        state = indicators_agent.run()
        score = state.get("composite_score", {}).get("total", 0)
        _append_orch_log({"job": "indicators", "status": "done", "score": score})
    except Exception as e:
        logger.error("indicators job failed: %s", e)
        _append_orch_log({"job": "indicators", "status": "error", "error": str(e)})


def _job_signal() -> None:
    """Scheduled job: run signal_agent (only if composite_score ≥ 65)."""
    try:
        from agents import signal_agent
        result = signal_agent.run()
        _append_orch_log({
            "job":    "signal_agent",
            "status": result.get("outcome", "?"),
            "score":  result.get("composite_score", 0),
        })
    except Exception as e:
        logger.error("signal_agent job failed: %s", e)
        _append_orch_log({"job": "signal_agent", "status": "error", "error": str(e)})


def _job_indicators_then_signal() -> None:
    """Combined job: indicators first, then signal (if escalate)."""
    from agents import indicators_agent
    _job_indicators()
    state = indicators_agent.load_state()
    if state.get("composite_score", {}).get("escalate", False):
        _job_signal()
    else:
        _append_orch_log({
            "job":    "signal_agent",
            "status": "skipped",
            "reason": f"score {state.get('composite_score', {}).get('total', 0)} < 65",
        })


def _job_analysis() -> None:
    """Scheduled job: run analysis_agent (auto-evaluate trades + write insights)."""
    try:
        from agents import analysis_agent
        result = analysis_agent.run()
        _append_orch_log({
            "job":      "analysis_agent",
            "status":   result.get("outcome", "?"),
            "insights": result.get("insights_written", 0),
            "evaluated": result.get("trades_evaluated", 0),
        })
    except Exception as e:
        logger.error("analysis_agent job failed: %s", e)
        _append_orch_log({"job": "analysis_agent", "status": "error", "error": str(e)})


def _job_config_optimizer() -> None:
    """Scheduled job: daily config optimization."""
    try:
        from agents import config_optimizer
        result = config_optimizer.run()
        _append_orch_log({
            "job":     "config_optimizer",
            "status":  result.get("outcome", "?"),
            "changes": result.get("changes_applied", 0),
        })
    except Exception as e:
        logger.error("config_optimizer job failed: %s", e)
        _append_orch_log({"job": "config_optimizer", "status": "error", "error": str(e)})


def _job_daily_pipeline() -> None:
    """Scheduled job: run the daily scores_news backend pipeline at 07:30 IL."""
    import subprocess
    pipeline = _ROOT / "scores_news" / "cat_scores" / "final_score.py"
    if not pipeline.exists():
        _append_orch_log({"job": "daily_pipeline", "status": "skipped", "reason": "pipeline not found"})
        return
    try:
        proc = subprocess.run(
            ["python", str(pipeline)],
            capture_output=True, text=True, timeout=300,
        )
        status = "done" if proc.returncode == 0 else "error"
        _append_orch_log({
            "job":    "daily_pipeline",
            "status": status,
            "stdout": proc.stdout[-500:] if proc.stdout else "",
            "stderr": proc.stderr[-300:] if proc.stderr else "",
        })
    except Exception as e:
        logger.error("daily_pipeline job failed: %s", e)
        _append_orch_log({"job": "daily_pipeline", "status": "error", "error": str(e)})


# ─── Scheduler lifecycle ───────────────────────────────────────────────────────

def start(
    indicators_interval_minutes: int = 5,
    signal_interval_minutes: int = 5,
) -> bool:
    """
    Start the background scheduler.
    Returns True if started, False if already running or APScheduler unavailable.
    """
    global _scheduler

    if not _APScheduler_available:
        logger.warning("APScheduler not installed — scheduler disabled")
        return False

    with _lock:
        if _scheduler is not None and _scheduler.running:
            logger.info("Scheduler already running")
            return False

        _scheduler = BackgroundScheduler(timezone=IL_TZ)

        # Combined indicators + signal job every N minutes
        _scheduler.add_job(
            _job_indicators_then_signal,
            trigger=IntervalTrigger(minutes=indicators_interval_minutes),
            id="indicators_signal",
            replace_existing=True,
            misfire_grace_time=60,
        )

        # Analysis agent every 60 minutes
        _scheduler.add_job(
            _job_analysis,
            trigger=IntervalTrigger(minutes=60),
            id="analysis_agent",
            replace_existing=True,
            misfire_grace_time=300,
        )

        # Config optimizer daily at 07:00 Israel time
        _scheduler.add_job(
            _job_config_optimizer,
            trigger=CronTrigger(hour=7, minute=0, timezone=IL_TZ),
            id="config_optimizer",
            replace_existing=True,
            misfire_grace_time=1800,
        )

        # Daily backend pipeline (final_score.py) at 07:30 Israel time
        _scheduler.add_job(
            _job_daily_pipeline,
            trigger=CronTrigger(hour=7, minute=30, timezone=IL_TZ),
            id="daily_pipeline",
            replace_existing=True,
            misfire_grace_time=1800,
        )

        _scheduler.start()
        _update_orch_status("running", f"scheduler started, interval={indicators_interval_minutes}m")
        _append_orch_log({"event": "scheduler_start",
                           "interval_m": indicators_interval_minutes})
        logger.info("Orchestrator scheduler started (interval=%dm)", indicators_interval_minutes)
        return True


def stop() -> bool:
    """Stop the background scheduler."""
    global _scheduler
    with _lock:
        if _scheduler is None or not _scheduler.running:
            return False
        _scheduler.shutdown(wait=False)
        _scheduler = None
        _update_orch_status("idle", "scheduler stopped")
        _append_orch_log({"event": "scheduler_stop"})
        logger.info("Orchestrator scheduler stopped")
        return True


def is_running() -> bool:
    """True if the scheduler is active."""
    global _scheduler
    return _scheduler is not None and _scheduler.running


def run_once(agent_name: str) -> dict:
    """
    Manually trigger a single agent run (blocking).
    Returns the agent result dict.
    """
    _update_orch_status("running", f"manual run: {agent_name}")
    try:
        if agent_name == "indicators":
            from agents import indicators_agent
            result = indicators_agent.run()
        elif agent_name == "signal":
            from agents import signal_agent
            result = signal_agent.run(force=True)
        elif agent_name == "indicators_signal":
            from agents import indicators_agent, signal_agent
            _job_indicators()
            result = signal_agent.run()
        elif agent_name == "analysis":
            from agents import analysis_agent
            result = analysis_agent.run()
        elif agent_name == "optimizer":
            from agents import config_optimizer
            result = config_optimizer.run(force=True)
        else:
            result = {"error": f"Unknown agent: {agent_name}"}

        _update_orch_status("done", f"manual run complete: {agent_name}")
        return result

    except Exception as exc:
        err = traceback.format_exc()
        _update_orch_status("error", str(exc)[:200])
        return {"error": str(exc), "traceback": err[:500]}


def get_status() -> dict:
    """Return current status of all agents + scheduler."""
    status = _load_status()
    status["_scheduler_running"] = is_running()
    status["_apscheduler_available"] = _APScheduler_available
    status["_timestamp"] = datetime.now(IL_TZ).isoformat()
    return status


def get_orch_log(n: int = 50) -> list:
    """Return last n orchestrator log entries."""
    if _ORCH_LOG.exists():
        try:
            entries = json.loads(_ORCH_LOG.read_text())
            return entries[-n:]
        except Exception:
            pass
    return []
