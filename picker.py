#!/usr/bin/env python3
"""
picker.py

Hourly NIFTY-250 picker.

Flow:
1. Read the CURRENT daily LuxAlgo scanner candidates from S3.
2. Start tracking LONG WATCH candidates that pass:
      - Financial score >= MIN_FINANCIAL_SCORE
      - Present PE < Sector PE * PE_SECTOR_MULTIPLIER
3. Track each accepted candidate from the time it was first selected.
4. On completed 1H candles, detect the FIRST bearish -> bullish Supertrend flip.
5. Enter at that completed candle close.
6. Fixed SL/Target:
      SL     = entry * (1 - STOP_LOSS_PCT)
      Target = entry * (1 + TARGET_PCT)
7. Monitor hourly until TARGET, SL, or 30-day expiry.
8. Keep ALL history permanently in S3.

The script never places broker orders. It is an experiment / signal tracker.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

# =============================================================================
# CONFIG — change values here
# =============================================================================

# ---- Scanner / S3 -----------------------------------------------------------
S3_BUCKET = os.getenv("S3_BUCKET", "shri-trading-bot-storage")

SCANNER_STATE_KEY = os.getenv(
    "SCANNER_STATE_KEY",
    "holding/state.json",
)
PICKER_STATE_KEY = os.getenv(
    "PICKER_STATE_KEY",
    "holding/picker-state.json",
)
PICKER_HISTORY_CSV_KEY = os.getenv(
    "PICKER_HISTORY_CSV_KEY",
    "holding/picker-history.csv",
)

# ---- Fundamental / valuation filters ---------------------------------------
MIN_FINANCIAL_SCORE = 52.0

# Present PE must be below:
#     Sector PE * PE_SECTOR_MULTIPLIER
PE_SECTOR_MULTIPLIER = 1.50

# Default and currently supported historical PE period.
# Screener's "5Yrs PE" is used as the historical PE value.
HISTORICAL_PE_YEARS = 5

# Cache Screener valuation data locally for this many hours.
# PE is a filter at candidate-selection time, not an hourly trading signal.
# ---- Technical entry --------------------------------------------------------
SUPERTREND_PERIOD = 10
SUPERTREND_MULTIPLIER = 4.0

# Entry is the CLOSE of the completed bullish-flip 1H candle.
STOP_LOSS_PCT = 0.08
TARGET_PCT = 0.10

# A WATCHING candidate expires if no first bullish flip occurs within this time.
WATCHING_EXPIRY_DAYS = 30

# An ACTIVE trade expires if neither target nor SL is hit.
TRADE_EXPIRY_DAYS = 30

# Enough history to safely detect a recent flip even for older candidates.
HOURLY_LOOKBACK_DAYS = 60

# A bullish flip in any of the most recent completed 1H candles is eligible.
RECENT_FLIP_CANDLES = 4

# ---- Runtime / data ---------------------------------------------------------
TIMEFRAME = "60minute"
MARKET_EXCHANGE = "NSE"

# The cron schedule requested is:
#     47 10-15 * * 1-5
# This script itself does not sleep or run a scheduler.

# ---- Central secrets / AWS --------------------------------------------------
AWS_REGION = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "ap-south-1"

# picker.py fetches sensitive values directly from AWS SSM.
# No config.py is required.
#   /trading/zerodha_api_key
#   /trading/zerodha_api_secret
#   /trading/zerodha_access_token
#   /trading/telegram_bot_token
KITE_ACCESS_TOKEN_SSM_PARAM = os.getenv(
    "KITE_ACCESS_TOKEN_PARAM",
    "/trading/zerodha_access_token",
)

# ---- Telegram ---------------------------------------------------------------
TELEGRAM_ENABLED = os.getenv("TELEGRAM_ENABLED", "true").lower() == "true"

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

LOG = logging.getLogger("picker")


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class Candidate:
    symbol: str
    name: str
    selected_at: str

    financial_score: float
    financial_grade: str

    equilibrium_distance: float
    scanner_distance_percent: float
    scanner_price: float
    scanner_level: float

    present_pe: float | None
    sector_pe: float | None
    historical_pe: float | None
    pe_status: str

    scanner_run_at: str = ""
    scanner_version: str = ""
    daily_data_as_of: str = ""
    financial_status: str = "INCOMPLETE"

    status: str = "WATCHING"
    expires_at: str = ""

    last_processed_candle: str = ""

    # First bearish state seen/confirmed after selection.
    last_supertrend_direction: int | None = None

    # Filled only after a trade is created.
    trade_id: str | None = None


@dataclass
class Trade:
    trade_id: str
    symbol: str
    name: str

    selected_at: str
    entry_time: str
    entry: float
    stop_loss: float
    target: float

    financial_score: float
    financial_grade: str
    equilibrium_distance: float

    present_pe: float | None
    sector_pe: float | None
    historical_pe: float | None
    pe_status: str

    scanner_run_at: str = ""
    scanner_version: str = ""
    daily_data_as_of: str = ""
    financial_status: str = "INCOMPLETE"

    entry_detected_at: str = ""
    status: str = "ACTIVE"
    exit_time: str | None = None
    exit_price: float | None = None
    exit_reason: str | None = None
    expires_at: str = ""

    # Prevent duplicate Telegram messages after restarts.
    telegram_entry_sent: bool = False
    telegram_exit_sent: bool = False


@dataclass
class PickerState:
    candidates: dict[str, Candidate] = field(default_factory=dict)
    trades: dict[str, Trade] = field(default_factory=dict)


# =============================================================================
# GENERIC HELPERS
# =============================================================================

def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat()


def parse_dt(value: str) -> datetime:
    value = value.replace("Z", "+00:00") if isinstance(value, str) else value
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def normalize_dt(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip()
        if not text:
            return None
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None

    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def number(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if not text or text in {"-", "N/A", "NA", "None"}:
        return None
    text = text.replace("%", "")
    text = re.sub(r"[^0-9.\-]", "", text)
    if text in {"", "-", ".", "-."}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def fmt(value: float | None, digits: int = 2) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


# =============================================================================
# AWS / S3
# =============================================================================

def s3_client():
    import boto3  # type: ignore
    return boto3.client("s3", region_name=AWS_REGION)


def load_s3_json(key: str, default: Any) -> Any:
    s3 = s3_client()
    try:
        raw = s3.get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()
    except Exception as exc:
        text = str(exc).lower()
        if "nosuchkey" in text or "not found" in text or "404" in text:
            return default
        raise RuntimeError(f"Unable to read s3://{S3_BUCKET}/{key}") from exc

    return json.loads(raw.decode("utf-8"))


def save_s3_json(key: str, payload: Any) -> None:
    s3 = s3_client()
    body = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=key,
        Body=body,
        ContentType="application/json",
    )


def load_s3_text(key: str) -> str | None:
    s3 = s3_client()
    try:
        raw = s3.get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()
    except Exception as exc:
        text = str(exc).lower()
        if "nosuchkey" in text or "not found" in text or "404" in text:
            return None
        raise RuntimeError(f"Unable to read s3://{S3_BUCKET}/{key}") from exc
    return raw.decode("utf-8")


def save_s3_text(key: str, text: str, content_type: str) -> None:
    s3 = s3_client()
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=key,
        Body=text.encode("utf-8"),
        ContentType=content_type,
    )


# =============================================================================
# PICKER STATE
# =============================================================================

def load_picker_state() -> PickerState:
    raw = load_s3_json(PICKER_STATE_KEY, {"candidates": {}, "trades": {}})

    candidates = {}
    for symbol, value in raw.get("candidates", {}).items():
        try:
            candidates[symbol] = Candidate(**value)
        except (TypeError, ValueError) as exc:
            LOG.warning("Skipping malformed candidate %s: %s", symbol, exc)

    trades = {}
    for trade_id, value in raw.get("trades", {}).items():
        try:
            trades[trade_id] = Trade(**value)
        except (TypeError, ValueError) as exc:
            LOG.warning("Skipping malformed trade %s: %s", trade_id, exc)

    return PickerState(candidates=candidates, trades=trades)


def save_picker_state(state: PickerState) -> None:
    save_s3_json(
        PICKER_STATE_KEY,
        {
            "updated_at": iso(now_utc()),
            "config": {
                "min_financial_score": MIN_FINANCIAL_SCORE,
                "pe_sector_multiplier": PE_SECTOR_MULTIPLIER,
                "historical_pe_years": HISTORICAL_PE_YEARS,
                "supertrend_period": SUPERTREND_PERIOD,
                "supertrend_multiplier": SUPERTREND_MULTIPLIER,
                "stop_loss_pct": STOP_LOSS_PCT,
                "target_pct": TARGET_PCT,
                "watching_expiry_days": WATCHING_EXPIRY_DAYS,
                "trade_expiry_days": TRADE_EXPIRY_DAYS,
            },
            "candidates": {
                symbol: asdict(candidate)
                for symbol, candidate in state.candidates.items()
            },
            "trades": {
                trade_id: asdict(trade)
                for trade_id, trade in state.trades.items()
            },
        },
    )


# =============================================================================
# SCANNER INPUT
# =============================================================================

def load_scanner_candidates() -> dict[str, dict[str, Any]]:
    """
    Scanner state is the source of truth for the current daily candidates.

    The existing scanner stores a flat object:
        {
            "SYMBOL": {...WatchlistEntry fields...}
        }

    Be tolerant of an optional wrapper just in case.
    """
    raw = load_s3_json(SCANNER_STATE_KEY, {})

    if isinstance(raw, dict) and isinstance(raw.get("entries"), dict):
        raw = raw["entries"]

    if not isinstance(raw, dict):
        raise ValueError("Scanner state must be a JSON object")

    return {
        str(symbol).upper(): value
        for symbol, value in raw.items()
        if isinstance(value, dict)
    }

# =============================================================================
# ZERODHA KITE
# =============================================================================

class KiteClient:
    def __init__(self) -> None:
        import boto3  # type: ignore
        from kiteconnect import KiteConnect  # type: ignore        # Fetch all sensitive values directly from AWS SSM.
        # picker.py does not depend on config.py.
        ssm = boto3.client("ssm", region_name=AWS_REGION)

        def get_ssm_secret(parameter_name: str) -> str:
            return ssm.get_parameter(
                Name=parameter_name,
                WithDecryption=True,
            )["Parameter"]["Value"]

        api_key = get_ssm_secret("/trading/zerodha_api_key")
        api_secret = get_ssm_secret("/trading/zerodha_api_secret")
        access_token = os.getenv("ZERODHA_ACCESS_TOKEN") or get_ssm_secret(
            KITE_ACCESS_TOKEN_SSM_PARAM
        )

        # API_SECRET is fetched from SSM as requested. Historical-data calls
        # authenticate with the Zerodha access token.
        _ = api_secret

        self.kite = KiteConnect(api_key=api_key)
        self.kite.set_access_token(access_token)
        self.tokens: dict[str, int] = {}

    def _load_tokens(self) -> None:
        rows = self.kite.instruments(MARKET_EXCHANGE)
        self.tokens = {
            row["tradingsymbol"]: int(row["instrument_token"])
            for row in rows
        }

    def historical_1h(self, symbol: str) -> list[dict[str, Any]]:
        if not self.tokens:
            self._load_tokens()

        token = self.tokens.get(symbol)
        if token is None:
            raise ValueError(f"No NSE instrument token for {symbol}")

        end = datetime.now()
        start = end - timedelta(days=HOURLY_LOOKBACK_DAYS)

        last_error: Exception | None = None
        for attempt in range(3):
            try:
                rows = self.kite.historical_data(
                    token,
                    start,
                    end,
                    TIMEFRAME,
                )
                return rows
            except Exception as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(2 ** attempt)

        raise RuntimeError(
            f"Kite 1H request failed for {symbol}"
        ) from last_error


# =============================================================================
# SUPERTREND
# =============================================================================

def completed_candles(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Remove the current in-progress candle.

    Kite timestamps are candle start times. A candle is considered completed
    only after one full hour has elapsed from its timestamp.
    """
    now = now_utc().replace(tzinfo=None)
    result = []

    for row in rows:
        raw_dt = row.get("date")
        dt = normalize_dt(raw_dt)
        if dt is None:
            continue

        if dt + timedelta(hours=1) > now:
            continue

        try:
            result.append(
                {
                    "timestamp": dt,
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                }
            )
        except (KeyError, TypeError, ValueError):
            continue

    result.sort(key=lambda item: item["timestamp"])

    deduped = {}
    for row in result:
        deduped[row["timestamp"]] = row

    return list(deduped.values())


def supertrend_directions(
    candles: list[dict[str, Any]],
    period: int,
    multiplier: float,
) -> list[int | None]:
    """
    Standard ATR/RMA Supertrend.

    Direction:
        +1 = bullish
        -1 = bearish
        None = not enough data
    """
    if len(candles) < period + 1:
        return [None] * len(candles)

    tr: list[float] = []
    for i, candle in enumerate(candles):
        if i == 0:
            tr.append(candle["high"] - candle["low"])
        else:
            previous_close = candles[i - 1]["close"]
            tr.append(
                max(
                    candle["high"] - candle["low"],
                    abs(candle["high"] - previous_close),
                    abs(candle["low"] - previous_close),
                )
            )

    atr: list[float | None] = [None] * len(candles)
    atr[period - 1] = sum(tr[:period]) / period

    for i in range(period, len(candles)):
        previous_atr = atr[i - 1]
        if previous_atr is None:
            continue
        atr[i] = (
            (previous_atr * (period - 1)) + tr[i]
        ) / period

    final_upper: list[float | None] = [None] * len(candles)
    final_lower: list[float | None] = [None] * len(candles)
    direction: list[int | None] = [None] * len(candles)

    for i in range(len(candles)):
        if atr[i] is None:
            continue

        hl2 = (candles[i]["high"] + candles[i]["low"]) / 2.0
        basic_upper = hl2 + multiplier * atr[i]
        basic_lower = hl2 - multiplier * atr[i]

        if i == period - 1:
            final_upper[i] = basic_upper
            final_lower[i] = basic_lower
            direction[i] = -1
            continue

        previous_upper = final_upper[i - 1]
        previous_lower = final_lower[i - 1]
        previous_close = candles[i - 1]["close"]

        if previous_upper is None or previous_lower is None:
            final_upper[i] = basic_upper
            final_lower[i] = basic_lower
            direction[i] = -1
            continue

        final_upper[i] = (
            basic_upper
            if basic_upper < previous_upper or previous_close > previous_upper
            else previous_upper
        )

        final_lower[i] = (
            basic_lower
            if basic_lower > previous_lower or previous_close < previous_lower
            else previous_lower
        )

        previous_direction = direction[i - 1]
        if previous_direction is None:
            direction[i] = -1
            continue

        if previous_direction == -1:
            direction[i] = (
                1 if candles[i]["close"] > final_upper[i]
                else -1
            )
        else:
            direction[i] = (
                -1 if candles[i]["close"] < final_lower[i]
                else 1
            )

    return direction


# =============================================================================
# CANDIDATE / TRADE LOGIC
# =============================================================================

def candidate_passes(scanner_row: dict[str, Any]) -> bool:
    """Apply picker-specific filters using only the scanner's persisted snapshot."""
    if str(scanner_row.get("signal", "")).upper() != "LONG WATCH":
        return False

    score = number(scanner_row.get("financial_score"))
    if score is None or score < MIN_FINANCIAL_SCORE:
        return False

    current_pe = number(scanner_row.get("current_pe"))
    sector_pe = number(scanner_row.get("sector_pe"))
    if current_pe is None or sector_pe is None:
        return False
    if current_pe <= 0 or sector_pe <= 0:
        return False

    return current_pe < sector_pe * PE_SECTOR_MULTIPLIER


def make_candidate(
    symbol: str,
    scanner_row: dict[str, Any],
    selected_at: datetime,
) -> Candidate:
    score = number(scanner_row.get("financial_score")) or 0.0

    return Candidate(
        symbol=symbol,
        name=str(scanner_row.get("name") or symbol),
        selected_at=iso(selected_at),
        financial_score=score,
        financial_grade=str(scanner_row.get("financial_grade") or "N/A"),
        financial_status=str(scanner_row.get("financial_status") or "INCOMPLETE"),
        equilibrium_distance=number(scanner_row.get("equilibrium_distance")) or 0.0,
        scanner_distance_percent=number(scanner_row.get("distance_percent")) or 0.0,
        scanner_price=number(scanner_row.get("price")) or 0.0,
        scanner_level=number(scanner_row.get("level")) or 0.0,
        present_pe=number(scanner_row.get("current_pe")),
        sector_pe=number(scanner_row.get("sector_pe")),
        historical_pe=number(scanner_row.get("historical_pe")),
        pe_status=str(scanner_row.get("pe_status") or "FAIL"),
        scanner_run_at=str(scanner_row.get("scanner_run_at") or ""),
        scanner_version=str(scanner_row.get("scanner_version") or ""),
        daily_data_as_of=str(scanner_row.get("daily_data_as_of") or ""),
        expires_at=iso(selected_at + timedelta(days=WATCHING_EXPIRY_DAYS)),
    )


def trade_id(symbol: str, entry_time: datetime) -> str:
    return f"{symbol}_{entry_time.strftime('%Y%m%d%H%M')}"


def create_trade(
    candidate: Candidate,
    candle: dict[str, Any],
) -> Trade:
    entry = float(candle["close"])
    entry_detected_at = candle["timestamp"].replace(tzinfo=timezone.utc)
    signal_expires_at = (
        entry_detected_at + timedelta(days=TRADE_EXPIRY_DAYS)
    )

    return Trade(
        trade_id=trade_id(candidate.symbol, candle["timestamp"]),
        symbol=candidate.symbol,
        name=candidate.name,
        selected_at=candidate.selected_at,
        entry_time=iso(entry_detected_at),
        entry=round(entry, 2),
        stop_loss=round(entry * (1.0 - STOP_LOSS_PCT), 2),
        target=round(entry * (1.0 + TARGET_PCT), 2),
        financial_score=candidate.financial_score,
        financial_grade=candidate.financial_grade,
        equilibrium_distance=candidate.equilibrium_distance,
        financial_status=candidate.financial_status,
        present_pe=candidate.present_pe,
        sector_pe=candidate.sector_pe,
        historical_pe=candidate.historical_pe,
        pe_status=candidate.pe_status,
        scanner_run_at=candidate.scanner_run_at,
        scanner_version=candidate.scanner_version,
        daily_data_as_of=candidate.daily_data_as_of,
        entry_detected_at=iso(entry_detected_at),
        expires_at=iso(signal_expires_at),
    )


def last_supertrend_flip_info(
    candles: list[dict[str, Any]],
    directions: list[int | None],
) -> tuple[datetime | None, int | None]:
    """Return the most recent completed bearish -> bullish flip and candles ago."""
    latest_flip_index: int | None = None

    for index in range(1, len(candles)):
        if directions[index - 1] == -1 and directions[index] == 1:
            latest_flip_index = index

    if latest_flip_index is None:
        return None, None

    candles_ago = len(candles) - 1 - latest_flip_index
    return candles[latest_flip_index]["timestamp"], candles_ago


def process_candidate(
    candidate: Candidate,
    candles: list[dict[str, Any]],
    state: PickerState,
    notifications: list[str],
) -> None:
    directions = supertrend_directions(
        candles,
        SUPERTREND_PERIOD,
        SUPERTREND_MULTIPLIER,
    )

    if not candles:
        return

    # Diagnostic only: show the latest completed bearish -> bullish flip.
    last_flip_time, candles_ago = last_supertrend_flip_info(candles, directions)
    latest_direction = directions[-1] if directions else None
    latest_direction_text = (
        "BULLISH" if latest_direction == 1
        else "BEARISH" if latest_direction == -1
        else "N/A"
    )

    if last_flip_time is None:
        LOG.info(
            "%s Supertrend | latest=%s | last bearish->bullish flip=NONE",
            candidate.symbol,
            latest_direction_text,
        )
    else:
        LOG.info(
            "%s Supertrend | latest=%s | last bearish->bullish flip=%s | %d completed candle(s) ago",
            candidate.symbol,
            latest_direction_text,
            last_flip_time.strftime("%Y-%m-%d %H:%M"),
            candles_ago,
        )

    # Check only the most recent completed 1H candles. A bearish -> bullish
    # flip in this recent window is eligible even if it happened before the
    # candidate was added to the watchlist.
    start_index = max(1, len(candles) - RECENT_FLIP_CANDLES)

    for index in range(start_index, len(candles)):
        candle = candles[index]
        direction = directions[index]
        previous_direction = directions[index - 1]

        if direction is None or previous_direction is None:
            continue

        if (
            previous_direction == -1
            and direction == 1
            and candidate.trade_id is None
        ):
            trade = create_trade(candidate, candle)

            # Never duplicate the same setup.
            if trade.trade_id not in state.trades:
                state.trades[trade.trade_id] = trade
                candidate.trade_id = trade.trade_id
                candidate.status = "ACTIVE"

                notifications.append(
                    format_entry_message(trade)
                )

            candidate.last_supertrend_direction = direction
            candidate.last_processed_candle = candle["timestamp"].isoformat()
            break

    # Record the latest completed candle processed by this run.
    if candidate.trade_id is None:
        latest_index = len(candles) - 1
        latest_direction = directions[latest_index]
        candidate.last_supertrend_direction = latest_direction
        candidate.last_processed_candle = candles[latest_index]["timestamp"].isoformat()

    # Candidate remains ACTIVE through its linked trade.
    if candidate.trade_id:
        candidate.status = "ACTIVE"


def process_active_trade(
    trade: Trade,
    candles: list[dict[str, Any]],
    candidate: Candidate | None,
    notifications: list[str],
) -> None:
    if trade.status != "ACTIVE":
        return

    entry_time = parse_dt(trade.entry_time).replace(tzinfo=None)
    expiry = parse_dt(trade.expires_at)

    # Only candles after the entry candle can hit SL/target.
    for candle in candles:
        if candle["timestamp"] <= entry_time:
            continue

        candle_time_utc = candle["timestamp"].replace(tzinfo=timezone.utc)

        if candle_time_utc > now_utc():
            continue

        # Same-candle rule: TARGET WINS if both are touched.
        hit_target = candle["high"] >= trade.target
        hit_sl = candle["low"] <= trade.stop_loss

        if hit_target:
            trade.status = "TARGET_HIT"
            trade.exit_time = iso(candle_time_utc)
            trade.exit_price = trade.target
            trade.exit_reason = "TARGET"
            notifications.append(format_exit_message(trade))
            if candidate:
                candidate.status = "TARGET_HIT"
            return

        if hit_sl:
            trade.status = "SL_HIT"
            trade.exit_time = iso(candle_time_utc)
            trade.exit_price = trade.stop_loss
            trade.exit_reason = "SL"
            notifications.append(format_exit_message(trade))
            if candidate:
                candidate.status = "SL_HIT"
            return

    if now_utc() >= expiry:
        trade.status = "EXPIRED"
        trade.exit_time = iso(now_utc())
        trade.exit_price = None
        trade.exit_reason = "30_DAY_EXPIRY"
        notifications.append(format_exit_message(trade))
        if candidate:
            candidate.status = "EXPIRED"


# =============================================================================
# TELEGRAM
# =============================================================================

def load_telegram_config() -> tuple[str | None, str | None]:
    """Load Telegram credentials directly from AWS SSM."""
    if not TELEGRAM_ENABLED:
        return None, None

    try:
        import boto3  # type: ignore

        ssm = boto3.client("ssm", region_name=AWS_REGION)
        token = ssm.get_parameter(
            Name="/trading/telegram_bot_token",
            WithDecryption=True,
        )["Parameter"]["Value"]
        return token, "419846617"
    except Exception as exc:
        LOG.warning("Telegram SSM lookup failed: %s", exc)
        return None, None


def send_telegram(messages: list[str]) -> None:
    if not messages or not TELEGRAM_ENABLED:
        return

    token, chat_id = load_telegram_config()
    if not token or not chat_id:
        LOG.info("Telegram skipped: token or chat ID not configured")
        return

    import requests  # type: ignore

    for message in messages:
        response = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={
                "chat_id": chat_id,
                "text": message,
            },
            timeout=15,
        )
        response.raise_for_status()


def format_entry_message(trade: Trade) -> str:
    return (
        f"PICKER ENTRY — {trade.symbol}\n"
        f"Entry: {trade.entry:.2f}\n"
        f"Target: {trade.target:.2f}\n"
        f"SL: {trade.stop_loss:.2f}\n"
        f"Financial Score: {trade.financial_score:.2f}\n"
        f"Eq Distance: {trade.equilibrium_distance:.2f}%\n"
        f"Present PE: {fmt(trade.present_pe)}\n"
        f"Sector PE: {fmt(trade.sector_pe)}\n"
        f"Historical PE ({HISTORICAL_PE_YEARS}Y): {fmt(trade.historical_pe)}\n"
        f"PE Status: {trade.pe_status}\n"
        f"Status: ACTIVE"
    )


def format_exit_message(trade: Trade) -> str:
    return (
        f"PICKER {trade.status} — {trade.symbol}\n"
        f"Entry: {trade.entry:.2f}\n"
        f"Target: {trade.target:.2f}\n"
        f"SL: {trade.stop_loss:.2f}\n"
        f"Exit: {fmt(trade.exit_price)}\n"
        f"Reason: {trade.exit_reason}\n"
        f"Financial Score: {trade.financial_score:.2f}"
    )


# =============================================================================
# HISTORY CSV
# =============================================================================

HISTORY_FIELDS = [
    "trade_id",
    "symbol",
    "name",
    "selected_at",
    "entry_time",
    "entry_detected_at",
    "entry",
    "stop_loss",
    "target",
    "status",
    "exit_time",
    "exit_price",
    "exit_reason",
    "expires_at",
    "financial_score",
    "financial_grade",
    "equilibrium_distance",
    "present_pe",
    "sector_pe",
    "historical_pe",
    "pe_status",
    "scanner_run_at",
    "scanner_version",
    "daily_data_as_of",
    "financial_status",
]


def trade_history_row(trade: Trade) -> dict[str, Any]:
    row = asdict(trade)
    row.pop("telegram_entry_sent", None)
    row.pop("telegram_exit_sent", None)
    normalized = {field: "" for field in HISTORY_FIELDS}
    for field in HISTORY_FIELDS:
        value = row.get(field)
        if value is None:
            normalized[field] = ""
        elif isinstance(value, datetime):
            normalized[field] = iso(value)
        else:
            normalized[field] = value
    return normalized


def history_csv(state: PickerState, existing_csv: str | None = None) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=HISTORY_FIELDS,
        lineterminator="\n",
    )

    rows_by_trade_id: dict[str, dict[str, Any]] = {}
    if existing_csv:
        try:
            reader = csv.DictReader(io.StringIO(existing_csv))
            if reader.fieldnames:
                for row in reader:
                    trade_id = str(row.get("trade_id") or "").strip()
                    if not trade_id:
                        continue
                    rows_by_trade_id[trade_id] = {
                        field: (row.get(field, "") or "") for field in HISTORY_FIELDS
                    }
        except (csv.Error, ValueError):
            rows_by_trade_id = {}

    for trade in sorted(state.trades.values(), key=lambda item: item.entry_time):
        rows_by_trade_id[trade.trade_id] = trade_history_row(trade)

    ordered_rows = sorted(
        rows_by_trade_id.values(),
        key=lambda row: (str(row.get("entry_time") or ""), str(row.get("trade_id") or "")),
    )

    writer.writeheader()
    for row in ordered_rows:
        writer.writerow({field: row.get(field, "") for field in HISTORY_FIELDS})

    return output.getvalue()


def save_history(state: PickerState) -> None:
    existing_csv = load_s3_text(PICKER_HISTORY_CSV_KEY)
    save_s3_text(
        PICKER_HISTORY_CSV_KEY,
        history_csv(state, existing_csv=existing_csv),
        "text/csv",
    )


# =============================================================================
# DISPLAY
# =============================================================================

def print_report(state: PickerState) -> None:
    print()
    print("=" * 170)
    print("PICKER — CURRENT WATCHING / ACTIVE")
    print("=" * 170)

    headers = [
        "Symbol",
        "Entry",
        "Target",
        "SL",
        "Status",
        "FinScore",
        "EqDist%",
        "PresentPE",
        "SectorPE",
        "HistPE",
        "PEStatus",
    ]

    print(" | ".join(headers))
    print("-" * 170)

    active_trade_symbols = set()

    for trade in sorted(
        state.trades.values(),
        key=lambda item: item.entry_time,
        reverse=True,
    ):
        if trade.status == "ACTIVE":
            active_trade_symbols.add(trade.symbol)
            print(
                f"{trade.symbol:<12} | "
                f"{trade.entry:>8.2f} | "
                f"{trade.target:>8.2f} | "
                f"{trade.stop_loss:>8.2f} | "
                f"{trade.status:<12} | "
                f"{trade.financial_score:>7.2f} | "
                f"{trade.equilibrium_distance:>7.2f} | "
                f"{fmt(trade.present_pe):>9} | "
                f"{fmt(trade.sector_pe):>8} | "
                f"{fmt(trade.historical_pe):>8} | "
                f"{trade.pe_status:<8}"
            )

    for candidate in sorted(
        state.candidates.values(),
        key=lambda item: item.symbol,
    ):
        if candidate.status == "WATCHING":
            print(
                f"{candidate.symbol:<12} | "
                f"{'WAIT':>8} | "
                f"{'WAIT':>8} | "
                f"{'WAIT':>8} | "
                f"{candidate.status:<12} | "
                f"{candidate.financial_score:>7.2f} | "
                f"{candidate.equilibrium_distance:>7.2f} | "
                f"{fmt(candidate.present_pe):>9} | "
                f"{fmt(candidate.sector_pe):>8} | "
                f"{fmt(candidate.historical_pe):>8} | "
                f"{candidate.pe_status:<8}"
            )

    print()
    print(
        f"Permanent trades: {len(state.trades)} | "
        f"Active: {sum(1 for t in state.trades.values() if t.status == 'ACTIVE')} | "
        f"Target: {sum(1 for t in state.trades.values() if t.status == 'TARGET_HIT')} | "
        f"SL: {sum(1 for t in state.trades.values() if t.status == 'SL_HIT')} | "
        f"Expired: {sum(1 for t in state.trades.values() if t.status == 'EXPIRED')}"
    )
    print()


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    LOG.info("Starting picker")

    state = load_picker_state()
    scanner = load_scanner_candidates()

    selected_at = now_utc()

    # -------------------------------------------------------------------------
    # 1. Add NEW scanner candidates.
    # -------------------------------------------------------------------------
    for symbol, scanner_row in scanner.items():
        # Never replace an existing candidate/trade.
        if symbol in state.candidates:
            continue

        if not candidate_passes(scanner_row):
            LOG.info(
                "%s rejected by picker filters | signal=%s | score=%s | PE=%s",
                symbol,
                scanner_row.get("signal"),
                fmt(number(scanner_row.get("financial_score"))),
                scanner_row.get("pe_status"),
            )
            continue

        candidate = make_candidate(
            symbol,
            scanner_row,
            selected_at,
        )
        state.candidates[symbol] = candidate

        LOG.info(
            "%s added to hourly watch | score=%.2f | PE=%s | sector PE=%s | hist PE=%s",
            symbol,
            candidate.financial_score,
            fmt(candidate.present_pe),
            fmt(candidate.sector_pe),
            fmt(candidate.historical_pe),
        )

    # -------------------------------------------------------------------------
    # 2. Fetch 1H candles for WATCHING candidates and detect first flip.
    # -------------------------------------------------------------------------
    kite = KiteClient()
    notifications: list[str] = []

    for candidate in list(state.candidates.values()):
        if candidate.status != "WATCHING":
            continue

        if now_utc() >= parse_dt(candidate.expires_at):
            candidate.status = "EXPIRED"
            continue

        try:
            rows = kite.historical_1h(candidate.symbol)
            candles = completed_candles(rows)

            process_candidate(
                candidate,
                candles,
                state,
                notifications,
            )
        except Exception as exc:
            LOG.exception(
                "Hourly scan failed for %s: %s",
                candidate.symbol,
                exc,
            )

    # -------------------------------------------------------------------------
    # 3. Monitor every ACTIVE trade.
    # -------------------------------------------------------------------------
    for trade in list(state.trades.values()):
        if trade.status != "ACTIVE":
            continue

        candidate = state.candidates.get(trade.symbol)

        try:
            rows = kite.historical_1h(trade.symbol)
            candles = completed_candles(rows)

            process_active_trade(
                trade,
                candles,
                candidate,
                notifications,
            )
        except Exception as exc:
            LOG.exception(
                "Trade monitoring failed for %s: %s",
                trade.symbol,
                exc,
            )

    # -------------------------------------------------------------------------
    # 4. Telegram only for actual entry/outcome events.
    # -------------------------------------------------------------------------
    if notifications:
        try:
            send_telegram(notifications)
        except Exception as exc:
            LOG.exception("Telegram send failed: %s", exc)

    # -------------------------------------------------------------------------
    # 5. Persist state + permanent history.
    # -------------------------------------------------------------------------
    save_picker_state(state)
    save_history(state)

    # -------------------------------------------------------------------------
    # 6. Print current experiment state.
    # -------------------------------------------------------------------------
    print_report(state)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
