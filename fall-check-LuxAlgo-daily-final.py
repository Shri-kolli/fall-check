#!/usr/bin/env python3
"""Daily NIFTY 250 long-term watchlist scanner.

This is an idea generator, not a trading system. It never places orders.
LuxAlgo attribution: Smart Money Concepts by LuxAlgo, licensed under
CC BY-NC-SA 4.0. This implementation intentionally mirrors the relevant
observed LuxAlgo swing-state logic and trailing-extremes labeling rather than
replacing it with a generic indicator approximation.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

# ==================================== CONFIG ====================================
TIMEFRAME = "day"
UNIVERSE = "NIFTY250"
UNIVERSE_FILE = os.getenv("UNIVERSE_FILE", "nifty250.csv")
FINANCIALS_FILE = os.getenv("FINANCIALS_FILE", "financials.json")
THIRD_PARTY_FILE = os.getenv("THIRD_PARTY_FILE", "third_party_scores.json")
STATE_FILE = os.getenv("STATE_FILE", "watchlist-state.json")

SWING_LENGTH = 50
SMC_LOOKBACK_DAYS = 1300
STRONG_LOW_ENTRY_DISTANCE = 10.0
WEAK_HIGH_ENTRY_DISTANCE = 5.0
MAX_LEVEL_BREAK_DISTANCE = 0.0
MAX_EQUILIBRIUM_DISTANCE = 15.0
WATCHLIST_RETENTION_DAYS = 30
SCREENER_CACHE_TTL_DAYS = 7
MIN_FINANCIAL_SCORE = 65.0
FINANCIAL_LOOKBACK_YEARS = 5
MIN_REVENUE_CAGR_5Y = 30.0
MIN_PAT_CAGR_5Y = 30.0
MIN_EPS_CAGR_5Y = 30.0
MAX_DECLINING_YEARS = 1
MAX_STAGNANT_YEARS = 1
MAX_SINGLE_YEAR_DECLINE = 20.0
MAX_RECOVERY_BONUS = 5.0
MIN_ROE = 12.0
MIN_ROCE = 15.0
MIN_OPERATING_MARGIN = 10.0
REJECT_INCOMPLETE_FINANCIAL_DATA = True

GROWTH_WEIGHT = 30.0
CONSISTENCY_WEIGHT = 20.0
PROFITABILITY_WEIGHT = 20.0
CASHFLOW_WEIGHT = 15.0
BALANCE_SHEET_WEIGHT = 15.0

S3_BUCKET = os.getenv("S3_BUCKET", "shri-trading-bot-storage")
S3_KEY = os.getenv("S3_KEY", "holding/state.json")
S3_CHECK_CSV_KEY = os.getenv("S3_CHECK_CSV_KEY", "holding/state-check.csv")
SSM_PARAM = os.getenv("SSM_PARAM", "/trading/zerodha_access_token")
KITE_API_KEY_PARAM = os.getenv("KITE_API_KEY_PARAM", "/trading/zerodha_api_key")
KITE_ACCESS_TOKEN_PARAM = os.getenv("KITE_ACCESS_TOKEN_PARAM", SSM_PARAM)

LOG = logging.getLogger("fall_check")


def _finite(value: Any) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _pct_change(start: float, end: float) -> float | None:
    if not _finite(start) or not _finite(end) or start <= 0:
        return None
    return (end / start - 1.0) * 100.0


def _cagr(values: Sequence[float]) -> float | None:
    if len(values) < 2 or any(not _finite(value) or value <= 0 for value in values):
        return None
    return ((values[-1] / values[0]) ** (1.0 / (len(values) - 1)) - 1.0) * 100.0


@dataclass(frozen=True)
class Stock:
    symbol: str
    name: str
    universe: str = UNIVERSE


@dataclass(frozen=True)
class DailyCandle:
    timestamp: date
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


@dataclass(frozen=True)
class DailyMarketData:
    stock: Stock
    candles: tuple[DailyCandle, ...]


@dataclass(frozen=True)
class SMCStructure:
    swing_trend: str
    swing_high: float | None
    swing_low: float | None
    strong_high: float | None
    weak_high: float | None
    strong_low: float | None
    weak_low: float | None
    equilibrium: float | None
    equilibrium_distance: float | None
    last_structure_event: str | None
    trailing_high: float | None = None
    trailing_low: float | None = None
    last_structure_event_time: date | None = None
    strong_high_time: date | None = None
    weak_high_time: date | None = None
    strong_low_time: date | None = None
    weak_low_time: date | None = None
    last_swing_event: str | None = None
    last_swing_event_date: date | None = None
    last_swing_event_price: float | None = None


@dataclass(frozen=True)
class FinancialMetrics:
    revenue: tuple[float, ...]
    pat: tuple[float, ...]
    eps: tuple[float, ...]
    roe: tuple[float, ...] = ()
    roce: tuple[float, ...] = ()
    operating_margin: tuple[float, ...] = ()
    operating_cash_flow: tuple[float, ...] = ()
    debt: tuple[float, ...] = ()
    interest_coverage: tuple[float, ...] = ()


@dataclass(frozen=True)
class FinancialScore:
    score: float
    grade: str
    status: str
    revenue_cagr: float | None
    pat_cagr: float | None
    eps_cagr: float | None
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class ThirdPartyScores:
    trendlyne_durability: float | None = None
    stockedge_fundamental: float | None = None


@dataclass
class WatchlistEntry:
    symbol: str
    name: str
    signal: str
    price: float
    level: float
    distance_percent: float
    financial_score: float
    financial_grade: str
    trendlyne_durability: float | None
    stockedge_score: float | None
    equilibrium_distance: float
    created_at: str
    expires_at: str
    status: str = "WATCHING"
    cycle: int = 1


@dataclass(frozen=True)
class BacktestRecord:
    symbol: str
    signal_date: date
    price: float
    strong_low: float | None
    weak_high: float | None
    financial_score: float
    equilibrium_distance: float | None
    signal: str
    returns: Mapping[str, float | None]
    maximum_drawdown: float | None


class MarketDataProvider(Protocol):
    def daily(self, stock: Stock, days: int) -> DailyMarketData: ...


class FinancialDataProvider(Protocol):
    def metrics(self, stock: Stock) -> FinancialMetrics | None: ...


class ThirdPartyScoreProvider(Protocol):
    def scores(self, stock: Stock) -> ThirdPartyScores: ...


def load_universe(path: str) -> list[Stock]:
    """Load only the explicit configured universe."""
    result: list[Stock] = []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            symbol = (row.get("symbol") or row.get("ticker") or "").strip().upper()
            if symbol and not symbol.startswith("#"):
                result.append(Stock(symbol, (row.get("name") or symbol).strip()))
    if not result:
        raise ValueError(f"Universe file {path!r} has no symbols")
    if Path(path).name.lower().startswith("nifty250") and len(result) != 250:
        raise ValueError(f"Universe file {path!r} should contain 250 symbols, found {len(result)}")
    return result


def _tuple_numbers(value: Any) -> tuple[float, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(float(item) for item in value if _finite(item))


class JsonFinancialDataProvider:
    def __init__(self, path: str) -> None:
        with open(path, encoding="utf-8") as handle:
            self.data: Mapping[str, Mapping[str, Any]] = json.load(handle)

    def metrics(self, stock: Stock) -> FinancialMetrics | None:
        item = self.data.get(stock.symbol)
        if not item:
            return None
        return FinancialMetrics(
            revenue=_tuple_numbers(item.get("revenue")),
            pat=_tuple_numbers(item.get("pat")),
            eps=_tuple_numbers(item.get("eps")),
            roe=_tuple_numbers(item.get("roe")),
            roce=_tuple_numbers(item.get("roce")),
            operating_margin=_tuple_numbers(item.get("operating_margin")),
            operating_cash_flow=_tuple_numbers(item.get("operating_cash_flow")),
            debt=_tuple_numbers(item.get("debt")),
            interest_coverage=_tuple_numbers(item.get("interest_coverage")),
        )


class ScreenerFinancialDataProvider:
    """Fetch and parse Screener.in fundamentals without forcing a local JSON export."""

    def __init__(self, cache_path: str | None = None, timeout: int = 15, base_url: str | None = None, cache_ttl_days: int = SCREENER_CACHE_TTL_DAYS) -> None:
        self.cache_path = cache_path or os.getenv("SCREENER_CACHE_FILE", "financials-cache.json")
        self.timeout = timeout
        self.base_url = base_url or os.getenv("SCREENER_BASE_URL", "https://www.screener.in/company/{symbol}/consolidated/")
        self.cache_ttl_days = cache_ttl_days
        self._cache: dict[str, Mapping[str, Any]] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        path = Path(self.cache_path)
        if not path.exists():
            return
        try:
            with path.open("r", encoding="utf-8") as handle:
                raw = json.load(handle)
            if isinstance(raw, dict):
                self._cache = dict(raw)
        except (OSError, ValueError, json.JSONDecodeError):
            self._cache = {}

    def _save_cache(self) -> None:
        path = Path(self.cache_path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as handle:
                json.dump(self._cache, handle, indent=2, sort_keys=True)
        except OSError:
            LOG.warning("Unable to persist Screener cache to %s", self.cache_path)

    def _cache_valid(self, payload: Mapping[str, Any]) -> bool:
        fetched_at = payload.get("fetched_at")
        if not fetched_at:
            return False
        try:
            fetched = datetime.fromisoformat(str(fetched_at))
        except ValueError:
            return False
        age = datetime.utcnow() - fetched
        return age < timedelta(days=self.cache_ttl_days)

    @staticmethod
    def _normalize_label(value: str) -> str:
        return re.sub(r"\s+", " ", value.strip().lower())

    @staticmethod
    def _parse_number(value: str) -> float | None:
        text = value.strip()
        if not text:
            return None
        if text.lower().endswith("cr"):
            text = text[:-2]
        elif text.lower().endswith("l"):
            text = text[:-1]
        if "%" in text:
            text = text.replace("%", "")
        text = text.replace(",", "")
        text = re.sub(r"[^0-9.\-]", "", text)
        if text in {"", ".", "-", "-."}:
            return None
        try:
            return float(text)
        except ValueError:
            return None

    @staticmethod
    def _looks_like_annual_header(header_cells: Sequence[str]) -> bool:
        cells = [cell for cell in header_cells if cell]
        if not cells:
            return False
        count = 0
        for cell in cells:
            text = cell.lower()
            if "ttm" in text:
                continue
            if re.search(r"\b(mar|march)\s+\d{4}\b", text):
                count += 1
            elif re.search(r"\b\d{4}\b", text):
                count += 1
        return count >= max(2, len(cells) // 2)

    @staticmethod
    def _detect_ttm_index(header_cells: Sequence[str]) -> int | None:
        for index, cell in enumerate(header_cells):
            if "ttm" in cell.lower():
                return index
        return None

    @staticmethod
    def _field_for_label(label: str) -> str | None:
        normalized = ScreenerFinancialDataProvider._normalize_label(label)
        field_aliases = {
            "revenue": "revenue",
            "sales": "revenue",
            "sales +": "revenue",
            "net profit": "pat",
            "net profit +": "pat",
            "profit after tax": "pat",
            "profit after tax +": "pat",
            "eps": "eps",
            "eps in rs": "eps",
            "earning per share": "eps",
            "roe": "roe",
            "return on equity": "roe",
            "roce": "roce",
            "roce %": "roce",
            "return on capital employed": "roce",
            "operating margin": "operating_margin",
            "operating profit margin": "operating_margin",
            "cash from operating activity": "operating_cash_flow",
            "cash from operations": "operating_cash_flow",
            "ocf": "operating_cash_flow",
            "borrowings": "debt",
            "borrowings +": "debt",
            "debt": "debt",
            "total debt": "debt",
            "interest coverage": "interest_coverage",
            "interest cover": "interest_coverage",
        }
        for alias, canonical in field_aliases.items():
            if alias in normalized or normalized.startswith(alias):
                return canonical
        return None

    @staticmethod
    def _extract_metric_rows(table: Any) -> dict[str, list[float]]:
        values_by_field: dict[str, list[float]] = {}
        rows = table.find_all("tr")
        if not rows:
            return values_by_field

        header_cells = [cell.get_text(" ", strip=True) for cell in rows[0].find_all(["th", "td"]) if cell.get_text(strip=True)]
        if not header_cells:
            return values_by_field
        if not ScreenerFinancialDataProvider._looks_like_annual_header(header_cells[1:]):
            return values_by_field

        ttm_index = ScreenerFinancialDataProvider._detect_ttm_index(header_cells[1:])
        for row in rows[1:]:
            cells = [cell.get_text(" ", strip=True) for cell in row.find_all(["th", "td"]) if cell.get_text(strip=True)]
            if len(cells) <= 1:
                continue
            label = cells[0]
            key = ScreenerFinancialDataProvider._field_for_label(label)
            if key is None:
                continue
            parsed: list[float] = []
            for idx, cell in enumerate(cells[1:], start=0):
                if ttm_index is not None and idx == ttm_index:
                    continue
                text = cell.strip()
                if not text or "ttm" in text.lower():
                    continue
                number = ScreenerFinancialDataProvider._parse_number(text)
                if number is not None:
                    parsed.append(number)
            if parsed:
                values_by_field[key] = parsed
        return values_by_field

    @staticmethod
    def _parse_html_metrics(raw_html: str) -> FinancialMetrics | None:
        try:
            from bs4 import BeautifulSoup  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Install beautifulsoup4 to use the Screener provider") from exc

        soup = BeautifulSoup(raw_html, "html.parser")
        metrics: dict[str, tuple[float, ...]] = {"revenue": (), "pat": (), "eps": (), "roe": (), "roce": (), "operating_margin": (), "operating_cash_flow": (), "debt": (), "interest_coverage": ()}

        for table in soup.find_all("table"):
            row_values = ScreenerFinancialDataProvider._extract_metric_rows(table)
            for key, values in row_values.items():
                if values:
                    metrics[key] = tuple(values[-5:]) if len(values) > 5 else tuple(values)

        required = ("revenue", "pat", "eps")
        if not all(metrics[key] for key in required):
            return None

        return FinancialMetrics(
            revenue=metrics["revenue"],
            pat=metrics["pat"],
            eps=metrics["eps"],
            roe=metrics["roe"],
            roce=metrics["roce"],
            operating_margin=metrics["operating_margin"],
            operating_cash_flow=metrics["operating_cash_flow"],
            debt=metrics["debt"],
            interest_coverage=metrics["interest_coverage"],
        )

    def _fetch_html(self, stock: Stock) -> str:
        try:
            import requests  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Install requests to use the Screener provider") from exc
        url = self.base_url.format(symbol=stock.symbol)
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
        session = requests.Session()
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = session.get(url, headers=headers, timeout=self.timeout)
                response.raise_for_status()
                return response.text
            except Exception as exc:  # pragma: no cover - network path
                last_error = exc
                if attempt < 2:
                    time.sleep(0.75 * (attempt + 1))
        raise RuntimeError(f"Screener request for {stock.symbol} failed after retries") from last_error

    def metrics(self, stock: Stock, force_refresh: bool = False) -> FinancialMetrics | None:
        cached = self._cache.get(stock.symbol)
        if cached is not None and self._cache_valid(cached) and not force_refresh:
            return FinancialMetrics(
                revenue=_tuple_numbers(cached.get("metrics", {}).get("revenue")),
                pat=_tuple_numbers(cached.get("metrics", {}).get("pat")),
                eps=_tuple_numbers(cached.get("metrics", {}).get("eps")),
                roe=_tuple_numbers(cached.get("metrics", {}).get("roe")),
                roce=_tuple_numbers(cached.get("metrics", {}).get("roce")),
                operating_margin=_tuple_numbers(cached.get("metrics", {}).get("operating_margin")),
                operating_cash_flow=_tuple_numbers(cached.get("metrics", {}).get("operating_cash_flow")),
                debt=_tuple_numbers(cached.get("metrics", {}).get("debt")),
                interest_coverage=_tuple_numbers(cached.get("metrics", {}).get("interest_coverage")),
            )
        stale = cached if cached is not None else None
        try:
            parsed = self._parse_html_metrics(self._fetch_html(stock))
        except Exception as exc:
            LOG.warning("Screener lookup failed for %s: %s", stock.symbol, exc)
            if stale is not None:
                LOG.warning("Using stale Screener cache for %s", stock.symbol)
                return FinancialMetrics(
                    revenue=_tuple_numbers(stale.get("metrics", {}).get("revenue")),
                    pat=_tuple_numbers(stale.get("metrics", {}).get("pat")),
                    eps=_tuple_numbers(stale.get("metrics", {}).get("eps")),
                    roe=_tuple_numbers(stale.get("metrics", {}).get("roe")),
                    roce=_tuple_numbers(stale.get("metrics", {}).get("roce")),
                    operating_margin=_tuple_numbers(stale.get("metrics", {}).get("operating_margin")),
                    operating_cash_flow=_tuple_numbers(stale.get("metrics", {}).get("operating_cash_flow")),
                    debt=_tuple_numbers(stale.get("metrics", {}).get("debt")),
                    interest_coverage=_tuple_numbers(stale.get("metrics", {}).get("interest_coverage")),
                )
            return None
        if parsed is None:
            if stale is not None:
                LOG.warning("Using stale Screener cache for %s after parse failure", stock.symbol)
                return FinancialMetrics(
                    revenue=_tuple_numbers(stale.get("metrics", {}).get("revenue")),
                    pat=_tuple_numbers(stale.get("metrics", {}).get("pat")),
                    eps=_tuple_numbers(stale.get("metrics", {}).get("eps")),
                    roe=_tuple_numbers(stale.get("metrics", {}).get("roe")),
                    roce=_tuple_numbers(stale.get("metrics", {}).get("roce")),
                    operating_margin=_tuple_numbers(stale.get("metrics", {}).get("operating_margin")),
                    operating_cash_flow=_tuple_numbers(stale.get("metrics", {}).get("operating_cash_flow")),
                    debt=_tuple_numbers(stale.get("metrics", {}).get("debt")),
                    interest_coverage=_tuple_numbers(stale.get("metrics", {}).get("interest_coverage")),
                )
            return None
        payload = {
            "fetched_at": datetime.utcnow().isoformat(),
            "metrics": {
                "revenue": list(parsed.revenue),
                "pat": list(parsed.pat),
                "eps": list(parsed.eps),
                "roe": list(parsed.roe),
                "roce": list(parsed.roce),
                "operating_margin": list(parsed.operating_margin),
                "operating_cash_flow": list(parsed.operating_cash_flow),
                "debt": list(parsed.debt),
                "interest_coverage": list(parsed.interest_coverage),
            },
        }
        self._cache[stock.symbol] = payload
        self._save_cache()
        return parsed


ScreenerFinancialProvider = ScreenerFinancialDataProvider


class JsonThirdPartyScoreProvider:
    def __init__(self, path: str) -> None:
        self.data: Mapping[str, Mapping[str, Any]] = {}
        if Path(path).exists():
            with open(path, encoding="utf-8") as handle:
                self.data = json.load(handle)

    def scores(self, stock: Stock) -> ThirdPartyScores:
        item = self.data.get(stock.symbol, {})
        return ThirdPartyScores(item.get("trendlyne_durability"), item.get("stockedge_fundamental"))


class YahooMarketDataProvider:
    """Optional yfinance adapter. Production providers can replace this."""

    def daily(self, stock: Stock, days: int) -> DailyMarketData:
        try:
            import yfinance as yf  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Install yfinance or inject a MarketDataProvider") from exc
        frame = yf.Ticker(stock.symbol + ".NS").history(period=f"{max(days // 250 + 1, 2)}y", interval="1d", auto_adjust=False)
        candles = tuple(
            DailyCandle(index.date(), float(row.Open), float(row.High), float(row.Low), float(row.Close), float(row.Volume))
            for index, row in frame.dropna().iterrows()
        )[-days:]
        return DailyMarketData(stock, candles)


class KiteMarketDataProvider:
    """Read-only Zerodha adapter matching the existing SSM pattern."""

    def __init__(self, api_key: str | None = None, access_token: str | None = None) -> None:
        try:
            import boto3  # type: ignore
            from kiteconnect import KiteConnect  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Install boto3 and kiteconnect to use the Kite provider") from exc
        self._ssm = boto3.client("ssm", region_name=os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION"))
        self._api_key = api_key or self._secret(KITE_API_KEY_PARAM)
        self._access_token = access_token or os.getenv("ZERODHA_ACCESS_TOKEN") or self._secret(KITE_ACCESS_TOKEN_PARAM)
        self._kite = KiteConnect(api_key=self._api_key)
        self._kite.set_access_token(self._access_token)
        self._tokens: dict[str, int] = {}

    def _secret(self, name: str) -> str:
        response = self._ssm.get_parameter(Name=name, WithDecryption=True)
        return response["Parameter"]["Value"]

    def _call(self, function: Any, *args: Any, **kwargs: Any) -> Any:
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                return function(*args, **kwargs)
            except Exception as exc:
                last_error = exc
                text = str(exc).lower()
                if "token" in text or "api_key" in text or "invalid" in text:
                    raise
                if attempt < 2:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"Kite request failed after retries: {last_error}") from last_error

    def _instrument_tokens(self) -> None:
        instruments = self._call(self._kite.instruments, "NSE")
        self._tokens = {item["tradingsymbol"]: int(item["instrument_token"]) for item in instruments}

    def daily(self, stock: Stock, days: int) -> DailyMarketData:
        if not self._tokens:
            self._instrument_tokens()
        token = self._tokens.get(stock.symbol)
        if token is None:
            raise ValueError(f"No NSE instrument token found for {stock.symbol}")
        end = datetime.now()
        start = end - timedelta(days=max(days + 30, 60))
        rows = self._call(self._kite.historical_data, token, start, end, TIMEFRAME)

# Kite includes the current in-progress daily candle.
# The scanner is daily-close based, so exclude it.
        today = datetime.now().date()

        candles = tuple(
            DailyCandle(
                timestamp=(
                    row["date"].date()
                    if hasattr(row["date"], "date")
                    else date.fromisoformat(str(row["date"])[:10])
                ),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row.get("volume", 0)),
            )
            for row in rows
            if all(_finite(row.get(field)) for field in ("open", "high", "low", "close"))
            and (
                row["date"].date()
                if hasattr(row["date"], "date")
                else date.fromisoformat(str(row["date"])[:10])
            ) < today
        )[-days:]
        
        return DailyMarketData(stock, candles)        


def _detect_pivots(candles: Sequence[DailyCandle], length: int) -> tuple[list[tuple[int, float, date]], list[tuple[int, float, date]]]:
    """Return pivot highs and lows using the LuxAlgo-style local-extremum window."""
    highs: list[tuple[int, float, date]] = []
    lows: list[tuple[int, float, date]] = []
    for index in range(len(candles)):
        left = max(0, index - length)
        right = min(len(candles), index + length + 1)
        window = candles[left:right]
        local_high = candles[index].high
        local_low = candles[index].low
        if local_high == max(item.high for item in window):
            highs.append((index, local_high, candles[index].timestamp))
        if local_low == min(item.low for item in window):
            lows.append((index, local_low, candles[index].timestamp))
    return highs, lows


def calculate_smc(data: DailyMarketData, length: int = SWING_LENGTH) -> SMCStructure:
    """Faithfully reproduce the LuxAlgo swing-state logic on daily candles.

    Reference: Smart Money Concepts [LuxAlgo] Pine v5.
    The swing leg creates confirmed pivots; swing bias changes only when the
    latest stored swing high/low is crossed by price. Trailing extremes are
    reset on each new swing pivot and then extended by the current bar.
    """
    candles = data.candles
    if not candles:
        return SMCStructure(
            swing_trend="neutral", swing_high=None, swing_low=None,
            strong_high=None, weak_high=None, strong_low=None, weak_low=None,
            equilibrium=None, equilibrium_distance=None, last_structure_event=None,
        )

    # LuxAlgo globals: trend=0, top_y=0, btm_y=0, top_cross=true, btm_cross=true.
    trend = 0
    top_y = 0.0
    btm_y = 0.0
    top_x: int | None = None
    btm_x: int | None = None
    top_cross = True
    btm_cross = True

    # LuxAlgo: var trail_up = high, trail_dn = low.
    trailing_top = candles[0].high
    trailing_bottom = candles[0].low
    trailing_top_index = 0
    trailing_bottom_index = 0

    # LuxAlgo swings(len): persistent os starts at 0.
    os = 0
    previous_close: float | None = None
    previous_top_y = 0.0
    previous_btm_y = 0.0

    last_structure_event: str | None = None
    last_structure_event_time: date | None = None
    last_swing_event: str | None = None
    last_swing_event_date: date | None = None
    last_swing_event_price: float | None = None

    for index, current in enumerate(candles):
        previous_os = os
        top = 0.0
        btm = 0.0

        if index >= length:
            # Pine ta.highest(length)/lowest(length) includes the current bar
            # and the preceding length-1 bars; high[length]/low[length] is the
            # candidate pivot exactly length bars back.
            upper = max(c.high for c in candles[index - length + 1:index + 1])
            lower = min(c.low for c in candles[index - length + 1:index + 1])
            if candles[index - length].high > upper:
                os = 0
            elif candles[index - length].low < lower:
                os = 1
            else:
                os = previous_os

            if os == 0 and previous_os != 0:
                top = candles[index - length].high
            if os == 1 and previous_os != 1:
                btm = candles[index - length].low

        # Pivot High: reset trailing high first, exactly as Pine does.
        if top:
            top_cross = True
            top_y = top
            top_x = index - length
            trailing_top = top
            trailing_top_index = index - length

        # Pivot Low: reset trailing low first.
        if btm:
            btm_cross = True
            btm_y = btm
            btm_x = index - length
            trailing_bottom = btm
            trailing_bottom_index = index - length

        # Pine updates trailing extremes AFTER the pivot reset on the same bar.
        if current.high >= trailing_top:
            trailing_top = current.high
            trailing_top_index = index
        if current.low <= trailing_bottom:
            trailing_bottom = current.low
            trailing_bottom_index = index

        # Swing structure breaks: close crossing stored swing levels changes
        # trend/bias. This is where LuxAlgo creates BOS/CHoCH.
        if previous_close is not None:
            bullish_cross = previous_close <= previous_top_y and current.close > top_y and top_cross
            bearish_cross = previous_close >= previous_btm_y and current.close < btm_y and btm_cross

            if bullish_cross:
                event = "bullish CHoCH" if trend < 0 else "bullish BOS"
                top_cross = False
                trend = 1
                last_structure_event = event
                last_structure_event_time = current.timestamp
                last_swing_event = event
                last_swing_event_date = current.timestamp
                last_swing_event_price = top_y

            if bearish_cross:
                event = "bearish CHoCH" if trend > 0 else "bearish BOS"
                btm_cross = False
                trend = -1
                last_structure_event = event
                last_structure_event_time = current.timestamp
                last_swing_event = event
                last_swing_event_date = current.timestamp
                last_swing_event_price = btm_y

        previous_close = current.close
        previous_top_y = top_y
        previous_btm_y = btm_y

    # Initialize all label timestamps before trend-specific assignment.
    strong_high_time = None
    weak_high_time = None
    strong_low_time = None
    weak_low_time = None

    if trend > 0:
        swing_trend = "bullish"
        strong_low = trailing_bottom
        strong_low_time = candles[trailing_bottom_index].timestamp
        weak_high = trailing_top
        weak_high_time = candles[trailing_top_index].timestamp
        strong_high = None
        weak_low = None
    elif trend < 0:
        swing_trend = "bearish"
        strong_high = trailing_top
        strong_high_time = candles[trailing_top_index].timestamp
        weak_low = trailing_bottom
        weak_low_time = candles[trailing_bottom_index].timestamp
        strong_low = None
        weak_high = None
    else:
        swing_trend = "neutral"
        strong_high = weak_high = strong_low = weak_low = None
        strong_high_time = weak_high_time = strong_low_time = weak_low_time = None

    equilibrium = (trailing_top + trailing_bottom) / 2.0
    equilibrium_distance = None
    if equilibrium > 0:
        # This is the structural range half-width used by the scanner's
        # diagnostic, not a LuxAlgo trading rule. LuxAlgo itself defines EQ
        # simply as math.avg(trailing.top, trailing.bottom).
        equilibrium_distance = max(
            abs(trailing_top - equilibrium) / equilibrium * 100.0,
            abs(equilibrium - trailing_bottom) / equilibrium * 100.0,
        )

    return SMCStructure(
        swing_trend=swing_trend,
        swing_high=trailing_top,
        swing_low=trailing_bottom,
        strong_high=strong_high,
        weak_high=weak_high,
        strong_low=strong_low,
        weak_low=weak_low,
        equilibrium=equilibrium,
        equilibrium_distance=equilibrium_distance,
        last_structure_event=last_structure_event,
        trailing_high=trailing_top,
        trailing_low=trailing_bottom,
        last_structure_event_time=last_structure_event_time,
        strong_high_time=strong_high_time,
        weak_high_time=weak_high_time,
        strong_low_time=strong_low_time,
        weak_low_time=weak_low_time,
        last_swing_event=last_swing_event,
        last_swing_event_date=last_swing_event_date,
        last_swing_event_price=last_swing_event_price,
    )

def score_financials(metrics: FinancialMetrics | None) -> FinancialScore:
    """Score fundamentals from 0-100 without inventing missing observations."""
    if metrics is None:
        return FinancialScore(0.0, "D", "INCOMPLETE", None, None, None, ("missing financial record",))

    required = (metrics.revenue, metrics.pat, metrics.eps)
    if any(len(values) < FINANCIAL_LOOKBACK_YEARS for values in required):
        return FinancialScore(0.0, "D", "INCOMPLETE", None, None, None, ("incomplete 5-year financial history",))

    revenue_cagr = _cagr(metrics.revenue[-FINANCIAL_LOOKBACK_YEARS:])
    pat_cagr = _cagr(metrics.pat[-FINANCIAL_LOOKBACK_YEARS:])
    eps_cagr = _cagr(metrics.eps[-FINANCIAL_LOOKBACK_YEARS:])

    if None in (revenue_cagr, pat_cagr, eps_cagr):
        return FinancialScore(0.0, "D", "INCOMPLETE", revenue_cagr, pat_cagr, eps_cagr, ("non-positive starting value prevents CAGR",))

    score = 0.0

    growth_values = (
        (revenue_cagr, MIN_REVENUE_CAGR_5Y),
        (pat_cagr, MIN_PAT_CAGR_5Y),
        (eps_cagr, MIN_EPS_CAGR_5Y),
    )
    growth_component = sum(min(max(cagr / threshold, 0.0), 1.0) for cagr, threshold in growth_values) / 3.0
    score += growth_component * GROWTH_WEIGHT

    revenue_series = metrics.revenue[-FINANCIAL_LOOKBACK_YEARS:]
    trend_series = [
        ("revenue", metrics.revenue[-FINANCIAL_LOOKBACK_YEARS:]),
        ("pat", metrics.pat[-FINANCIAL_LOOKBACK_YEARS:]),
        ("eps", metrics.eps[-FINANCIAL_LOOKBACK_YEARS:]),
    ]
    available_series = [(name, values) for name, values in trend_series if len(values) >= 2]
    if not available_series:
        consistency = 0.0
    else:
        consistency_scores: list[float] = []
        for _, values in available_series:
            declines = sum(1 for before, after in zip(values, values[1:]) if after < before * (1.0 - MAX_SINGLE_YEAR_DECLINE / 100.0))
            stagnant = sum(1 for before, after in zip(values, values[1:]) if before > 0 and 0.0 <= (after / before - 1.0) * 100.0 < 5.0)
            component = 1.0 if declines <= MAX_DECLINING_YEARS and stagnant <= MAX_STAGNANT_YEARS else 0.5
            consistency_scores.append(component)
        consistency = sum(consistency_scores) / len(consistency_scores)
    score += consistency * CONSISTENCY_WEIGHT
    if len(revenue_series) >= 3 and revenue_series[-1] > revenue_series[-2] < revenue_series[-3]:
        score += MAX_RECOVERY_BONUS

    profitability_inputs: list[float] = []
    if metrics.roe:
        profitability_inputs.append(min(max(metrics.roe[-1] / MIN_ROE, 0.0), 1.0) * 8.0)
    if metrics.roce:
        profitability_inputs.append(min(max(metrics.roce[-1] / MIN_ROCE, 0.0), 1.0) * 8.0)
    if metrics.operating_margin:
        profitability_inputs.append(min(max(metrics.operating_margin[-1] / MIN_OPERATING_MARGIN, 0.0), 1.0) * 4.0)
    if profitability_inputs:
        profitability_component = sum(profitability_inputs) / 20.0
        score += profitability_component * PROFITABILITY_WEIGHT

    cashflow_score = 0.0
    if metrics.operating_cash_flow:
        ocf_trend = _pct_change(metrics.operating_cash_flow[0], metrics.operating_cash_flow[-1]) or 0.0
        cashflow_score += min(max(ocf_trend / 50.0, 0.0), 1.0) * 8.0
        positive_ocf_years = sum(1 for value in metrics.operating_cash_flow if value > 0)
        cashflow_score += min(positive_ocf_years / len(metrics.operating_cash_flow), 1.0) * 7.0
        score += min(cashflow_score, 15.0)

    debt_score = 0.0
    if metrics.debt:
        debt_score += 0.5 if metrics.debt[-1] <= metrics.debt[0] else 0.0
    if metrics.interest_coverage:
        debt_score += 0.5 if metrics.interest_coverage[-1] >= 3.0 else 0.0
    score += debt_score * BALANCE_SHEET_WEIGHT

    grade = "A+" if score >= 85 else "A" if score >= 75 else "B+" if score >= 65 else "B" if score >= 55 else "C" if score >= 45 else "D"
    return FinancialScore(round(min(score, 100.0), 2), grade, "COMPLETE", revenue_cagr, pat_cagr, eps_cagr)


def evaluate_signal(data: DailyMarketData, structure: SMCStructure, financial: FinancialScore, third_party: ThirdPartyScores, as_of: date | None = None) -> WatchlistEntry | None:
    """Apply only the scanner's proximity verdict on top of LuxAlgo structure.

    No custom equilibrium filter is applied. Equilibrium and its distances are
    diagnostics only; LuxAlgo itself defines equilibrium as the midpoint of
    the trailing high and trailing low.
    """
    if not data.candles:
        return None

    price = data.candles[-1].close
    candidates: list[tuple[str, float, float]] = []

    if structure.strong_low is not None and price >= structure.strong_low:
        distance = (price - structure.strong_low) / structure.strong_low * 100.0
        if distance <= STRONG_LOW_ENTRY_DISTANCE:
            candidates.append(("LONG WATCH", structure.strong_low, distance))

    if structure.weak_high is not None and price <= structure.weak_high:
        distance = (structure.weak_high - price) / structure.weak_high * 100.0
        if distance <= WEAK_HIGH_ENTRY_DISTANCE:
            candidates.append(("SELL / EXIT WATCH", structure.weak_high, distance))

    if not candidates:
        return None

    signal, level, distance = min(candidates, key=lambda item: item[2])
    created = as_of or date.today()
    equilibrium_distance = structure.equilibrium_distance if structure.equilibrium_distance is not None else 0.0
    return WatchlistEntry(
        symbol=data.stock.symbol,
        name=data.stock.name,
        signal=signal,
        price=price,
        level=level,
        distance_percent=round(distance, 2),
        financial_score=0.0,
        financial_grade="N/A",
        trendlyne_durability=third_party.trendlyne_durability,
        stockedge_score=third_party.stockedge_fundamental,
        equilibrium_distance=equilibrium_distance,
        created_at=created.isoformat(),
        expires_at=(created + timedelta(days=WATCHLIST_RETENTION_DAYS)).isoformat(),
    )


def smc_verdict(price: float, structure: SMCStructure) -> tuple[str, float | None, str]:
    """Return current LuxAlgo-level proximity verdict and distance."""
    candidates: list[tuple[str, float, str]] = []
    if structure.strong_low is not None and price >= structure.strong_low:
        d = (price - structure.strong_low) / structure.strong_low * 100.0
        if d <= STRONG_LOW_ENTRY_DISTANCE:
            candidates.append(("LONG WATCH", d, "Strong Low"))
    if structure.weak_high is not None and price <= structure.weak_high:
        d = (structure.weak_high - price) / structure.weak_high * 100.0
        if d <= WEAK_HIGH_ENTRY_DISTANCE:
            candidates.append(("SELL / EXIT WATCH", d, "Weak High"))
    if not candidates:
        return "NO WATCH", None, "-"
    signal, distance, level_name = min(candidates, key=lambda item: item[1])
    return signal, distance, level_name

def backtest_daily(
    data: DailyMarketData,
    financial_scores: Mapping[date, FinancialScore],
    third_party: ThirdPartyScores | None = None,
    length: int = SWING_LENGTH,
) -> list[BacktestRecord]:
    """Replay daily signals causally and measure only subsequent price data.

    ``financial_scores`` must contain the score known on each historical date;
    this function deliberately does not derive a current score for old bars.
    Future candles are used only for outcome measurement after a signal exists.
    """
    records: list[BacktestRecord] = []
    scores = third_party or ThirdPartyScores()
    minimum_bars = length * 2 + 2
    for index in range(minimum_bars, len(data.candles)):
        as_of = data.candles[index].timestamp
        financial = financial_scores.get(as_of)
        if financial is None:
            continue
        prefix = DailyMarketData(data.stock, data.candles[: index + 1])
        structure = calculate_smc(prefix, length)
        candidate = evaluate_signal(prefix, structure, financial, scores, as_of)
        if candidate is None:
            continue
        future = data.candles[index + 1:]
        returns: dict[str, float | None] = {}
        maximum_drawdown: float | None = None
        for label, days in (("1M", 21), ("3M", 63), ("6M", 126), ("1Y", 252), ("3Y", 756)):
            if len(future) >= days:
                returns[label] = (future[days - 1].close / candidate.price - 1.0) * 100.0
        if future:
            peak = candidate.price
            drawdowns: list[float] = []
            for candle in future[:756]:
                peak = max(peak, candle.close)
                drawdowns.append((candle.close / peak - 1.0) * 100.0)
            maximum_drawdown = min(drawdowns)
        records.append(BacktestRecord(data.stock.symbol, as_of, candidate.price, structure.strong_low, structure.weak_high, financial.score, structure.equilibrium_distance, candidate.signal, returns, maximum_drawdown))
    return records


class WatchlistStore:
    def __init__(self, path: str = STATE_FILE) -> None:
        self.path = path

    def load(self) -> dict[str, WatchlistEntry]:
        if not Path(self.path).exists():
            return {}
        with open(self.path, encoding="utf-8") as handle:
            raw = json.load(handle)
        allowed = set(WatchlistEntry.__dataclass_fields__)
        required = {
            "symbol",
            "name",
            "signal",
            "price",
            "level",
            "distance_percent",
            "financial_score",
            "financial_grade",
            "trendlyne_durability",
            "stockedge_score",
            "equilibrium_distance",
            "created_at",
            "expires_at",
        }

        entries: dict[str, WatchlistEntry] = {}

        for key, value in raw.items():
            if not isinstance(value, dict):
                LOG.warning("Skipping malformed watchlist entry: %s", key)
                continue

            filtered = {
                field: item
                for field, item in value.items()
                if field in allowed
            }

            missing = required - set(filtered)

            if missing:
                LOG.warning(
                    "Skipping legacy watchlist entry %s; missing=%s",
                    key,
                    sorted(missing),
                )
                continue

            try:
                entries[key] = WatchlistEntry(**filtered)
            except (TypeError, ValueError) as exc:
                LOG.warning(
                    "Skipping invalid watchlist entry %s: %s",
                    key,
                    exc,
                )

        return entries

    def save(self, entries: Mapping[str, WatchlistEntry]) -> None:
        temporary = self.path + ".tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump({key: asdict(value) for key, value in entries.items()}, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.path)

    def active(self, today: date | None = None) -> dict[str, WatchlistEntry]:
        today = today or date.today()
        return {symbol: entry for symbol, entry in self.load().items() if date.fromisoformat(entry.expires_at) >= today}

    def merge(self, candidate: WatchlistEntry, today: date | None = None) -> tuple[WatchlistEntry, str]:
        entries = self.load()
        today = today or date.today()
        existing = entries.get(candidate.symbol)
        if existing and date.fromisoformat(existing.expires_at) >= today:
            return existing, "ACTIVE"
        candidate.cycle = existing.cycle + 1 if existing else 1
        candidate.status = "RENEWED" if existing else "NEW"
        entries[candidate.symbol] = candidate
        self.save(entries)
        return candidate, candidate.status


class S3WatchlistStore(WatchlistStore):
    """Persist the same JSON schema in S3 using the existing bot settings."""

    def __init__(self, bucket: str = S3_BUCKET, key: str = S3_KEY) -> None:
        try:
            import boto3  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Install boto3 to use S3 watchlist persistence") from exc
        super().__init__(path="")
        region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
        self._s3 = boto3.client("s3", region_name=region) if region else boto3.client("s3")
        self.bucket = bucket
        self.key = key

    def load(self) -> dict[str, WatchlistEntry]:
        try:
            body = self._s3.get_object(Bucket=self.bucket, Key=self.key)["Body"].read()
        except Exception as exc:
            text = str(exc).lower()
            if "nosuchkey" in text or "not found" in text or "404" in text:
                return {}
            raise RuntimeError(f"Unable to read watchlist state from s3://{self.bucket}/{self.key}") from exc
        raw = json.loads(body.decode("utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("Watchlist state must be a JSON object")
        allowed = set(WatchlistEntry.__dataclass_fields__)
        required = {
            "symbol",
            "name",
            "signal",
            "price",
            "level",
            "distance_percent",
            "financial_score",
            "financial_grade",
            "trendlyne_durability",
            "stockedge_score",
            "equilibrium_distance",
            "created_at",
            "expires_at",
        }

        entries: dict[str, WatchlistEntry] = {}

        for key, value in raw.items():
            if not isinstance(value, dict):
                LOG.warning("Skipping malformed watchlist entry: %s", key)
                continue

            filtered = {
                field: item
                for field, item in value.items()
                if field in allowed
            }

            missing = required - set(filtered)

            if missing:
                LOG.warning(
                    "Skipping legacy watchlist entry %s; missing=%s",
                    key,
                    sorted(missing),
                )
                continue

            try:
                entries[key] = WatchlistEntry(**filtered)
            except (TypeError, ValueError) as exc:
                LOG.warning(
                    "Skipping invalid watchlist entry %s: %s",
                    key,
                    exc,
                )

        return entries

    def save(self, entries: Mapping[str, WatchlistEntry]) -> None:
        payload = json.dumps({key: asdict(value) for key, value in entries.items()}, indent=2, sort_keys=True).encode("utf-8")
        try:
            self._s3.put_object(Bucket=self.bucket, Key=self.key, Body=payload, ContentType="application/json")
        except Exception as exc:
            raise RuntimeError(f"Unable to write watchlist state to s3://{self.bucket}/{self.key}") from exc


TELEGRAM_SSM_PARAM = "/trading/telegram_bot_token"
TELEGRAM_CHAT_ID = "419846617"


def _load_secret(name: str) -> str:
    """Load an encrypted secret from AWS SSM Parameter Store."""
    try:
        import boto3  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Install boto3 to load secrets from AWS SSM") from exc
    region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
    ssm = boto3.client("ssm", region_name=region) if region else boto3.client("ssm")
    response = ssm.get_parameter(Name=name, WithDecryption=True)
    return response["Parameter"]["Value"]


def send_telegram(message: str) -> None:
    """Send the final shortlist report using the EC2 SSM Telegram secret."""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        token = _load_secret(TELEGRAM_SSM_PARAM)
    chat_id = os.getenv("TELEGRAM_CHAT_ID", TELEGRAM_CHAT_ID)
    if not token or not chat_id:
        LOG.info("Telegram disabled: Telegram token or chat ID is unavailable")
        return
    try:
        import requests  # type: ignore

        # Telegram sendMessage has a practical 4096-character message limit.
        chunks: list[str] = []
        remaining = message
        while len(remaining) > 3900:
            split_at = remaining.rfind("\n", 0, 3900)
            if split_at <= 0:
                split_at = 3900
            chunks.append(remaining[:split_at])
            remaining = remaining[split_at:].lstrip("\n")
        if remaining:
            chunks.append(remaining)

        for chunk in chunks:
            response = requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                data={"chat_id": chat_id, "text": chunk},
                timeout=10,
            )
            response.raise_for_status()
    except Exception as exc:
        raise RuntimeError("Telegram notification failed") from exc


def format_report(
    new_entries: Sequence[WatchlistEntry],
    active: Sequence[WatchlistEntry],
    expired: Sequence[WatchlistEntry] = (),
    scan_rows: Sequence[str] = (),
) -> str:
    lines = [
        "DAILY NIFTY 250 — LUXALGO SMC SWING SCAN",
        f"Swing length={SWING_LENGTH} | Daily candles | Completed candles only",
        "Equilibrium = (trailing high + trailing low) / 2. EqRange% is structural half-range; PriceEq% is current-price distance from EQ. Neither is a LuxAlgo filter.",
        "",
        "CURRENT SCAN — ALL STOCKS",
        "| Stock | Price | Trend | Strong High | Weak High | Strong Low | Weak Low | Equilibrium | EqRange% | PriceEq% | Last BOS/CHoCH | Level Dist% | Verdict |",
    ]
    lines.extend(scan_rows or ["None"])

    lines.extend(["", "CURRENT WATCHES CREATED/RENEWED THIS RUN"])
    if new_entries:
        for entry in new_entries:
            level_name = "Strong Low" if entry.signal == "LONG WATCH" else "Weak High"
            lines.append(
                f"{entry.symbol} | {entry.signal} | Price {entry.price:.2f} | {level_name} {entry.level:.2f} | Dist {entry.distance_percent:.2f}%"
            )
    else:
        lines.append("None")

    lines.extend(["", "PERSISTED ACTIVE WATCHLIST (historical state)"])
    if active:
        for entry in sorted(active, key=lambda item: (item.signal, item.distance_percent, -item.financial_score)):
            lines.append(
                f"{entry.symbol} | {entry.signal} | Current {entry.price:.2f} | Level {entry.level:.2f} | Dist {entry.distance_percent:.2f}% | Added {entry.created_at} | Expires {entry.expires_at}"
            )
    else:
        lines.append("None")

    lines.extend(["", "EXPIRED WATCHES"])
    lines.extend(f"{entry.symbol} | cycle {entry.cycle} | expired {entry.expires_at}" for entry in expired)
    if not expired:
        lines.append("None")
    return "\n".join(lines)

def format_final_financial_report(entries: Sequence[WatchlistEntry], financials: FinancialDataProvider) -> str:
    """Build the final shortlist report and refresh fundamentals for those stocks only."""
    lines = [
        "DAILY NIFTY 250 — FINAL LUXALGO SMC WATCHLIST",
        f"Swing length={SWING_LENGTH} | Daily candles | Completed candles only",
        "Equilibrium 15% remains diagnostic only and is NOT a filter.",
        "",
        f"FINAL WATCHLIST: {len(entries)} STOCK(S)",
    ]

    if not entries:
        lines.append("None")
        return "\n".join(lines)

    for entry in sorted(entries, key=lambda item: (item.signal, item.distance_percent, item.symbol)):
        stock = Stock(entry.symbol, entry.name)
        metrics = financials.metrics(stock, force_refresh=True)
        score = score_financials(metrics)

        lines.extend([
            "",
            f"{entry.symbol} | {entry.name}",
            f"Signal: {entry.signal}",
            f"Price: {entry.price:.2f}",
            f"Level: {entry.level:.2f}",
            f"Level distance: {entry.distance_percent:.2f}%",
            f"Equilibrium range: {entry.equilibrium_distance:.2f}%",
            f"Financial score: {score.score:.2f} | Grade: {score.grade} | Status: {score.status}",
        ])

        if metrics is None:
            lines.append("Financial data: UNAVAILABLE")
            continue

        def latest(values: tuple[float, ...]) -> str:
            return f"{values[-1]:.2f}" if values else "N/A"

        lines.extend([
            f"Revenue: {latest(metrics.revenue)}",
            f"PAT: {latest(metrics.pat)}",
            f"EPS: {latest(metrics.eps)}",
            f"ROE: {latest(metrics.roe)}",
            f"ROCE: {latest(metrics.roce)}",
            f"Operating margin: {latest(metrics.operating_margin)}",
            f"Operating cash flow: {latest(metrics.operating_cash_flow)}",
            f"Debt: {latest(metrics.debt)}",
            f"Interest coverage: {latest(metrics.interest_coverage)}",
            f"5Y CAGR — Revenue: {score.revenue_cagr:.2f}% | PAT: {score.pat_cagr:.2f}% | EPS: {score.eps_cagr:.2f}%" if None not in (score.revenue_cagr, score.pat_cagr, score.eps_cagr) else "5Y CAGR: N/A",
        ])

    return "\n".join(lines)


def write_state_check_csv(entries: Sequence[WatchlistEntry], financials: FinancialDataProvider) -> Path:
    """Write the current final LuxAlgo shortlist as an analysis-only CSV snapshot and upload it to S3."""
    path = Path("state-check.csv")
    fieldnames = [
        "date", "symbol", "name", "signal", "price", "level", "level_distance_pct",
        "equilibrium_range_pct", "financial_score", "financial_grade", "financial_status",
        "revenue", "pat", "eps", "roe", "roce", "operating_margin",
        "operating_cash_flow", "debt", "interest_coverage",
        "revenue_cagr_5y", "pat_cagr_5y", "eps_cagr_5y",
    ]

    def latest(values: tuple[float, ...]) -> str:
        return f"{values[-1]:.2f}" if values else ""

    rows: list[dict[str, Any]] = []
    for entry in sorted(entries, key=lambda item: (item.signal, item.distance_percent, item.symbol)):
        stock = Stock(entry.symbol, entry.name)
        metrics = financials.metrics(stock, force_refresh=False)
        score = score_financials(metrics)
        rows.append({
            "date": date.today().isoformat(),
            "symbol": entry.symbol,
            "name": entry.name,
            "signal": entry.signal,
            "price": f"{entry.price:.2f}",
            "level": f"{entry.level:.2f}",
            "level_distance_pct": f"{entry.distance_percent:.2f}",
            "equilibrium_range_pct": f"{entry.equilibrium_distance:.2f}",
            "financial_score": f"{score.score:.2f}",
            "financial_grade": score.grade,
            "financial_status": score.status,
            "revenue": latest(metrics.revenue) if metrics else "",
            "pat": latest(metrics.pat) if metrics else "",
            "eps": latest(metrics.eps) if metrics else "",
            "roe": latest(metrics.roe) if metrics else "",
            "roce": latest(metrics.roce) if metrics else "",
            "operating_margin": latest(metrics.operating_margin) if metrics else "",
            "operating_cash_flow": latest(metrics.operating_cash_flow) if metrics else "",
            "debt": latest(metrics.debt) if metrics else "",
            "interest_coverage": latest(metrics.interest_coverage) if metrics else "",
            "revenue_cagr_5y": f"{score.revenue_cagr:.2f}" if score.revenue_cagr is not None else "",
            "pat_cagr_5y": f"{score.pat_cagr:.2f}" if score.pat_cagr is not None else "",
            "eps_cagr_5y": f"{score.eps_cagr:.2f}" if score.eps_cagr is not None else "",
        })

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    try:
        import boto3  # type: ignore
        region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
        s3 = boto3.client("s3", region_name=region) if region else boto3.client("s3")
        s3.upload_file(str(path), S3_BUCKET, S3_CHECK_CSV_KEY, ExtraArgs={"ContentType": "text/csv"})
        LOG.info("State-check CSV uploaded: s3://%s/%s (%d stock(s))", S3_BUCKET, S3_CHECK_CSV_KEY, len(rows))
    except Exception as exc:
        raise RuntimeError(f"Unable to upload state-check CSV to s3://{S3_BUCKET}/{S3_CHECK_CSV_KEY}") from exc
    return path


def _daily_cache_path(symbol: str) -> Path:
    return Path("data") / "daily" / f"{symbol}.csv"


def _clean_daily_candle_row(row: Mapping[str, Any]) -> DailyCandle | None:
    try:
        timestamp = date.fromisoformat(str(row.get("date") or row.get("timestamp") or "")[:10])
    except ValueError:
        return None
    try:
        open_price = float(row.get("open", 0.0))
        high_price = float(row.get("high", 0.0))
        low_price = float(row.get("low", 0.0))
        close_price = float(row.get("close", 0.0))
        volume = float(row.get("volume", 0.0))
    except (TypeError, ValueError):
        return None
    return DailyCandle(timestamp, open_price, high_price, low_price, close_price, volume)


def save_daily_market_data(symbol: str, data: DailyMarketData) -> None:
    path = _daily_cache_path(symbol)
    path.parent.mkdir(parents=True, exist_ok=True)
    filtered = tuple(candle for candle in data.candles if candle.timestamp < date.today())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["date", "open", "high", "low", "close", "volume"])
        for candle in filtered:
            writer.writerow([candle.timestamp.isoformat(), candle.open, candle.high, candle.low, candle.close, candle.volume])


def load_daily_market_data(symbol: str) -> DailyMarketData | None:
    path = _daily_cache_path(symbol)
    if not path.exists():
        return None
    rows: list[DailyCandle] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            candle = _clean_daily_candle_row(row)
            if candle is not None:
                rows.append(candle)
    if not rows:
        return None
    rows.sort(key=lambda item: item.timestamp)
    deduped: list[DailyCandle] = []
    seen: set[date] = set()
    for candle in rows:
        if candle.timestamp not in seen:
            deduped.append(candle)
            seen.add(candle.timestamp)
    return DailyMarketData(Stock(symbol, symbol), tuple(deduped))


def refresh_daily_market_cache(universe: Sequence[Stock], market: MarketDataProvider) -> tuple[int, int, list[str]]:
    loaded = 0
    failed: list[str] = []
    for stock in universe:
        try:
            data = market.daily(stock, SMC_LOOKBACK_DAYS)
            if not data.candles:
                raise ValueError("no daily candles returned")
            cleaned = tuple(candle for candle in data.candles if candle.timestamp < date.today())
            if not cleaned:
                raise ValueError("daily history contains only the current partial candle")
            save_daily_market_data(stock.symbol, DailyMarketData(stock, cleaned))
            loaded += 1
        except Exception as exc:  # pragma: no cover - provider/network path
            LOG.warning("%s data failed: %s", stock.symbol, exc)
            failed.append(f"{stock.symbol}: {exc}")
    return loaded, len(failed), failed


def run_scan(universe: Sequence[Stock], market: MarketDataProvider, financials: FinancialDataProvider, scores: ThirdPartyScoreProvider, store: WatchlistStore) -> tuple[str, list[WatchlistEntry]]:
    new_entries: list[WatchlistEntry] = []
    final_entries: list[WatchlistEntry] = []
    scan_rows: list[str] = []
    today = date.today()
    previous = store.load()
    expired_entries = [entry for entry in previous.values() if date.fromisoformat(entry.expires_at) < today]

    for stock in universe:
        cached = load_daily_market_data(stock.symbol)
        if cached is None:
            scan_rows.append(f"| {stock.symbol} | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | DATA MISSING | N/A | SKIPPED |")
            continue
        try:
            structure = calculate_smc(cached)
            price = cached.candles[-1].close
            verdict, level_distance, level_name = smc_verdict(price, structure)
            eq = structure.equilibrium
            eq_range = structure.equilibrium_distance
            price_eq = abs(price - eq) / eq * 100.0 if eq else None
            event = structure.last_structure_event or "-"
            event_date = structure.last_structure_event_time.isoformat() if structure.last_structure_event_time else "-"

            def fmt(value: float | None) -> str:
                return f"{value:.2f}" if value is not None else "N/A"

            level_text = fmt(level_distance) if level_distance is not None else "-"
            scan_rows.append(
                f"| {stock.symbol} | {price:.2f} | {structure.swing_trend.upper()} | "
                f"{fmt(structure.strong_high)} | {fmt(structure.weak_high)} | "
                f"{fmt(structure.strong_low)} | {fmt(structure.weak_low)} | "
                f"{fmt(eq)} | {fmt(eq_range)}% | {fmt(price_eq)}% | "
                f"{event} {event_date} | {level_text}% {level_name} | {verdict} |"
            )

            # Watchlist persistence is based only on the current LuxAlgo-level verdict.
            neutral_financial = FinancialScore(0.0, "N/A", "DISABLED", None, None, None)
            candidate = evaluate_signal(cached, structure, neutral_financial, ThirdPartyScores())
            if candidate:
                final_entries.append(candidate)
                entry, status = store.merge(candidate)
                if status in {"NEW", "RENEWED"}:
                    new_entries.append(entry)
        except (OSError, ValueError, RuntimeError, KeyError) as exc:
            LOG.warning("%s skipped: %s", stock.symbol, exc)
            scan_rows.append(f"| {stock.symbol} | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | ERROR: {exc} | N/A | SKIPPED |")

    return format_report(new_entries, list(store.active(today).values()), expired_entries, scan_rows), final_entries

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", default=UNIVERSE_FILE)
    parser.add_argument("--financials", default=FINANCIALS_FILE)
    parser.add_argument("--third-party", default=THIRD_PARTY_FILE)
    parser.add_argument("--state", default=STATE_FILE)
    parser.add_argument("--state-backend", choices=("local", "s3", "auto"), default="auto")
    parser.add_argument("--market-provider", choices=("auto", "kite", "yahoo"), default="auto")
    parser.add_argument("--financial-provider", choices=("auto", "json", "screener"), default="auto")
    parser.add_argument("--notify-telegram", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")
    try:
        if args.market_provider == "auto":
            market_provider = "kite"
            try:
                market = KiteMarketDataProvider()
            except (OSError, ValueError, RuntimeError, KeyError):
                LOG.warning("Kite market data unavailable; falling back to Yahoo Finance")
                market = YahooMarketDataProvider()
                market_provider = "yahoo"
        elif args.market_provider == "kite":
            market = KiteMarketDataProvider()
            market_provider = "kite"
        else:
            market = YahooMarketDataProvider()
            market_provider = "yahoo"

        if args.state_backend == "auto":
            state_backend = "local"
            try:
                store = S3WatchlistStore()
                state_backend = "s3"
            except (OSError, ValueError, RuntimeError, json.JSONDecodeError):
                LOG.warning("S3 state unavailable; falling back to local watchlist state")
                store = WatchlistStore(args.state)
        elif args.state_backend == "s3":
            store = S3WatchlistStore()
        else:
            store = WatchlistStore(args.state)

        if args.financial_provider == "auto":
            # Finalist financials are refreshed from Screener after the LuxAlgo scan.
            # Keep the existing JSON provider option for explicit --financial-provider json.
            financial_provider = "screener"
            try:
                if financial_provider == "screener":
                    financials = ScreenerFinancialDataProvider()
                else:
                    financials = JsonFinancialDataProvider(args.financials)
            except (OSError, ValueError, RuntimeError, json.JSONDecodeError):
                LOG.warning("Primary financial provider failed; falling back to local JSON file")
                financials = JsonFinancialDataProvider(args.financials)
        elif args.financial_provider == "screener":
            financials = ScreenerFinancialDataProvider()
        else:
            financials = JsonFinancialDataProvider(args.financials)

        universe = load_universe(args.universe)
        if len(universe) != 250:
            raise ValueError(f"Universe file {args.universe!r} should contain exactly 250 stocks, found {len(universe)}")
        LOG.info("NIFTY 250 universe loaded: %d stocks from %s", len(universe), args.universe)
        refreshed, failed_count, failed_symbols = refresh_daily_market_cache(universe, market)
        LOG.info("Daily cache refresh finished: refreshed=%d, failed=%d, universe=%d", refreshed, failed_count, len(universe))
        if failed_symbols:
            LOG.warning("Daily cache refresh failures: %s", "; ".join(failed_symbols))

        report, final_entries = run_scan(
            universe,
            market,
            financials,
            JsonThirdPartyScoreProvider(args.third_party),
            store,
        )

        # IMPORTANT: LuxAlgo completes the shortlist first.
        # Only stocks in that final shortlist receive a fresh Screener financial fetch.
        final_financials = financials
        if args.financial_provider != "json":
            final_financials = ScreenerFinancialDataProvider()
        else:
            # Even when JSON is selected for compatibility, the production shortlist
            # is refreshed from Screener so the final list receives latest data.
            final_financials = ScreenerFinancialDataProvider()
        final_financial_report = format_final_financial_report(final_entries, final_financials)
        write_state_check_csv(final_entries, final_financials)
        LOG.info(
            "Final LuxAlgo shortlist complete: %d stock(s). Latest financial data refreshed for shortlist.",
            len(final_entries),
        )

        if args.notify_telegram:
            send_telegram(final_financial_report)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        LOG.error("Scanner failed: %s", exc)
        return 1
    print(report)
    print("\n" + final_financial_report)
    return 0


if __name__ == "__main__":
    sys.exit(main())