"""
Economic calendar data providers.

Fetches news events from Forex Factory and Investing.com
for use in trading filters.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from ..utils.logger import get_logger

logger = get_logger("news.provider")


# =============================================================================
# Data Models
# =============================================================================

class EventImpact(Enum):
    """Impact level of economic event."""
    
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    HOLIDAY = "holiday"
    
    @classmethod
    def from_string(cls, value: str) -> EventImpact:
        """Parse impact from string."""
        value_lower = value.lower().strip()
        
        if value_lower in ("high", "red", "3", "bull3", "gry"):
            return cls.HIGH
        elif value_lower in ("medium", "orange", "2", "bull2", "yel"):
            return cls.MEDIUM
        elif value_lower in ("low", "yellow", "1", "bull1", "ora"):
            return cls.LOW
        elif value_lower in ("holiday", "gray", "0"):
            return cls.HOLIDAY
        
        return cls.LOW


@dataclass
class EconomicEvent:
    """Represents an economic calendar event."""
    
    id: str
    """Unique identifier for the event."""
    
    title: str
    """Event title/name."""
    
    country: str
    """Country code (ISO 3166-1 alpha-2)."""
    
    currency: str
    """Currency code affected."""
    
    datetime_utc: datetime
    """Event time in UTC."""
    
    impact: EventImpact
    """Impact level."""
    
    actual: str | None = None
    """Actual value (if released)."""
    
    forecast: str | None = None
    """Forecast value."""
    
    previous: str | None = None
    """Previous value."""
    
    source: str = ""
    """Data source identifier."""
    
    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata."""
    
    def __hash__(self) -> int:
        """Hash based on title, country, and time."""
        return hash((self.title, self.country, self.datetime_utc.isoformat()))
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "country": self.country,
            "currency": self.currency,
            "datetime_utc": self.datetime_utc.isoformat(),
            "impact": self.impact.value,
            "actual": self.actual,
            "forecast": self.forecast,
            "previous": self.previous,
            "source": self.source,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EconomicEvent:
        """Create from dictionary."""
        return cls(
            id=data["id"],
            title=data["title"],
            country=data["country"],
            currency=data["currency"],
            datetime_utc=datetime.fromisoformat(data["datetime_utc"]),
            impact=EventImpact(data["impact"]),
            actual=data.get("actual"),
            forecast=data.get("forecast"),
            previous=data.get("previous"),
            source=data.get("source", ""),
            metadata=data.get("metadata", {}),
        )


# =============================================================================
# Base Provider
# =============================================================================

class NewsProvider(ABC):
    """Abstract base class for economic calendar providers."""
    
    def __init__(
        self,
        cache_dir: Path | None = None,
        cache_ttl_hours: int = 1,
        request_timeout: int = 30,
    ):
        """
        Initialize provider.
        
        Args:
            cache_dir: Directory for caching data
            cache_ttl_hours: Cache time-to-live in hours
            request_timeout: HTTP request timeout in seconds
        """
        self.cache_dir = cache_dir or Path("data/cache/news")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_ttl = timedelta(hours=cache_ttl_hours)
        self.request_timeout = request_timeout
        
        # Rate limiting
        self._last_request_time: float = 0
        self._min_request_interval: float = 2.0  # seconds
        
        # Session with headers
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        })
    
    @abstractmethod
    def fetch_events(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[EconomicEvent]:
        """
        Fetch economic events.
        
        Args:
            start_date: Start of date range (defaults to today)
            end_date: End of date range (defaults to end of week)
            
        Returns:
            List of economic events
        """
        pass
    
    @property
    @abstractmethod
    def source_name(self) -> str:
        """Provider source name."""
        pass
    
    def _rate_limit(self) -> None:
        """Apply rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path for a key."""
        key_hash = hashlib.md5(cache_key.encode()).hexdigest()[:12]
        return self.cache_dir / f"{self.source_name}_{key_hash}.json"
    
    def _load_from_cache(self, cache_key: str) -> list[EconomicEvent] | None:
        """Load events from cache if valid."""
        cache_path = self._get_cache_path(cache_key)
        
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            cached_time = datetime.fromisoformat(data["cached_at"])
            if datetime.now(timezone.utc) - cached_time > self.cache_ttl:
                return None
            
            return [EconomicEvent.from_dict(e) for e in data["events"]]
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to load cache: {e}")
            return None
    
    def _save_to_cache(self, cache_key: str, events: list[EconomicEvent]) -> None:
        """Save events to cache."""
        cache_path = self._get_cache_path(cache_key)
        
        try:
            data = {
                "cached_at": datetime.now(timezone.utc).isoformat(),
                "source": self.source_name,
                "events": [e.to_dict() for e in events],
            }
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")


# =============================================================================
# Forex Factory Provider
# =============================================================================

class ForexFactoryProvider(NewsProvider):
    """
    Forex Factory economic calendar provider.
    
    Scrapes data from forexfactory.com calendar.
    """
    
    BASE_URL = "https://www.forexfactory.com"
    CALENDAR_URL = "https://www.forexfactory.com/calendar"
    
    # Country code mapping for Forex Factory currency flags
    CURRENCY_TO_COUNTRY = {
        "USD": "US",
        "EUR": "EU",
        "GBP": "GB",
        "JPY": "JP",
        "AUD": "AU",
        "NZD": "NZ",
        "CAD": "CA",
        "CHF": "CH",
        "CNY": "CN",
    }
    
    @property
    def source_name(self) -> str:
        return "forexfactory"
    
    def fetch_events(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[EconomicEvent]:
        """
        Fetch events from Forex Factory.
        
        Args:
            start_date: Start date (defaults to today)
            end_date: End date (defaults to 7 days from start)
            
        Returns:
            List of economic events
        """
        if start_date is None:
            start_date = datetime.now(timezone.utc).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
        
        if end_date is None:
            end_date = start_date + timedelta(days=7)
        
        # Check cache
        cache_key = f"{start_date.date()}_{end_date.date()}"
        cached = self._load_from_cache(cache_key)
        if cached:
            logger.debug(f"Loaded {len(cached)} events from cache")
            return cached
        
        # Fetch fresh data
        events = []
        
        try:
            self._rate_limit()
            
            # Forex Factory uses week parameter
            # Format: month.day.year
            date_param = start_date.strftime("%b%d.%Y").lower()
            url = f"{self.CALENDAR_URL}?week={date_param}"
            
            logger.info(f"Fetching Forex Factory calendar: {url}")
            
            response = self.session.get(url, timeout=self.request_timeout)
            response.raise_for_status()
            
            events = self._parse_calendar_page(response.text, start_date.year)
            
            # Filter by date range
            events = [
                e for e in events
                if start_date <= e.datetime_utc <= end_date
            ]
            
            logger.info(f"Fetched {len(events)} events from Forex Factory")
            
            # Cache results
            self._save_to_cache(cache_key, events)
            
        except requests.RequestException as e:
            logger.error(f"Failed to fetch Forex Factory calendar: {e}")
        except Exception as e:
            logger.error(f"Error parsing Forex Factory calendar: {e}")
        
        return events
    
    def _parse_calendar_page(self, html: str, year: int) -> list[EconomicEvent]:
        """Parse Forex Factory calendar HTML."""
        soup = BeautifulSoup(html, "html.parser")
        events = []
        
        # Find calendar table
        calendar_table = soup.find("table", class_="calendar__table")
        if not calendar_table:
            logger.warning("Calendar table not found")
            return events
        
        current_date = None
        current_time = None
        
        # Process each row
        for row in calendar_table.find_all("tr", class_="calendar__row"):
            try:
                # Check for date row
                date_cell = row.find("td", class_="calendar__date")
                if date_cell:
                    date_text = date_cell.get_text(strip=True)
                    if date_text:
                        current_date = self._parse_ff_date(date_text, year)
                
                # Check for event row
                if "calendar__row--event" not in row.get("class", []):
                    continue
                
                # Get time
                time_cell = row.find("td", class_="calendar__time")
                if time_cell:
                    time_text = time_cell.get_text(strip=True)
                    if time_text and time_text not in ("", "All Day", "Tentative"):
                        current_time = self._parse_ff_time(time_text)
                
                if current_date is None:
                    continue
                
                # Get currency
                currency_cell = row.find("td", class_="calendar__currency")
                currency = currency_cell.get_text(strip=True) if currency_cell else ""
                
                if not currency:
                    continue
                
                # Get impact
                impact_cell = row.find("td", class_="calendar__impact")
                impact = EventImpact.LOW
                if impact_cell:
                    impact_span = impact_cell.find("span")
                    if impact_span:
                        impact_class = " ".join(impact_span.get("class", []))
                        if "high" in impact_class or "red" in impact_class:
                            impact = EventImpact.HIGH
                        elif "medium" in impact_class or "ora" in impact_class:
                            impact = EventImpact.MEDIUM
                        elif "yel" in impact_class:
                            impact = EventImpact.LOW
                
                # Get event title
                event_cell = row.find("td", class_="calendar__event")
                title = ""
                if event_cell:
                    title_span = event_cell.find("span", class_="calendar__event-title")
                    title = title_span.get_text(strip=True) if title_span else ""
                
                if not title:
                    continue
                
                # Get actual/forecast/previous
                actual_cell = row.find("td", class_="calendar__actual")
                actual = actual_cell.get_text(strip=True) if actual_cell else None
                
                forecast_cell = row.find("td", class_="calendar__forecast")
                forecast = forecast_cell.get_text(strip=True) if forecast_cell else None
                
                previous_cell = row.find("td", class_="calendar__previous")
                previous = previous_cell.get_text(strip=True) if previous_cell else None
                
                # Build datetime
                if current_time:
                    event_dt = current_date.replace(
                        hour=current_time[0],
                        minute=current_time[1],
                    )
                else:
                    event_dt = current_date
                
                # Create event
                event_id = hashlib.md5(
                    f"{title}_{currency}_{event_dt.isoformat()}".encode()
                ).hexdigest()[:16]
                
                country = self.CURRENCY_TO_COUNTRY.get(currency, currency[:2])
                
                event = EconomicEvent(
                    id=event_id,
                    title=title,
                    country=country,
                    currency=currency,
                    datetime_utc=event_dt,
                    impact=impact,
                    actual=actual if actual else None,
                    forecast=forecast if forecast else None,
                    previous=previous if previous else None,
                    source=self.source_name,
                )
                events.append(event)
                
            except Exception as e:
                logger.debug(f"Error parsing calendar row: {e}")
                continue
        
        return events
    
    def _parse_ff_date(self, date_text: str, year: int) -> datetime:
        """Parse Forex Factory date format."""
        # Formats: "Mon Jan 15", "Tue Jan 16"
        try:
            # Remove day name prefix if present
            parts = date_text.split()
            if len(parts) >= 3:
                month_str = parts[-2]
                day_str = parts[-1]
            elif len(parts) == 2:
                month_str = parts[0]
                day_str = parts[1]
            else:
                raise ValueError(f"Unknown date format: {date_text}")
            
            month_map = {
                "jan": 1, "feb": 2, "mar": 3, "apr": 4,
                "may": 5, "jun": 6, "jul": 7, "aug": 8,
                "sep": 9, "oct": 10, "nov": 11, "dec": 12,
            }
            
            month = month_map.get(month_str.lower()[:3], 1)
            day = int(re.sub(r"\D", "", day_str))
            
            return datetime(year, month, day, tzinfo=timezone.utc)
            
        except Exception as e:
            logger.warning(f"Failed to parse date '{date_text}': {e}")
            return datetime(year, 1, 1, tzinfo=timezone.utc)
    
    def _parse_ff_time(self, time_text: str) -> tuple[int, int]:
        """Parse Forex Factory time format."""
        # Formats: "8:30am", "2:00pm", "10:00pm"
        try:
            time_text = time_text.lower().strip()
            
            # Handle 12-hour format
            is_pm = "pm" in time_text
            time_text = time_text.replace("am", "").replace("pm", "")
            
            if ":" in time_text:
                parts = time_text.split(":")
                hour = int(parts[0])
                minute = int(parts[1]) if len(parts) > 1 else 0
            else:
                hour = int(time_text)
                minute = 0
            
            # Convert to 24-hour
            if is_pm and hour < 12:
                hour += 12
            elif not is_pm and hour == 12:
                hour = 0
            
            return (hour, minute)
            
        except Exception as e:
            logger.warning(f"Failed to parse time '{time_text}': {e}")
            return (0, 0)


# =============================================================================
# Investing.com Provider
# =============================================================================

class InvestingComProvider(NewsProvider):
    """
    Investing.com economic calendar provider.
    
    Uses Investing.com's AJAX API for calendar data.
    """
    
    BASE_URL = "https://www.investing.com"
    CALENDAR_API = "https://www.investing.com/economic-calendar/Service/getCalendarFilteredData"
    
    # Country ID mapping for Investing.com
    COUNTRY_IDS = {
        "US": 5,
        "EU": 72,
        "GB": 4,
        "JP": 35,
        "AU": 25,
        "NZ": 43,
        "CA": 6,
        "CH": 12,
        "CN": 37,
        "DE": 17,
        "FR": 22,
    }
    
    COUNTRY_TO_CURRENCY = {
        5: ("US", "USD"),
        72: ("EU", "EUR"),
        4: ("GB", "GBP"),
        35: ("JP", "JPY"),
        25: ("AU", "AUD"),
        43: ("NZ", "NZD"),
        6: ("CA", "CAD"),
        12: ("CH", "CHF"),
        37: ("CN", "CNY"),
        17: ("DE", "EUR"),
        22: ("FR", "EUR"),
    }
    
    @property
    def source_name(self) -> str:
        return "investing"
    
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        
        # Additional headers for Investing.com
        self.session.headers.update({
            "X-Requested-With": "XMLHttpRequest",
            "Referer": "https://www.investing.com/economic-calendar/",
        })
    
    def fetch_events(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[EconomicEvent]:
        """
        Fetch events from Investing.com.
        
        Args:
            start_date: Start date (defaults to today)
            end_date: End date (defaults to 7 days from start)
            
        Returns:
            List of economic events
        """
        if start_date is None:
            start_date = datetime.now(timezone.utc).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
        
        if end_date is None:
            end_date = start_date + timedelta(days=7)
        
        # Check cache
        cache_key = f"{start_date.date()}_{end_date.date()}"
        cached = self._load_from_cache(cache_key)
        if cached:
            logger.debug(f"Loaded {len(cached)} events from cache")
            return cached
        
        events = []
        
        try:
            self._rate_limit()
            
            # Fetch the calendar page first to get cookies
            logger.info("Fetching Investing.com calendar page")
            page_response = self.session.get(
                f"{self.BASE_URL}/economic-calendar/",
                timeout=self.request_timeout,
            )
            page_response.raise_for_status()
            
            # Parse the page for events
            events = self._parse_calendar_page(page_response.text)
            
            # Filter by date range
            events = [
                e for e in events
                if start_date <= e.datetime_utc <= end_date
            ]
            
            logger.info(f"Fetched {len(events)} events from Investing.com")
            
            # Cache results
            self._save_to_cache(cache_key, events)
            
        except requests.RequestException as e:
            logger.error(f"Failed to fetch Investing.com calendar: {e}")
        except Exception as e:
            logger.error(f"Error parsing Investing.com calendar: {e}")
        
        return events
    
    def _parse_calendar_page(self, html: str) -> list[EconomicEvent]:
        """Parse Investing.com calendar HTML."""
        soup = BeautifulSoup(html, "html.parser")
        events = []
        
        # Find calendar table
        calendar_table = soup.find("table", id="economicCalendarData")
        if not calendar_table:
            logger.warning("Calendar table not found")
            return events
        
        current_date = None
        
        for row in calendar_table.find_all("tr"):
            try:
                # Check for date header row
                date_header = row.find("td", class_="theDay")
                if date_header:
                    date_text = date_header.get_text(strip=True)
                    current_date = self._parse_inv_date(date_text)
                    continue
                
                # Skip non-event rows
                if not row.get("event_attr_id"):
                    continue
                
                # Get time
                time_cell = row.find("td", class_="time")
                event_time = None
                if time_cell:
                    time_text = time_cell.get_text(strip=True)
                    if time_text and time_text not in ("", "All Day", "Tentative"):
                        event_time = self._parse_inv_time(time_text)
                
                # Get country flag
                flag_cell = row.find("td", class_="flagCur")
                country = "US"
                currency = "USD"
                if flag_cell:
                    flag_span = flag_cell.find("span", class_="cemark")
                    if flag_span:
                        # Extract country from flag class
                        flag_class = " ".join(flag_span.get("class", []))
                        for code in self.COUNTRY_IDS.keys():
                            if code.lower() in flag_class.lower():
                                country = code
                                currency = self.COUNTRY_TO_CURRENCY.get(
                                    self.COUNTRY_IDS[code], (code, code)
                                )[1]
                                break
                
                # Get impact
                impact = EventImpact.LOW
                impact_cell = row.find("td", class_="sentiment")
                if impact_cell:
                    bulls = impact_cell.find_all("i", class_="grayFullBullishIcon")
                    bull_count = len(bulls) if bulls else 0
                    if bull_count >= 3:
                        impact = EventImpact.HIGH
                    elif bull_count == 2:
                        impact = EventImpact.MEDIUM
                
                # Get event title
                event_cell = row.find("td", class_="event")
                title = ""
                if event_cell:
                    title_link = event_cell.find("a")
                    title = title_link.get_text(strip=True) if title_link else ""
                
                if not title:
                    continue
                
                # Get actual/forecast/previous
                actual_cell = row.find("td", class_="act")
                actual = actual_cell.get_text(strip=True) if actual_cell else None
                
                forecast_cell = row.find("td", class_="fore")
                forecast = forecast_cell.get_text(strip=True) if forecast_cell else None
                
                previous_cell = row.find("td", class_="prev")
                previous = previous_cell.get_text(strip=True) if previous_cell else None
                
                # Build datetime
                if current_date:
                    if event_time:
                        event_dt = current_date.replace(
                            hour=event_time[0],
                            minute=event_time[1],
                        )
                    else:
                        event_dt = current_date
                else:
                    event_dt = datetime.now(timezone.utc)
                
                # Create event
                event_id = hashlib.md5(
                    f"{title}_{country}_{event_dt.isoformat()}".encode()
                ).hexdigest()[:16]
                
                event = EconomicEvent(
                    id=event_id,
                    title=title,
                    country=country,
                    currency=currency,
                    datetime_utc=event_dt,
                    impact=impact,
                    actual=actual if actual else None,
                    forecast=forecast if forecast else None,
                    previous=previous if previous else None,
                    source=self.source_name,
                )
                events.append(event)
                
            except Exception as e:
                logger.debug(f"Error parsing calendar row: {e}")
                continue
        
        return events
    
    def _parse_inv_date(self, date_text: str) -> datetime:
        """Parse Investing.com date format."""
        # Formats: "Wednesday, January 15, 2025"
        try:
            # Try common formats
            for fmt in [
                "%A, %B %d, %Y",
                "%B %d, %Y",
                "%d %B %Y",
            ]:
                try:
                    dt = datetime.strptime(date_text, fmt)
                    return dt.replace(tzinfo=timezone.utc)
                except ValueError:
                    continue
            
            # Fallback: extract date parts
            parts = re.findall(r"(\w+)\s+(\d+),?\s+(\d{4})", date_text)
            if parts:
                month_str, day_str, year_str = parts[0]
                month_map = {
                    "january": 1, "february": 2, "march": 3, "april": 4,
                    "may": 5, "june": 6, "july": 7, "august": 8,
                    "september": 9, "october": 10, "november": 11, "december": 12,
                }
                month = month_map.get(month_str.lower(), 1)
                return datetime(int(year_str), month, int(day_str), tzinfo=timezone.utc)
            
            raise ValueError(f"Unknown date format: {date_text}")
            
        except Exception as e:
            logger.warning(f"Failed to parse date '{date_text}': {e}")
            return datetime.now(timezone.utc)
    
    def _parse_inv_time(self, time_text: str) -> tuple[int, int]:
        """Parse Investing.com time format."""
        # Formats: "08:30", "14:00"
        try:
            time_text = time_text.strip()
            
            if ":" in time_text:
                parts = time_text.split(":")
                hour = int(parts[0])
                minute = int(parts[1]) if len(parts) > 1 else 0
                return (hour, minute)
            
            raise ValueError(f"Unknown time format: {time_text}")
            
        except Exception as e:
            logger.warning(f"Failed to parse time '{time_text}': {e}")
            return (0, 0)


# =============================================================================
# Aggregated Provider
# =============================================================================

class AggregatedNewsProvider:
    """
    Aggregates events from multiple providers.
    
    Merges and deduplicates events from Forex Factory and Investing.com.
    """
    
    def __init__(
        self,
        providers: list[NewsProvider] | None = None,
        cache_dir: Path | None = None,
    ):
        """
        Initialize aggregated provider.
        
        Args:
            providers: List of providers (defaults to FF + Investing)
            cache_dir: Cache directory
        """
        if providers is None:
            cache_dir = cache_dir or Path("data/cache/news")
            providers = [
                ForexFactoryProvider(cache_dir=cache_dir),
                InvestingComProvider(cache_dir=cache_dir),
            ]
        
        self.providers = providers
    
    def fetch_events(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        min_impact: EventImpact = EventImpact.LOW,
    ) -> list[EconomicEvent]:
        """
        Fetch and merge events from all providers.
        
        Args:
            start_date: Start date
            end_date: End date
            min_impact: Minimum impact level to include
            
        Returns:
            Merged and deduplicated list of events
        """
        all_events: dict[str, EconomicEvent] = {}
        
        # Impact priority for deduplication (prefer higher impact)
        impact_priority = {
            EventImpact.HIGH: 3,
            EventImpact.MEDIUM: 2,
            EventImpact.LOW: 1,
            EventImpact.HOLIDAY: 0,
        }
        
        min_priority = impact_priority[min_impact]
        
        for provider in self.providers:
            try:
                events = provider.fetch_events(start_date, end_date)
                
                for event in events:
                    # Skip low impact if filtered
                    if impact_priority[event.impact] < min_priority:
                        continue
                    
                    # Create dedup key
                    key = self._make_dedup_key(event)
                    
                    # Keep higher impact version
                    if key in all_events:
                        existing = all_events[key]
                        if impact_priority[event.impact] > impact_priority[existing.impact]:
                            all_events[key] = event
                    else:
                        all_events[key] = event
                        
            except Exception as e:
                logger.error(f"Failed to fetch from {provider.source_name}: {e}")
        
        # Sort by datetime
        events = sorted(all_events.values(), key=lambda e: e.datetime_utc)
        
        logger.info(f"Aggregated {len(events)} events from {len(self.providers)} providers")
        
        return events
    
    def _make_dedup_key(self, event: EconomicEvent) -> str:
        """Create deduplication key for an event."""
        # Normalize title for comparison
        title_normalized = re.sub(r"[^a-z0-9]", "", event.title.lower())
        
        # Round time to nearest 15 minutes for fuzzy matching
        dt_rounded = event.datetime_utc.replace(
            minute=(event.datetime_utc.minute // 15) * 15,
            second=0,
            microsecond=0,
        )
        
        return f"{event.currency}_{title_normalized[:30]}_{dt_rounded.date()}"
