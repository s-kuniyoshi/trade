"""
Tests for news filter module.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from python.src.news.currency_impact import CurrencyImpactMapper, CurrencyPair
from python.src.news.news_filter import (
    BlackoutAction,
    BlackoutStatus,
    NewsFilter,
    NewsFilterConfig,
)
from python.src.news.news_provider import (
    AggregatedNewsProvider,
    EconomicEvent,
    EventImpact,
    ForexFactoryProvider,
)


# =============================================================================
# Currency Impact Tests
# =============================================================================

class TestCurrencyPair:
    """Tests for CurrencyPair parsing."""
    
    def test_standard_pair(self):
        """Test standard 6-character pair."""
        pair = CurrencyPair.from_symbol("USDJPY")
        assert pair.base == "USD"
        assert pair.quote == "JPY"
        assert pair.symbol == "USDJPY"
    
    def test_lowercase_pair(self):
        """Test lowercase symbol."""
        pair = CurrencyPair.from_symbol("eurusd")
        assert pair.base == "EUR"
        assert pair.quote == "USD"
    
    def test_pair_with_suffix(self):
        """Test symbol with broker suffix."""
        pair = CurrencyPair.from_symbol("USDJPY.m")
        assert pair.base == "USD"
        assert pair.quote == "JPY"
    
    def test_pair_with_separator(self):
        """Test symbol with separator."""
        pair = CurrencyPair.from_symbol("USD/JPY")
        assert pair.base == "USD"
        assert pair.quote == "JPY"
    
    def test_invalid_pair(self):
        """Test invalid symbol raises error."""
        with pytest.raises(ValueError):
            CurrencyPair.from_symbol("INVALID")


class TestCurrencyImpactMapper:
    """Tests for currency impact mapping."""
    
    def test_affected_currencies_usd(self):
        """Test USD events affect USD currency."""
        mapper = CurrencyImpactMapper()
        currencies = mapper.get_affected_currencies("US")
        assert "USD" in currencies
    
    def test_affected_currencies_eu(self):
        """Test EU events affect EUR currency."""
        mapper = CurrencyImpactMapper()
        currencies = mapper.get_affected_currencies("EU")
        assert "EUR" in currencies
    
    def test_affected_pairs(self):
        """Test affected pairs for USD news."""
        mapper = CurrencyImpactMapper(["USDJPY", "EURUSD", "GBPJPY"])
        
        pairs = mapper.get_affected_pairs("US")
        assert "USDJPY" in pairs
        assert "EURUSD" in pairs
        assert "GBPJPY" not in pairs  # No USD in pair
    
    def test_pair_affected_by_country(self):
        """Test specific pair affected check."""
        mapper = CurrencyImpactMapper()
        
        assert mapper.is_pair_affected("USDJPY", "US")
        assert mapper.is_pair_affected("USDJPY", "JP")
        assert not mapper.is_pair_affected("EURUSD", "JP")
    
    def test_keyword_detection(self):
        """Test keyword-based currency detection."""
        mapper = CurrencyImpactMapper(["USDJPY"])
        
        # Fed-related news should affect USD
        pairs = mapper.get_affected_pairs("US", "FOMC Meeting Minutes")
        assert "USDJPY" in pairs
    
    def test_add_remove_symbol(self):
        """Test adding and removing symbols."""
        mapper = CurrencyImpactMapper()
        
        assert mapper.add_symbol("USDJPY")
        assert "USDJPY" in mapper.tracked_pairs
        
        assert mapper.remove_symbol("USDJPY")
        assert "USDJPY" not in mapper.tracked_pairs


# =============================================================================
# News Provider Tests
# =============================================================================

class TestEconomicEvent:
    """Tests for EconomicEvent data class."""
    
    def test_event_creation(self):
        """Test creating an event."""
        event = EconomicEvent(
            id="test123",
            title="Non-Farm Payrolls",
            country="US",
            currency="USD",
            datetime_utc=datetime(2025, 1, 29, 13, 30, tzinfo=timezone.utc),
            impact=EventImpact.HIGH,
        )
        
        assert event.title == "Non-Farm Payrolls"
        assert event.impact == EventImpact.HIGH
    
    def test_event_to_dict(self):
        """Test event serialization."""
        event = EconomicEvent(
            id="test123",
            title="CPI",
            country="US",
            currency="USD",
            datetime_utc=datetime(2025, 1, 29, 13, 30, tzinfo=timezone.utc),
            impact=EventImpact.HIGH,
            forecast="3.2%",
            previous="3.1%",
        )
        
        data = event.to_dict()
        assert data["title"] == "CPI"
        assert data["impact"] == "high"
        assert data["forecast"] == "3.2%"
    
    def test_event_from_dict(self):
        """Test event deserialization."""
        data = {
            "id": "test123",
            "title": "GDP",
            "country": "US",
            "currency": "USD",
            "datetime_utc": "2025-01-29T13:30:00+00:00",
            "impact": "medium",
        }
        
        event = EconomicEvent.from_dict(data)
        assert event.title == "GDP"
        assert event.impact == EventImpact.MEDIUM


class TestEventImpact:
    """Tests for EventImpact enum."""
    
    def test_from_string_high(self):
        """Test parsing high impact."""
        assert EventImpact.from_string("high") == EventImpact.HIGH
        assert EventImpact.from_string("red") == EventImpact.HIGH
        assert EventImpact.from_string("3") == EventImpact.HIGH
    
    def test_from_string_medium(self):
        """Test parsing medium impact."""
        assert EventImpact.from_string("medium") == EventImpact.MEDIUM
        assert EventImpact.from_string("orange") == EventImpact.MEDIUM
    
    def test_from_string_low(self):
        """Test parsing low impact."""
        assert EventImpact.from_string("low") == EventImpact.LOW
        assert EventImpact.from_string("yellow") == EventImpact.LOW


# =============================================================================
# News Filter Tests
# =============================================================================

class TestNewsFilterConfig:
    """Tests for NewsFilterConfig."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = NewsFilterConfig()
        
        assert config.enabled is True
        assert config.blackout_before_minutes == 30
        assert config.blackout_after_minutes == 15
        assert config.min_impact == EventImpact.HIGH
        assert config.default_action == BlackoutAction.BLOCK_ENTRY
    
    def test_symbol_overrides(self):
        """Test symbol-specific overrides."""
        config = NewsFilterConfig(
            symbol_overrides={
                "USDJPY": {
                    "min_impact": "medium",
                    "blackout_before_minutes": 45,
                }
            }
        )
        
        usdjpy_config = config.get_symbol_config("USDJPY")
        assert usdjpy_config["min_impact"] == EventImpact.MEDIUM
        assert usdjpy_config["blackout_before_minutes"] == 45
        
        # Non-overridden symbol uses defaults
        eurusd_config = config.get_symbol_config("EURUSD")
        assert eurusd_config["min_impact"] == EventImpact.HIGH
        assert eurusd_config["blackout_before_minutes"] == 30


class TestBlackoutStatus:
    """Tests for BlackoutStatus."""
    
    def test_not_blocked(self):
        """Test status when not blocked."""
        status = BlackoutStatus(
            symbol="USDJPY",
            is_blocked=False,
            action=None,
        )
        
        assert not status.is_blocked
        data = status.to_dict()
        assert data["is_blocked"] is False
    
    def test_blocked_with_events(self):
        """Test status when blocked."""
        event = EconomicEvent(
            id="nfp",
            title="Non-Farm Payrolls",
            country="US",
            currency="USD",
            datetime_utc=datetime.now(timezone.utc),
            impact=EventImpact.HIGH,
        )
        
        status = BlackoutStatus(
            symbol="USDJPY",
            is_blocked=True,
            action=BlackoutAction.BLOCK_ENTRY,
            blocking_events=[event],
            blackout_ends=datetime.now(timezone.utc) + timedelta(minutes=15),
        )
        
        assert status.is_blocked
        data = status.to_dict()
        assert data["is_blocked"] is True
        assert len(data["blocking_events"]) == 1


class TestNewsFilter:
    """Tests for NewsFilter."""
    
    @pytest.fixture
    def mock_events(self):
        """Create mock events."""
        now = datetime.now(timezone.utc)
        return [
            EconomicEvent(
                id="nfp",
                title="Non-Farm Payrolls",
                country="US",
                currency="USD",
                datetime_utc=now + timedelta(minutes=10),
                impact=EventImpact.HIGH,
            ),
            EconomicEvent(
                id="cpi",
                title="CPI",
                country="EU",
                currency="EUR",
                datetime_utc=now + timedelta(hours=2),
                impact=EventImpact.HIGH,
            ),
            EconomicEvent(
                id="gdp",
                title="GDP",
                country="JP",
                currency="JPY",
                datetime_utc=now + timedelta(days=1),
                impact=EventImpact.MEDIUM,
            ),
        ]
    
    def test_filter_disabled(self):
        """Test filter when disabled."""
        config = NewsFilterConfig(enabled=False)
        news_filter = NewsFilter(config=config)
        
        status = news_filter.check_blackout("USDJPY")
        assert not status.is_blocked
    
    def test_blackout_detection(self, mock_events):
        """Test blackout detection with mock events."""
        config = NewsFilterConfig(
            enabled=True,
            blackout_before_minutes=30,
            blackout_after_minutes=15,
        )
        
        with patch.object(NewsFilter, '_refresh_events') as mock_refresh:
            news_filter = NewsFilter(config=config, symbols=["USDJPY", "EURUSD"])
            
            # Inject mock events
            news_filter._events = mock_events
            
            # NFP is in 10 minutes, blackout is 30 minutes before
            # So USDJPY should be blocked
            status = news_filter.check_blackout("USDJPY")
            assert status.is_blocked
            assert len(status.blocking_events) == 1
            assert status.blocking_events[0].title == "Non-Farm Payrolls"
    
    def test_no_blackout(self, mock_events):
        """Test no blackout when event is far away."""
        config = NewsFilterConfig(
            enabled=True,
            blackout_before_minutes=30,
            blackout_after_minutes=15,
        )
        
        with patch.object(NewsFilter, '_refresh_events') as mock_refresh:
            news_filter = NewsFilter(config=config, symbols=["GBPJPY"])
            
            # Inject mock events
            news_filter._events = mock_events
            
            # No GBP events, so GBPJPY should not be blocked
            status = news_filter.check_blackout("GBPJPY")
            assert not status.is_blocked
    
    def test_upcoming_events(self, mock_events):
        """Test getting upcoming events."""
        config = NewsFilterConfig(enabled=True)
        
        with patch.object(NewsFilter, '_refresh_events') as mock_refresh:
            news_filter = NewsFilter(config=config, symbols=["USDJPY"])
            news_filter._events = mock_events
            
            events = news_filter.get_upcoming_events(hours_ahead=48)
            assert len(events) >= 1
    
    def test_status_summary(self, mock_events):
        """Test filter status summary."""
        config = NewsFilterConfig(enabled=True)
        
        with patch.object(NewsFilter, '_refresh_events') as mock_refresh:
            news_filter = NewsFilter(config=config)
            news_filter._events = mock_events
            
            status = news_filter.get_status()
            assert status["enabled"] is True
            assert status["event_count"] == 3


# =============================================================================
# Integration Tests
# =============================================================================

class TestNewsFilterIntegration:
    """Integration tests for news filter with real-ish scenarios."""
    
    def test_multi_symbol_blackout(self):
        """Test blackout affecting multiple symbols."""
        now = datetime.now(timezone.utc)
        
        # USD event should block both USDJPY and EURUSD
        events = [
            EconomicEvent(
                id="fomc",
                title="FOMC Rate Decision",
                country="US",
                currency="USD",
                datetime_utc=now + timedelta(minutes=5),
                impact=EventImpact.HIGH,
            ),
        ]
        
        config = NewsFilterConfig(enabled=True)
        
        with patch.object(NewsFilter, '_refresh_events'):
            news_filter = NewsFilter(
                config=config,
                symbols=["USDJPY", "EURUSD", "GBPJPY"],
            )
            news_filter._events = events
            
            # USD pairs should be blocked
            assert news_filter.check_blackout("USDJPY").is_blocked
            assert news_filter.check_blackout("EURUSD").is_blocked
            
            # Non-USD pair should not be blocked
            assert not news_filter.check_blackout("GBPJPY").is_blocked
    
    def test_blackout_timing(self):
        """Test blackout window timing."""
        now = datetime.now(timezone.utc)
        
        config = NewsFilterConfig(
            enabled=True,
            blackout_before_minutes=30,
            blackout_after_minutes=15,
        )
        
        # Event exactly 30 minutes from now - should be blocked (edge case)
        event_at_30 = EconomicEvent(
            id="test",
            title="Test Event",
            country="US",
            currency="USD",
            datetime_utc=now + timedelta(minutes=30),
            impact=EventImpact.HIGH,
        )
        
        with patch.object(NewsFilter, '_refresh_events'):
            news_filter = NewsFilter(config=config, symbols=["USDJPY"])
            news_filter._events = [event_at_30]
            
            # Should be blocked (within blackout window)
            status = news_filter.check_blackout("USDJPY")
            assert status.is_blocked
            
            # Blackout should end 15 minutes after event
            expected_end = event_at_30.datetime_utc + timedelta(minutes=15)
            assert status.blackout_ends == expected_end


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
