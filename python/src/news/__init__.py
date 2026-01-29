"""
News filtering module for trading.

Provides economic calendar integration to block trading
during high-impact news events.
"""

from .currency_impact import CurrencyImpactMapper
from .news_filter import NewsFilter, NewsFilterConfig
from .news_provider import (
    EconomicEvent,
    EventImpact,
    InvestingComProvider,
    NewsProvider,
)

__all__ = [
    "CurrencyImpactMapper",
    "NewsFilter",
    "NewsFilterConfig",
    "NewsProvider",
    "InvestingComProvider",
    "EconomicEvent",
    "EventImpact",
]
