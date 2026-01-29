"""
Currency pair to news impact mapping.

Maps trading symbols to their constituent currencies and determines
which economic events affect each currency pair.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


@dataclass
class CurrencyPair:
    """Represents a currency pair with base and quote currencies."""
    
    symbol: str
    base: str
    quote: str
    
    @classmethod
    def from_symbol(cls, symbol: str) -> CurrencyPair:
        """
        Parse a currency pair symbol.
        
        Args:
            symbol: Trading symbol (e.g., "USDJPY", "EURUSD")
            
        Returns:
            CurrencyPair instance
            
        Raises:
            ValueError: If symbol format is invalid
        """
        # Remove common suffixes
        clean_symbol = symbol.upper()
        for suffix in (".m", ".i", "_m", "_i", ".pro", "-m"):
            if clean_symbol.lower().endswith(suffix):
                clean_symbol = clean_symbol[:-len(suffix)].upper()
        
        # Standard 6-character FX pairs
        if len(clean_symbol) == 6:
            return cls(
                symbol=symbol,
                base=clean_symbol[:3],
                quote=clean_symbol[3:],
            )
        
        # Handle pairs with separators (e.g., "USD/JPY")
        for sep in ("/", "_", "-"):
            if sep in clean_symbol:
                parts = clean_symbol.split(sep)
                if len(parts) == 2 and len(parts[0]) == 3 and len(parts[1]) == 3:
                    return cls(
                        symbol=symbol,
                        base=parts[0],
                        quote=parts[1],
                    )
        
        raise ValueError(f"Invalid currency pair symbol: {symbol}")


class CurrencyImpactMapper:
    """
    Maps currencies to their country codes and determines
    which currency pairs are affected by news from specific countries.
    """
    
    # Currency code to country/region code mapping
    # Based on ISO 4217 currency codes and ISO 3166-1 alpha-2 country codes
    CURRENCY_TO_COUNTRY: ClassVar[dict[str, list[str]]] = {
        # Major currencies
        "USD": ["US"],       # United States Dollar
        "EUR": ["EU", "DE", "FR", "IT", "ES"],  # Euro (Eurozone)
        "JPY": ["JP"],       # Japanese Yen
        "GBP": ["GB", "UK"], # British Pound
        "AUD": ["AU"],       # Australian Dollar
        "NZD": ["NZ"],       # New Zealand Dollar
        "CAD": ["CA"],       # Canadian Dollar
        "CHF": ["CH"],       # Swiss Franc
        
        # Minor currencies
        "CNY": ["CN"],       # Chinese Yuan
        "CNH": ["CN"],       # Chinese Yuan (offshore)
        "HKD": ["HK"],       # Hong Kong Dollar
        "SGD": ["SG"],       # Singapore Dollar
        "KRW": ["KR"],       # South Korean Won
        "INR": ["IN"],       # Indian Rupee
        "MXN": ["MX"],       # Mexican Peso
        "ZAR": ["ZA"],       # South African Rand
        "BRL": ["BR"],       # Brazilian Real
        "RUB": ["RU"],       # Russian Ruble
        "TRY": ["TR"],       # Turkish Lira
        "SEK": ["SE"],       # Swedish Krona
        "NOK": ["NO"],       # Norwegian Krone
        "DKK": ["DK"],       # Danish Krone
        "PLN": ["PL"],       # Polish Zloty
        "CZK": ["CZ"],       # Czech Koruna
        "HUF": ["HU"],       # Hungarian Forint
        "THB": ["TH"],       # Thai Baht
        "IDR": ["ID"],       # Indonesian Rupiah
        "MYR": ["MY"],       # Malaysian Ringgit
        "PHP": ["PH"],       # Philippine Peso
        "TWD": ["TW"],       # Taiwan Dollar
    }
    
    # Country code to currency mapping (reverse lookup)
    COUNTRY_TO_CURRENCY: ClassVar[dict[str, str]] = {
        "US": "USD",
        "EU": "EUR",
        "DE": "EUR",
        "FR": "EUR",
        "IT": "EUR",
        "ES": "EUR",
        "JP": "JPY",
        "GB": "GBP",
        "UK": "GBP",
        "AU": "AUD",
        "NZ": "NZD",
        "CA": "CAD",
        "CH": "CHF",
        "CN": "CNY",
        "HK": "HKD",
        "SG": "SGD",
        "KR": "KRW",
        "IN": "INR",
        "MX": "MXN",
        "ZA": "ZAR",
        "BR": "BRL",
        "RU": "RUB",
        "TR": "TRY",
        "SE": "SEK",
        "NO": "NOK",
        "DK": "DKK",
        "PL": "PLN",
        "CZ": "CZK",
        "HU": "HUF",
        "TH": "THB",
        "ID": "IDR",
        "MY": "MYR",
        "PH": "PHP",
        "TW": "TWD",
    }
    
    # News event title patterns that affect specific currencies
    # Used for additional keyword matching
    CURRENCY_KEYWORDS: ClassVar[dict[str, list[str]]] = {
        "USD": ["fed", "fomc", "nonfarm", "payroll", "cpi", "ppi", "gdp", "retail sales"],
        "EUR": ["ecb", "lagarde", "eurozone", "german", "french"],
        "JPY": ["boj", "bank of japan", "tankan", "japanese"],
        "GBP": ["boe", "bank of england", "british", "uk "],
        "AUD": ["rba", "reserve bank of australia", "australian"],
        "NZD": ["rbnz", "reserve bank of new zealand", "new zealand"],
        "CAD": ["boc", "bank of canada", "canadian"],
        "CHF": ["snb", "swiss national bank", "swiss"],
    }
    
    def __init__(self, symbols: list[str] | None = None):
        """
        Initialize the mapper.
        
        Args:
            symbols: List of trading symbols to track. If None, accepts all.
        """
        self.tracked_pairs: dict[str, CurrencyPair] = {}
        
        if symbols:
            for symbol in symbols:
                try:
                    pair = CurrencyPair.from_symbol(symbol)
                    self.tracked_pairs[symbol] = pair
                except ValueError:
                    pass  # Skip invalid symbols
    
    def get_affected_currencies(self, country_code: str) -> set[str]:
        """
        Get currencies affected by news from a country.
        
        Args:
            country_code: ISO 3166-1 alpha-2 country code
            
        Returns:
            Set of currency codes affected
        """
        country_upper = country_code.upper()
        
        # Direct mapping
        if country_upper in self.COUNTRY_TO_CURRENCY:
            return {self.COUNTRY_TO_CURRENCY[country_upper]}
        
        return set()
    
    def get_affected_pairs(
        self,
        country_code: str,
        event_title: str | None = None,
    ) -> list[str]:
        """
        Get trading pairs affected by news from a country.
        
        Args:
            country_code: ISO 3166-1 alpha-2 country code
            event_title: Optional event title for keyword matching
            
        Returns:
            List of affected symbol names
        """
        affected_currencies = self.get_affected_currencies(country_code)
        
        # Additional keyword-based detection
        if event_title:
            title_lower = event_title.lower()
            for currency, keywords in self.CURRENCY_KEYWORDS.items():
                if any(kw in title_lower for kw in keywords):
                    affected_currencies.add(currency)
        
        # Find pairs containing affected currencies
        affected_pairs = []
        
        for symbol, pair in self.tracked_pairs.items():
            if pair.base in affected_currencies or pair.quote in affected_currencies:
                affected_pairs.append(symbol)
        
        return affected_pairs
    
    def is_pair_affected(
        self,
        symbol: str,
        country_code: str,
        event_title: str | None = None,
    ) -> bool:
        """
        Check if a specific pair is affected by news from a country.
        
        Args:
            symbol: Trading symbol
            country_code: Country code of the news
            event_title: Optional event title
            
        Returns:
            True if pair is affected
        """
        try:
            pair = self.tracked_pairs.get(symbol) or CurrencyPair.from_symbol(symbol)
        except ValueError:
            return False
        
        affected_currencies = self.get_affected_currencies(country_code)
        
        # Additional keyword-based detection
        if event_title:
            title_lower = event_title.lower()
            for currency, keywords in self.CURRENCY_KEYWORDS.items():
                if any(kw in title_lower for kw in keywords):
                    affected_currencies.add(currency)
        
        return pair.base in affected_currencies or pair.quote in affected_currencies
    
    def add_symbol(self, symbol: str) -> bool:
        """
        Add a trading symbol to track.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            True if added successfully
        """
        if symbol in self.tracked_pairs:
            return True
        
        try:
            pair = CurrencyPair.from_symbol(symbol)
            self.tracked_pairs[symbol] = pair
            return True
        except ValueError:
            return False
    
    def remove_symbol(self, symbol: str) -> bool:
        """
        Remove a trading symbol from tracking.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            True if removed
        """
        if symbol in self.tracked_pairs:
            del self.tracked_pairs[symbol]
            return True
        return False
    
    def get_pair_currencies(self, symbol: str) -> tuple[str, str] | None:
        """
        Get base and quote currencies for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Tuple of (base, quote) or None if invalid
        """
        try:
            pair = self.tracked_pairs.get(symbol) or CurrencyPair.from_symbol(symbol)
            return (pair.base, pair.quote)
        except ValueError:
            return None
