"""
Production-ready prediction service for real-time inference.

Provides LightGBM model loading, real-time predictions with confidence estimation,
caching, and ZeroMQ integration preparation.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import pandas as pd

from ..training.trainer import TrainingResult
from ..utils.config import get_config
from ..utils.logger import get_logger

logger = get_logger("inference.predictor")


# =============================================================================
# Cache Management
# =============================================================================

@dataclass
class CacheEntry:
    """Single cache entry with TTL."""
    value: dict[str, Any]
    timestamp: float
    
    def is_expired(self, ttl_seconds: int) -> bool:
        """Check if entry has expired."""
        return (time.time() - self.timestamp) > ttl_seconds


class TTLCache:
    """LRU cache with TTL support."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        """
        Initialize TTL cache.
        
        Args:
            max_size: Maximum number of entries
            ttl_seconds: Time-to-live for entries in seconds
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: dict[str, CacheEntry] = {}
        self.lock = Lock()
    
    def _hash_features(self, features: pd.DataFrame) -> str:
        """Generate hash of feature values."""
        # Use pandas hash function for consistent hashing
        feature_hash = pd.util.hash_pandas_object(features, index=True).values[0]
        return str(feature_hash)
    
    def get(self, features: pd.DataFrame) -> dict[str, Any] | None:
        """Get cached prediction if available and not expired."""
        key = self._hash_features(features)
        
        with self.lock:
            if key not in self.cache:
                return None
            
            entry = self.cache[key]
            if entry.is_expired(self.ttl_seconds):
                del self.cache[key]
                return None
            
            return entry.value
    
    def set(self, features: pd.DataFrame, value: dict[str, Any]) -> None:
        """Store prediction in cache."""
        key = self._hash_features(features)
        
        with self.lock:
            # Remove oldest entry if cache is full
            if len(self.cache) >= self.max_size:
                oldest_key = min(
                    self.cache.keys(),
                    key=lambda k: self.cache[k].timestamp
                )
                del self.cache[oldest_key]
            
            self.cache[key] = CacheEntry(
                value=value,
                timestamp=time.time()
            )
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
    
    def size(self) -> int:
        """Get current cache size."""
        with self.lock:
            return len(self.cache)


# =============================================================================
# Prediction Service
# =============================================================================

class PredictionService:
    """
    Production-ready prediction service.
    
    Loads trained LightGBM models and performs real-time inference with
    confidence estimation, caching, and ZeroMQ integration preparation.
    """
    
    def __init__(
        self,
        model_path: Path | str | None = None,
        cache_enabled: bool | None = None,
        cache_ttl_seconds: int | None = None,
        warmup_enabled: bool | None = None,
        warmup_n_samples: int | None = None,
    ):
        """
        Initialize prediction service.
        
        Args:
            model_path: Path to trained model directory
            cache_enabled: Enable prediction caching (from config if None)
            cache_ttl_seconds: Cache TTL in seconds (from config if None)
            warmup_enabled: Enable model warmup (from config if None)
            warmup_n_samples: Number of warmup samples (from config if None)
        """
        config = get_config()
        
        # Load configuration
        self.cache_enabled = cache_enabled if cache_enabled is not None else \
            config.model.inference.get("cache", {}).get("enabled", True)
        self.cache_ttl_seconds = cache_ttl_seconds if cache_ttl_seconds is not None else \
            config.model.inference.get("cache", {}).get("ttl_seconds", 300)
        self.warmup_enabled = warmup_enabled if warmup_enabled is not None else \
            config.model.inference.get("warmup", {}).get("enabled", True)
        self.warmup_n_samples = warmup_n_samples if warmup_n_samples is not None else \
            config.model.inference.get("warmup", {}).get("n_samples", 100)
        self.batch_size = config.model.inference.get("batch_size", 32)
        self.direction_threshold = config.model.target.direction_threshold
        
        # Model path
        if model_path is None:
            model_path = config.model.model_dir
        self.model_path = Path(model_path)
        
        # Initialize cache
        self.cache = TTLCache(
            max_size=1000,
            ttl_seconds=self.cache_ttl_seconds
        ) if self.cache_enabled else None
        
        # Load model
        self.training_result: TrainingResult | None = None
        self.model = None
        self.feature_names: list[str] = []
        self._load_model()
        
        # Warmup
        self.warmup_time_ms: float = 0.0
        if self.warmup_enabled and self.model is not None:
            self._warmup()
        
        logger.info(
            f"PredictionService initialized: "
            f"cache={'enabled' if self.cache_enabled else 'disabled'}, "
            f"warmup_time={self.warmup_time_ms:.2f}ms"
        )
    
    def _load_model(self) -> None:
        """Load trained model from disk."""
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model path not found: {self.model_path}")
            
            self.training_result = TrainingResult.load(self.model_path)
            self.model = self.training_result.model
            
            # Extract feature names from metadata
            self.feature_names = list(
                self.training_result.feature_importance["feature"].values
            )
            
            logger.info(
                f"Model loaded from {self.model_path}: "
                f"{len(self.feature_names)} features, "
                f"direction_accuracy={self.training_result.metrics.direction_accuracy:.3f}"
            )
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}") from e
    
    def _warmup(self) -> None:
        """Warm up model with dummy predictions."""
        if self.model is None:
            logger.warning("Cannot warmup: model not loaded")
            return
        
        try:
            # Generate dummy features matching model's expected shape
            dummy_features = pd.DataFrame(
                np.random.randn(self.warmup_n_samples, len(self.feature_names)),
                columns=self.feature_names
            )
            
            # Measure warmup time
            start_time = time.time()
            _ = self.model.predict(dummy_features)
            self.warmup_time_ms = (time.time() - start_time) * 1000
            
            logger.info(
                f"Model warmup complete: "
                f"{self.warmup_n_samples} samples in {self.warmup_time_ms:.2f}ms"
            )
        except Exception as e:
            logger.error(f"Warmup failed: {e}")
            self.warmup_time_ms = 0.0
    
    def _validate_features(self, features: pd.DataFrame) -> None:
        """
        Validate input features.
        
        Args:
            features: Feature DataFrame
            
        Raises:
            ValueError: If features are invalid
        """
        if features.empty:
            raise ValueError("Features DataFrame is empty")
        
        if features.shape[0] != 1:
            raise ValueError(
                f"Expected 1 row of features, got {features.shape[0]}. "
                "Use only the last completed bar."
            )
        
        # Check for missing columns
        missing_cols = set(self.feature_names) - set(features.columns)
        if missing_cols:
            raise ValueError(
                f"Missing features: {missing_cols}. "
                f"Expected: {self.feature_names}"
            )
        
        # Check for NaN values
        if features.isna().any().any():
            nan_cols = features.columns[features.isna().any()].tolist()
            raise ValueError(f"NaN values found in features: {nan_cols}")
        
        # Check for infinite values
        if np.isinf(features.values).any():
            raise ValueError("Infinite values found in features")
    
    def _estimate_confidence(self, prediction: float) -> float:
        """
        Estimate prediction confidence.
        
        For regression models, confidence is based on absolute prediction value.
        Larger expected returns indicate stronger signals.
        
        Args:
            prediction: Model prediction (expected return)
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        # Confidence = min(abs(prediction) / 0.01, 1.0)
        # Rationale: 1% return = 100% confidence, cap at 1.0
        confidence = min(abs(prediction) / 0.01, 1.0)
        return float(confidence)
    
    def _map_direction(self, prediction: float) -> str:
        """
        Map prediction to trading direction.
        
        Args:
            prediction: Model prediction (expected return)
            
        Returns:
            Direction: "buy", "sell", or "neutral"
        """
        if prediction > self.direction_threshold:
            return "buy"
        elif prediction < -self.direction_threshold:
            return "sell"
        else:
            return "neutral"
    
    def predict(
        self,
        features: pd.DataFrame,
        min_confidence: float = 0.0,
        include_quantiles: bool = False,
    ) -> dict[str, Any]:
        """
        Make prediction on features.
        
        Args:
            features: Feature DataFrame (single row)
            min_confidence: Minimum confidence threshold (0.0-1.0)
            include_quantiles: Include quantile predictions
            
        Returns:
            Prediction result dict with keys:
            - direction: "buy" | "sell" | "neutral"
            - expected_return: float
            - confidence: float (0.0-1.0)
            - quantile_5: float (if include_quantiles=True)
            - quantile_50: float (if include_quantiles=True)
            - quantile_95: float (if include_quantiles=True)
            - timestamp: ISO format string
            
        Raises:
            RuntimeError: If model not loaded
            ValueError: If features invalid
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Validate features
        self._validate_features(features)
        
        # Check cache
        if self.cache_enabled:
            cached = self.cache.get(features)
            if cached is not None:
                logger.debug("Cache hit")
                return cached
        
        # Select only required features in correct order
        X = features[self.feature_names].copy()
        
        # Make prediction
        prediction = float(self.model.predict(X)[0])
        
        # Estimate confidence
        confidence = self._estimate_confidence(prediction)
        
        # Map to direction
        direction = self._map_direction(prediction)
        
        # Build result
        result = {
            "direction": direction,
            "expected_return": prediction,
            "confidence": confidence,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
        # Add quantiles if requested
        if include_quantiles:
            # For now, use simple heuristic based on prediction variance
            # In production, this would use quantile regression or ensemble
            std_estimate = abs(prediction) * 0.5  # Assume 50% std of prediction
            result["quantile_5"] = prediction - 2 * std_estimate
            result["quantile_50"] = prediction
            result["quantile_95"] = prediction + 2 * std_estimate
        else:
            result["quantile_5"] = None
            result["quantile_50"] = None
            result["quantile_95"] = None
        
        # Cache result
        if self.cache_enabled:
            self.cache.set(features, result)
        
        # Log prediction
        logger.debug(
            f"Prediction: direction={direction}, "
            f"return={prediction:.6f}, confidence={confidence:.3f}"
        )
        
        return result
    
    def predict_batch(
        self,
        features: pd.DataFrame,
        include_quantiles: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Make predictions on multiple rows.
        
        Args:
            features: Feature DataFrame (multiple rows)
            include_quantiles: Include quantile predictions
            
        Returns:
            List of prediction dicts
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        if features.empty:
            raise ValueError("Features DataFrame is empty")
        
        # Select only required features
        X = features[self.feature_names].copy()
        
        # Make predictions
        predictions = self.model.predict(X)
        
        # Build results
        results = []
        for i, pred in enumerate(predictions):
            confidence = self._estimate_confidence(pred)
            direction = self._map_direction(pred)
            
            result = {
                "direction": direction,
                "expected_return": float(pred),
                "confidence": confidence,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            
            if include_quantiles:
                std_estimate = abs(pred) * 0.5
                result["quantile_5"] = pred - 2 * std_estimate
                result["quantile_50"] = pred
                result["quantile_95"] = pred + 2 * std_estimate
            else:
                result["quantile_5"] = None
                result["quantile_50"] = None
                result["quantile_95"] = None
            
            results.append(result)
        
        logger.debug(f"Batch prediction: {len(results)} samples")
        return results
    
    def clear_cache(self) -> None:
        """Clear prediction cache."""
        if self.cache is not None:
            self.cache.clear()
            logger.info("Cache cleared")
    
    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        if self.cache is None:
            return {"enabled": False}
        
        return {
            "enabled": True,
            "size": self.cache.size(),
            "max_size": self.cache.max_size,
            "ttl_seconds": self.cache.ttl_seconds,
        }
    
    def get_model_info(self) -> dict[str, Any]:
        """Get model information."""
        if self.training_result is None:
            return {}
        
        return {
            "n_features": len(self.feature_names),
            "feature_names": self.feature_names,
            "trained_at": self.training_result.trained_at.isoformat(),
            "metrics": {
                "rmse": self.training_result.metrics.rmse,
                "mae": self.training_result.metrics.mae,
                "r_squared": self.training_result.metrics.r_squared,
                "direction_accuracy": self.training_result.metrics.direction_accuracy,
            },
            "warmup_time_ms": self.warmup_time_ms,
        }
