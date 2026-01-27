"""
Inference module for real-time predictions.

Provides production-ready prediction service with model loading,
confidence estimation, caching, and ZeroMQ integration preparation.
"""

from .predictor import PredictionService, TTLCache

__all__ = [
    "PredictionService",
    "TTLCache",
]
