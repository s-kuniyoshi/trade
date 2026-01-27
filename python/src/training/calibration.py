"""
Confidence calibration system for improving prediction reliability.

Calibrates raw model outputs to true probabilities using isotonic regression
or Platt scaling, with support for market regime-based adjustments.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..utils.config import get_config
from ..utils.logger import get_logger

logger = get_logger("training.calibration")


# =============================================================================
# Confidence Calibrator
# =============================================================================

class ConfidenceCalibrator:
    """
    Calibrates raw model predictions to true probabilities.
    
    Improves prediction reliability by mapping raw confidence scores to
    calibrated probabilities using isotonic regression or Platt scaling.
    Supports market regime-based confidence adjustments.
    
    Attributes:
        method: Calibration method ("isotonic" or "platt")
        calibrator: Fitted calibration model
        is_fitted: Whether calibrator has been fitted
        min_samples: Minimum samples required for calibration
    
    Example:
        >>> calibrator = ConfidenceCalibrator(method="isotonic")
        >>> calibrator.fit(y_true, y_pred, y_proba)
        >>> calibrated = calibrator.transform(y_pred, y_proba)
        >>> calibrator.save(Path("models/calibrator"))
    """
    
    def __init__(
        self,
        method: str = "isotonic",
        min_samples: int = 50,
    ):
        """
        Initialize confidence calibrator.
        
        Args:
            method: Calibration method ("isotonic" or "platt")
            min_samples: Minimum samples required for calibration
            
        Raises:
            ValueError: If method is not supported
        """
        if method not in ("isotonic", "platt"):
            raise ValueError(f"Unsupported calibration method: {method}")
        
        self.method = method
        self.min_samples = min_samples
        self.calibrator: Any = None
        self.is_fitted = False
        
        logger.info(f"ConfidenceCalibrator initialized: method={method}")
    
    def _prepare_calibration_data(
        self,
        y_true: pd.Series | np.ndarray,
        y_pred: pd.Series | np.ndarray,
    ) -> np.ndarray[Any, Any]:
        """
        Prepare binary calibration targets from predictions.
        
        Converts regression predictions to binary outcomes:
        - 1 if direction is correct (sign(y_pred) == sign(y_true))
        - 0 if direction is incorrect
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            
        Returns:
            Binary array (1 = correct direction, 0 = incorrect)
        """
        # Convert to numpy arrays
        y_true_arr: np.ndarray[Any, Any] = np.asarray(y_true).flatten()
        y_pred_arr: np.ndarray[Any, Any] = np.asarray(y_pred).flatten()
        
        # Check for zero predictions (neutral signals)
        # Treat as incorrect direction
        pred_sign: np.ndarray[Any, Any] = np.sign(y_pred_arr)
        true_sign: np.ndarray[Any, Any] = np.sign(y_true_arr)
        
        # Direction correct: signs match and both non-zero
        y_binary: np.ndarray[Any, Any] = (pred_sign == true_sign) & (pred_sign != 0)
        
        return y_binary.astype(int)
    
    def fit(
        self,
        y_true: pd.Series | np.ndarray,
        y_pred: pd.Series | np.ndarray,
        y_proba: pd.Series | np.ndarray,
    ) -> None:
        """
        Fit calibrator on validation data.
        
        Args:
            y_true: Actual values (continuous)
            y_pred: Predicted values (continuous)
            y_proba: Raw confidence scores (0.0-1.0)
            
        Raises:
            ValueError: If insufficient data for calibration
        """
        # Import sklearn modules
        from sklearn.isotonic import IsotonicRegression as IR  # type: ignore
        from sklearn.linear_model import LogisticRegression as LR  # type: ignore
        
        # Convert to numpy arrays
        y_true_arr: np.ndarray[Any, Any] = np.asarray(y_true).flatten()
        y_pred_arr: np.ndarray[Any, Any] = np.asarray(y_pred).flatten()
        y_proba_arr: np.ndarray[Any, Any] = np.asarray(y_proba).flatten()
        
        # Validate inputs
        if len(y_true_arr) < self.min_samples:
            logger.warning(
                f"Insufficient data for calibration: {len(y_true_arr)} < {self.min_samples}. "
                "Skipping calibration."
            )
            return
        
        if len(y_true_arr) != len(y_pred_arr) or len(y_true_arr) != len(y_proba_arr):
            raise ValueError(
                f"Input arrays have different lengths: "
                f"y_true={len(y_true_arr)}, y_pred={len(y_pred_arr)}, y_proba={len(y_proba_arr)}"
            )
        
        # Prepare binary targets
        y_binary = self._prepare_calibration_data(y_true_arr, y_pred_arr)
        
        # Fit calibrator
        try:
            if self.method == "isotonic":
                self.calibrator = IR(out_of_bounds="clip")
                self.calibrator.fit(y_proba_arr, y_binary)
            elif self.method == "platt":
                # Platt scaling: logistic regression on scores
                platt_model = LR(random_state=42, max_iter=1000)
                platt_model.fit(y_proba_arr.reshape(-1, 1), y_binary)
                self.calibrator = platt_model
            
            self.is_fitted = True
            
            # Log calibration statistics
            n_correct = np.sum(y_binary)
            accuracy = n_correct / len(y_binary)
            logger.info(
                f"Calibrator fitted: method={self.method}, "
                f"samples={len(y_binary)}, direction_accuracy={accuracy:.3f}"
            )
        except Exception as e:
            logger.error(f"Failed to fit calibrator: {e}")
            raise RuntimeError(f"Calibration fitting failed: {e}") from e
    
    def transform(
        self,
        y_pred: pd.Series | np.ndarray,
        y_proba: pd.Series | np.ndarray,
        is_high_vol: bool = False,
        is_counter_trend: bool = False,
    ) -> np.ndarray[Any, Any]:
        """
        Calibrate raw predictions to probabilities.
        
        Applies fitted calibrator to map raw confidence scores to calibrated
        probabilities, with optional market regime adjustments.
        
        Args:
            y_pred: Predicted values (for direction)
            y_proba: Raw confidence scores (0.0-1.0)
            is_high_vol: Whether in high volatility regime
            is_counter_trend: Whether signal is counter-trend
            
        Returns:
            Calibrated confidence scores (0.0-1.0)
            
        Raises:
            RuntimeError: If calibrator not fitted
        """
        if not self.is_fitted or self.calibrator is None:
            logger.warning("Calibrator not fitted. Returning raw probabilities.")
            return np.asarray(y_proba).flatten()
        
        # Convert to numpy array
        y_proba_arr: np.ndarray[Any, Any] = np.asarray(y_proba).flatten()
        
        # Apply calibrator
        try:
            if self.method == "isotonic":
                calibrated = self.calibrator.transform(y_proba_arr)
            elif self.method == "platt":
                # Get probability of positive class
                calibrated = self.calibrator.predict_proba(
                    y_proba_arr.reshape(-1, 1)
                )[:, 1]
            else:
                calibrated = y_proba_arr
            
            # Ensure values are in [0, 1]
            calibrated = np.clip(calibrated, 0.0, 1.0)
            
            # Apply regime adjustments
            calibrated = self._apply_adjustments(
                calibrated,
                is_high_vol=is_high_vol,
                is_counter_trend=is_counter_trend,
            )
            
            return calibrated
        except Exception as e:
            logger.error(f"Failed to transform predictions: {e}")
            return np.clip(y_proba_arr, 0.0, 1.0)
    
    def _apply_adjustments(
        self,
        confidence: np.ndarray[Any, Any],
        is_high_vol: bool = False,
        is_counter_trend: bool = False,
    ) -> np.ndarray[Any, Any]:
        """
        Apply market regime-based confidence adjustments.
        
        Reduces confidence during high volatility or for counter-trend signals
        to be more conservative.
        
        Args:
            confidence: Calibrated confidence scores
            is_high_vol: Whether in high volatility regime
            is_counter_trend: Whether signal is counter-trend
            
        Returns:
            Adjusted confidence scores
        """
        config = get_config()
        calibration_config = config.model.calibration
        
        # Get adjustment parameters
        adjustments = calibration_config.adjustment or {}
        high_vol_penalty = adjustments.get("high_vol_penalty", 0.1)
        counter_trend_penalty = adjustments.get("counter_trend_penalty", 0.05)
        min_confidence = calibration_config.min_confidence
        
        # Apply high volatility penalty
        if is_high_vol:
            confidence = confidence * (1.0 - high_vol_penalty)
        
        # Apply counter-trend penalty
        if is_counter_trend:
            confidence = confidence * (1.0 - counter_trend_penalty)
        
        # Ensure minimum confidence threshold
        confidence = np.maximum(confidence, min_confidence)
        
        # Clip to [0, 1]
        confidence = np.clip(confidence, 0.0, 1.0)
        
        return confidence
    
    def save(self, path: Path | str) -> None:
        """
        Save calibrator to disk.
        
        Args:
            path: Directory path to save calibrator
            
        Raises:
            RuntimeError: If calibrator not fitted
        """
        if not self.is_fitted or self.calibrator is None:
            logger.warning("Calibrator not fitted. Skipping save.")
            return
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        try:
            calibrator_path = path / "calibrator.pkl"
            with open(calibrator_path, "wb") as f:
                pickle.dump(self.calibrator, f)
            
            # Save metadata
            metadata = {
                "method": self.method,
                "is_fitted": self.is_fitted,
            }
            
            import json
            with open(path / "calibrator_metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Calibrator saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save calibrator: {e}")
            raise RuntimeError(f"Calibrator save failed: {e}") from e
    
    @classmethod
    def load(cls, path: Path | str) -> ConfidenceCalibrator:
        """
        Load calibrator from disk.
        
        Args:
            path: Directory path containing calibrator files
            
        Returns:
            Loaded ConfidenceCalibrator instance
            
        Raises:
            FileNotFoundError: If calibrator files not found
            RuntimeError: If loading fails
        """
        path = Path(path)
        
        try:
            # Load metadata
            import json
            metadata_path = path / "calibrator_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                method = metadata.get("method", "isotonic")
            else:
                method = "isotonic"
            
            # Create instance
            calibrator_instance = cls(method=method)
            
            # Load calibrator
            calibrator_path = path / "calibrator.pkl"
            if not calibrator_path.exists():
                raise FileNotFoundError(f"Calibrator file not found: {calibrator_path}")
            
            with open(calibrator_path, "rb") as f:
                calibrator_instance.calibrator = pickle.load(f)
            
            calibrator_instance.is_fitted = True
            
            logger.info(f"Calibrator loaded from {path}: method={method}")
            return calibrator_instance
        except FileNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to load calibrator: {e}")
            raise RuntimeError(f"Calibrator load failed: {e}") from e
