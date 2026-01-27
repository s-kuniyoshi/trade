"""
Model comparison and evaluation module.

Provides comprehensive model comparison functionality including metrics comparison,
statistical significance testing, and detailed reporting.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from ..utils.config import get_config
from ..utils.logger import get_logger
from .evaluation import PerformanceMetrics
from .registry import ModelRegistry

logger = get_logger("training.model_comparison")


# =============================================================================
# Comparison Result Data Classes
# =============================================================================

@dataclass
class MetricComparison:
    """Comparison result for a single metric."""
    metric_name: str
    values: dict[str, float]  # model_name -> value
    baseline_value: float
    best_model: str
    best_value: float
    improvements: dict[str, float]  # model_name -> improvement_pct
    p_values: dict[str, float] = field(default_factory=dict)  # model_name -> p_value
    is_significant: dict[str, bool] = field(default_factory=dict)  # model_name -> is_sig
    higher_is_better: bool = True
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metric_name": self.metric_name,
            "values": self.values,
            "baseline_value": self.baseline_value,
            "best_model": self.best_model,
            "best_value": self.best_value,
            "improvements": self.improvements,
            "p_values": self.p_values,
            "is_significant": self.is_significant,
            "higher_is_better": self.higher_is_better,
        }


@dataclass
class ComparisonResult:
    """Complete comparison result."""
    models: dict[str, dict[str, Any]]  # model_name -> {version, is_baseline, ...}
    metrics: dict[str, MetricComparison]  # metric_name -> MetricComparison
    baseline_model: str
    comparison_date: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "models": self.models,
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
            "baseline_model": self.baseline_model,
            "comparison_date": self.comparison_date,
        }


# =============================================================================
# Model Comparator
# =============================================================================

class ModelComparator:
    """
    Comprehensive model comparison and evaluation tool.
    
    Compares multiple model versions by metrics, performs statistical significance
    testing, and generates detailed comparison reports.
    
    Usage:
        comparator = ModelComparator()
        comparator.add_model("20260126_103000", name="baseline")
        comparator.add_model("20260127_140000", name="new_model")
        comparison = comparator.compare_metrics()
        report = comparator.generate_report(format="text")
        print(report)
    """
    
    # Metrics where higher is better
    HIGHER_IS_BETTER = {
        "direction_accuracy", "profit_factor", "sharpe_ratio", "sortino_ratio",
        "calmar_ratio", "r_squared", "annual_return", "total_return", "win_rate",
        "correlation"
    }
    
    # Metrics where lower is better
    LOWER_IS_BETTER = {
        "rmse", "mae", "mse", "max_drawdown", "avg_drawdown", "volatility"
    }
    
    def __init__(self):
        """Initialize model comparator."""
        self.registry = ModelRegistry()
        self.models: dict[str, dict[str, Any]] = {}  # model_name -> {version, result, ...}
        self.baseline_model: str | None = None
        self.comparison_result: ComparisonResult | None = None
        logger.info("ModelComparator initialized")
    
    def add_model(
        self,
        version: str,
        name: str | None = None,
        is_baseline: bool = False,
    ) -> bool:
        """
        Add a model to comparison.
        
        Args:
            version: Model version name (e.g., "20260126_103000" or "latest")
            name: Display name for model (if None, use version)
            is_baseline: Mark as baseline for comparison
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load model
            result = self.registry.load_version(version)
            if result is None:
                logger.warning(f"Failed to load model version: {version}")
                return False
            
            # Get metadata
            version_info = self.registry.get_version_info(version)
            if version_info is None:
                logger.warning(f"Failed to get version info: {version}")
                return False
            
            # Use version as name if not provided
            display_name = name or version
            
            # Store model
            self.models[display_name] = {
                "version": version,
                "result": result,
                "metrics": result.metrics,
                "is_baseline": is_baseline,
                "metadata": version_info,
            }
            
            # Set baseline if marked or if first model
            if is_baseline or self.baseline_model is None:
                self.baseline_model = display_name
            
            logger.info(
                f"Added model: {display_name} (version={version}, "
                f"baseline={is_baseline})"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to add model {version}: {e}")
            return False
    
    def _get_metric_value(self, metrics: PerformanceMetrics, metric_name: str) -> float:
        """Extract metric value from PerformanceMetrics."""
        try:
            return getattr(metrics, metric_name, np.nan)
        except AttributeError:
            return np.nan
    
    def _is_higher_better(self, metric_name: str) -> bool:
        """Determine if higher values are better for a metric."""
        if metric_name in self.HIGHER_IS_BETTER:
            return True
        elif metric_name in self.LOWER_IS_BETTER:
            return False
        else:
            # Default: assume higher is better
            return True
    
    def _calculate_improvement(
        self,
        baseline_value: float,
        current_value: float,
        higher_is_better: bool,
    ) -> float:
        """Calculate percentage improvement."""
        if baseline_value == 0 or np.isnan(baseline_value) or np.isnan(current_value):
            return 0.0
        
        if higher_is_better:
            return ((current_value - baseline_value) / abs(baseline_value)) * 100
        else:
            return ((baseline_value - current_value) / abs(baseline_value)) * 100
    
    def compare_metrics(self) -> ComparisonResult | None:
        """
        Compare metrics across all models.
        
        Returns:
            ComparisonResult or None if comparison fails
        """
        if not self.models or self.baseline_model is None:
            logger.error("No models added for comparison")
            return None
        
        try:
            metrics_dict: dict[str, MetricComparison] = {}
            
            # Get all metric names from baseline
            baseline_metrics = self.models[self.baseline_model]["metrics"]
            baseline_dict = baseline_metrics.to_dict()
            
            # Compare each metric
            for metric_name in baseline_dict.keys():
                # Get values for all models
                values = {}
                for model_name, model_info in self.models.items():
                    value = self._get_metric_value(model_info["metrics"], metric_name)
                    values[model_name] = value
                
                baseline_value = values[self.baseline_model]
                
                # Skip if baseline is NaN
                if np.isnan(baseline_value):
                    continue
                
                # Determine if higher is better
                higher_is_better = self._is_higher_better(metric_name)
                
                # Find best model
                valid_values = {k: v for k, v in values.items() if not np.isnan(v)}
                if not valid_values:
                    continue
                
                if higher_is_better:
                    best_model = max(valid_values, key=lambda k: valid_values[k])
                    best_value = valid_values[best_model]
                else:
                    best_model = min(valid_values, key=lambda k: valid_values[k])
                    best_value = valid_values[best_model]
                
                # Calculate improvements
                improvements = {}
                for model_name, value in values.items():
                    if not np.isnan(value):
                        improvement = self._calculate_improvement(
                            baseline_value, value, higher_is_better
                        )
                        improvements[model_name] = improvement
                
                # Create comparison
                comparison = MetricComparison(
                    metric_name=metric_name,
                    values=values,
                    baseline_value=baseline_value,
                    best_model=best_model,
                    best_value=best_value,
                    improvements=improvements,
                    higher_is_better=higher_is_better,
                )
                
                metrics_dict[metric_name] = comparison
            
            # Create result
            models_info = {}
            for model_name, model_info in self.models.items():
                models_info[model_name] = {
                    "version": model_info["version"],
                    "is_baseline": model_info["is_baseline"],
                }
            
            self.comparison_result = ComparisonResult(
                models=models_info,
                metrics=metrics_dict,
                baseline_model=self.baseline_model,
            )
            
            logger.info(
                f"Comparison complete: {len(self.models)} models, "
                f"{len(metrics_dict)} metrics"
            )
            return self.comparison_result
        except Exception as e:
            logger.error(f"Comparison failed: {e}")
            return None
    
    def statistical_test(
        self,
        metric_name: str,
        alpha: float = 0.05,
    ) -> dict[str, Any] | None:
        """
        Perform statistical significance test for a metric.
        
        Uses paired t-test for normally distributed metrics and Wilcoxon
        signed-rank test for non-normal metrics.
        
        Args:
            metric_name: Metric to test
            alpha: Significance level (default 0.05)
            
        Returns:
            Dictionary with test results or None if test fails
        """
        if self.baseline_model is None or metric_name not in self.models[self.baseline_model]["result"].metrics.to_dict():
            logger.warning(f"Cannot test metric: {metric_name}")
            return None
        
        try:
            baseline_result = self.models[self.baseline_model]["result"]
            baseline_predictions = baseline_result.validation_predictions
            
            if baseline_predictions is None or baseline_predictions.empty:
                logger.warning(f"No validation predictions for baseline model")
                return None
            
            baseline_actual = baseline_predictions["actual"].values
            baseline_pred = baseline_predictions["predicted"].values
            
            # Calculate baseline residuals
            baseline_residuals = baseline_actual - baseline_pred
            
            test_results = {}
            
            # Test against each other model
            for model_name, model_info in self.models.items():
                if model_name == self.baseline_model:
                    continue
                
                model_result = model_info["result"]
                model_predictions = model_result.validation_predictions
                
                if model_predictions is None or model_predictions.empty:
                    logger.warning(f"No validation predictions for {model_name}")
                    continue
                
                # Align predictions
                common_idx = baseline_predictions.index.intersection(model_predictions.index)
                if len(common_idx) < 2:
                    logger.warning(f"Insufficient common data for {model_name}")
                    continue
                
                model_actual = model_predictions.loc[common_idx, "actual"].values
                model_pred = model_predictions.loc[common_idx, "predicted"].values
                model_residuals = model_actual - model_pred
                
                # Align baseline residuals
                aligned_baseline_residuals = baseline_predictions.loc[common_idx, "actual"].values - baseline_predictions.loc[common_idx, "predicted"].values
                
                # Test normality (Shapiro-Wilk)
                _, p_baseline_norm = stats.shapiro(aligned_baseline_residuals)
                _, p_model_norm = stats.shapiro(model_residuals)
                
                is_normal = p_baseline_norm > alpha and p_model_norm > alpha
                
                # Perform appropriate test
                if is_normal:
                    # Paired t-test
                    t_stat, p_value = stats.ttest_rel(aligned_baseline_residuals, model_residuals)
                    test_type = "paired_t_test"
                else:
                    # Wilcoxon signed-rank test
                    w_stat, p_value = stats.wilcoxon(aligned_baseline_residuals, model_residuals)
                    t_stat = w_stat
                    test_type = "wilcoxon"
                
                test_results[model_name] = {
                    "test_type": test_type,
                    "statistic": float(t_stat),
                    "p_value": float(p_value),
                    "is_significant": p_value < alpha,
                    "alpha": alpha,
                    "n_samples": len(common_idx),
                }
            
            logger.info(f"Statistical test complete for {metric_name}")
            return test_results
        except Exception as e:
            logger.error(f"Statistical test failed: {e}")
            return None
    
    def identify_best(self, metric_name: str) -> dict[str, Any] | None:
        """
        Identify best model for a specific metric.
        
        Args:
            metric_name: Metric to evaluate
            
        Returns:
            Dictionary with best model info or None if not found
        """
        if self.comparison_result is None:
            logger.warning("No comparison result available")
            return None
        
        if metric_name not in self.comparison_result.metrics:
            logger.warning(f"Metric not found: {metric_name}")
            return None
        
        try:
            metric_comp = self.comparison_result.metrics[metric_name]
            
            return {
                "metric": metric_name,
                "best_model": metric_comp.best_model,
                "value": metric_comp.best_value,
                "version": self.models[metric_comp.best_model]["version"],
                "higher_is_better": metric_comp.higher_is_better,
                "all_values": metric_comp.values,
            }
        except Exception as e:
            logger.error(f"Failed to identify best model: {e}")
            return None
    
    def generate_report(self, format: str = "text") -> str:
        """
        Generate comparison report.
        
        Args:
            format: Report format ("text", "json", or "html")
            
        Returns:
            Formatted report string
        """
        if self.comparison_result is None:
            logger.warning("No comparison result available")
            return ""
        
        if format == "json":
            return self._generate_json_report()
        elif format == "html":
            return self._generate_html_report()
        else:
            return self._generate_text_report()
    
    def _generate_text_report(self) -> str:
        """Generate text format report."""
        if self.comparison_result is None:
            return ""
        
        lines: list[str] = []
        lines.append("MODEL COMPARISON REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        # Header
        lines.append(f"Comparison Date: {self.comparison_result.comparison_date}")
        lines.append(f"Models Compared: {len(self.comparison_result.models)}")
        lines.append("")
        
        # Models list
        lines.append("Models:")
        for model_name, model_info in self.comparison_result.models.items():
            baseline_marker = " (baseline)" if model_name == self.comparison_result.baseline_model else ""
            lines.append(f"  - {model_name}: {model_info['version']}{baseline_marker}")
        lines.append("")
        
        # Metrics comparison
        lines.append("METRICS COMPARISON")
        lines.append("-" * 80)
        lines.append("")
        
        for metric_name, metric_comp in self.comparison_result.metrics.items():
            lines.append(f"{metric_name}:")
            
            # Values for each model
            for model_name in self.comparison_result.models.keys():
                value = metric_comp.values.get(model_name, np.nan)
                
                if np.isnan(value):
                    lines.append(f"  {model_name}: N/A")
                    continue
                
                # Format value
                if isinstance(value, float):
                    if abs(value) < 0.01 and value != 0:
                        value_str = f"{value:.6f}"
                    else:
                        value_str = f"{value:.4f}"
                else:
                    value_str = str(value)
                
                # Add baseline marker
                if model_name == self.comparison_result.baseline_model:
                    lines.append(f"  {model_name}: {value_str} (baseline)")
                else:
                    # Add improvement
                    improvement = metric_comp.improvements.get(model_name, 0)
                    p_value = metric_comp.p_values.get(model_name)
                    is_sig = metric_comp.is_significant.get(model_name, False)
                    
                    improvement_str = f"{improvement:+.1f}%"
                    
                    if p_value is not None:
                        sig_marker = " ✓ SIGNIFICANT" if is_sig else ""
                        lines.append(
                            f"  {model_name}: {value_str} ({improvement_str}, "
                            f"p={p_value:.4f}){sig_marker}"
                        )
                    else:
                        lines.append(f"  {model_name}: {value_str} ({improvement_str})")
            
            # Best model
            lines.append(f"  BEST: {metric_comp.best_model}")
            lines.append("")
        
        # Recommendation
        lines.append("RECOMMENDATION")
        lines.append("-" * 80)
        
        # Count significant improvements
        sig_count = 0
        best_models = {}
        for metric_name, metric_comp in self.comparison_result.metrics.items():
            if metric_comp.best_model != self.comparison_result.baseline_model:
                for model_name in self.comparison_result.models.keys():
                    if model_name != self.comparison_result.baseline_model:
                        if metric_comp.is_significant.get(model_name, False):
                            sig_count += 1
                            best_models[metric_comp.best_model] = best_models.get(metric_comp.best_model, 0) + 1
        
        if best_models:
            best_overall = max(best_models, key=lambda k: best_models[k])
            lines.append(
                f"Model {best_overall} shows significant improvement in "
                f"{best_models[best_overall]} metrics."
            )
            lines.append(f"Recommended for production deployment.")
        else:
            lines.append("No significant improvements detected.")
            lines.append("Recommend further investigation or model refinement.")
        
        lines.append("")
        
        return "\n".join(lines)
    
    def _generate_json_report(self) -> str:
        """Generate JSON format report."""
        if self.comparison_result is None:
            return "{}"
        
        report_dict = {
            "models": self.comparison_result.models,
            "metrics": {},
            "baseline_model": self.comparison_result.baseline_model,
            "comparison_date": self.comparison_result.comparison_date,
        }
        
        for metric_name, metric_comp in self.comparison_result.metrics.items():
            report_dict["metrics"][metric_name] = {
                "values": {k: (float(v) if not np.isnan(v) else None) for k, v in metric_comp.values.items()},
                "baseline_value": float(metric_comp.baseline_value) if not np.isnan(metric_comp.baseline_value) else None,
                "best_model": metric_comp.best_model,
                "best_value": float(metric_comp.best_value) if not np.isnan(metric_comp.best_value) else None,
                "improvements": {k: float(v) for k, v in metric_comp.improvements.items()},
                "p_values": {k: float(v) for k, v in metric_comp.p_values.items()},
                "is_significant": metric_comp.is_significant,
                "higher_is_better": metric_comp.higher_is_better,
            }
        
        return json.dumps(report_dict, indent=2)
    
    def _generate_html_report(self) -> str:
        """Generate HTML format report."""
        if self.comparison_result is None:
            return ""
        
        html_lines = []
        html_lines.append("<!DOCTYPE html>")
        html_lines.append("<html>")
        html_lines.append("<head>")
        html_lines.append("<title>Model Comparison Report</title>")
        html_lines.append("<style>")
        html_lines.append("body { font-family: Arial, sans-serif; margin: 20px; }")
        html_lines.append("h1 { color: #333; }")
        html_lines.append("table { border-collapse: collapse; width: 100%; margin: 20px 0; }")
        html_lines.append("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }")
        html_lines.append("th { background-color: #4CAF50; color: white; }")
        html_lines.append("tr:nth-child(even) { background-color: #f2f2f2; }")
        html_lines.append(".positive { color: green; }")
        html_lines.append(".negative { color: red; }")
        html_lines.append(".significant { font-weight: bold; }")
        html_lines.append("</style>")
        html_lines.append("</head>")
        html_lines.append("<body>")
        
        html_lines.append("<h1>Model Comparison Report</h1>")
        html_lines.append(f"<p>Comparison Date: {self.comparison_result.comparison_date}</p>")
        
        # Models table
        html_lines.append("<h2>Models</h2>")
        html_lines.append("<table>")
        html_lines.append("<tr><th>Model</th><th>Version</th><th>Baseline</th></tr>")
        for model_name, model_info in self.comparison_result.models.items():
            is_baseline = "Yes" if model_name == self.comparison_result.baseline_model else "No"
            html_lines.append(
                f"<tr><td>{model_name}</td><td>{model_info['version']}</td>"
                f"<td>{is_baseline}</td></tr>"
            )
        html_lines.append("</table>")
        
        # Metrics table
        html_lines.append("<h2>Metrics Comparison</h2>")
        html_lines.append("<table>")
        html_lines.append("<tr><th>Metric</th>")
        for model_name in self.comparison_result.models.keys():
            html_lines.append(f"<th>{model_name}</th>")
        html_lines.append("<th>Best</th></tr>")
        
        for metric_name, metric_comp in self.comparison_result.metrics.items():
            html_lines.append(f"<tr><td><strong>{metric_name}</strong></td>")
            
            for model_name in self.comparison_result.models.keys():
                value = metric_comp.values.get(model_name, np.nan)
                
                if np.isnan(value):
                    html_lines.append("<td>N/A</td>")
                else:
                    if isinstance(value, float):
                        value_str = f"{value:.4f}"
                    else:
                        value_str = str(value)
                    
                    if model_name != self.comparison_result.baseline_model:
                        improvement = metric_comp.improvements.get(model_name, 0)
                        is_sig = metric_comp.is_significant.get(model_name, False)
                        
                        improvement_class = "positive" if improvement > 0 else "negative"
                        sig_class = "significant" if is_sig else ""
                        
                        html_lines.append(
                            f"<td class='{improvement_class} {sig_class}'>"
                            f"{value_str} ({improvement:+.1f}%)</td>"
                        )
                    else:
                        html_lines.append(f"<td>{value_str}</td>")
            
            html_lines.append(f"<td><strong>{metric_comp.best_model}</strong></td></tr>")
        
        html_lines.append("</table>")
        html_lines.append("</body>")
        html_lines.append("</html>")
        
        return "\n".join(html_lines)
    
    def export_results(self, path: Path | str, format: str = "json") -> bool:
        """
        Export comparison results to file.
        
        Args:
            path: Output file path
            format: Export format ("json", "text", or "html")
            
        Returns:
            True if successful, False otherwise
        """
        try:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            report = self.generate_report(format=format)
            
            with open(path, "w") as f:
                f.write(report)
            
            logger.info(f"Comparison results exported to {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to export results: {e}")
            return False
