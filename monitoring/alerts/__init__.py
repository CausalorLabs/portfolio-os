"""
Monitoring — Alerting Engine.

Comprehensive alert system covering:
  - Portfolio alerts (concentration, drift, turnover, cash)
  - Risk alerts (vol spike, drawdown, correlation stress, tail risk)
  - Regime alerts (transitions, panic, instability)
  - ML alerts (confidence collapse, feature drift, prediction instability)
  - Operational alerts (pipeline failure, stale data, API failure)

Three severity levels: INFO, WARNING, CRITICAL.

Also re-exports legacy functions from monitoring/alerts.py for backward compat.
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from omegaconf import OmegaConf


def _load_config() -> dict:
    cfg_path = Path("configs/monitoring.yaml")
    if cfg_path.exists():
        return OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    return {}


# ══════════════════════════════════════════════════════════════════════════════
# Alert Data Structures
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class Alert:
    """Unified alert record."""
    alert_id: str
    timestamp: datetime
    category: str      # portfolio | risk | regime | ml | operational
    severity: str      # INFO | WARNING | CRITICAL
    title: str
    message: str
    metric_name: str
    metric_value: float
    threshold: float
    acknowledged: bool = False
    metadata: dict = field(default_factory=dict)


class AlertEngine:
    """
    Centralized alerting system.

    Runs all alert checks and maintains alert history with deduplication.
    """

    def __init__(self):
        self._cfg = _load_config().get("alerts", {})
        self._alerts: list[Alert] = []
        self._seen: dict[str, datetime] = {}  # dedup key → last fired
        self._cooldown_minutes = 15

    def _make_id(self) -> str:
        return uuid.uuid4().hex[:12]

    def _dedup_key(self, category: str, title: str, metric: str) -> str:
        return f"{category}:{title}:{metric}"

    def _should_fire(self, key: str) -> bool:
        """Check cooldown — don't repeat same alert too quickly."""
        last = self._seen.get(key)
        if last is None:
            return True
        elapsed = (datetime.now(timezone.utc) - last).total_seconds() / 60
        return elapsed >= self._cooldown_minutes

    def _fire(
        self,
        category: str,
        severity: str,
        title: str,
        message: str,
        metric_name: str,
        metric_value: float,
        threshold: float,
        metadata: dict | None = None,
    ) -> Alert | None:
        """Fire an alert if not in cooldown."""
        key = self._dedup_key(category, title, metric_name)
        if not self._should_fire(key):
            return None

        alert = Alert(
            alert_id=self._make_id(),
            timestamp=datetime.now(timezone.utc),
            category=category,
            severity=severity,
            title=title,
            message=message,
            metric_name=metric_name,
            metric_value=metric_value,
            threshold=threshold,
            metadata=metadata or {},
        )
        self._alerts.append(alert)
        self._seen[key] = alert.timestamp

        logger.log(
            severity,
            f"  ALERT [{category}/{severity}] {title}: {message}",
        )
        return alert

    # ── Portfolio Alerts ─────────────────────────────────────────────────

    def check_portfolio(
        self,
        weights: dict[str, float],
        target_weights: dict[str, float] | None = None,
        cash_pct: float = 0.0,
        monthly_turnover: float = 0.0,
    ) -> list[Alert]:
        """Run all portfolio alert checks."""
        alerts = []

        # Concentration (HHI)
        hhi_threshold = self._cfg.get("concentration_hhi_threshold", 0.25)
        hhi = sum(w ** 2 for w in weights.values())
        if hhi > hhi_threshold:
            top = max(weights, key=weights.get)  # type: ignore[arg-type]
            a = self._fire(
                "portfolio", "WARNING",
                "High concentration",
                f"HHI={hhi:.3f} (threshold: {hhi_threshold:.3f}). "
                f"Largest: {top} at {weights[top]:.1%}",
                "hhi", hhi, hhi_threshold,
            )
            if a:
                alerts.append(a)

        # Single position limit
        max_pos = self._cfg.get("max_single_position", 0.25)
        for ticker, w in weights.items():
            if w > max_pos:
                a = self._fire(
                    "portfolio", "WARNING",
                    f"Position limit breach: {ticker}",
                    f"{ticker} at {w:.1%} exceeds {max_pos:.1%} limit",
                    "position_weight", w, max_pos,
                    {"ticker": ticker},
                )
                if a:
                    alerts.append(a)

        # Drift
        if target_weights:
            drift_threshold = self._cfg.get("drift_threshold", 0.05)
            for ticker in set(weights) | set(target_weights):
                curr = weights.get(ticker, 0.0)
                tgt = target_weights.get(ticker, 0.0)
                drift = abs(curr - tgt)
                if drift > drift_threshold:
                    sev = "CRITICAL" if drift > drift_threshold * 2 else "WARNING"
                    a = self._fire(
                        "portfolio", sev,
                        f"Allocation drift: {ticker}",
                        f"{ticker} drifted {drift:.1%} "
                        f"(current={curr:.1%}, target={tgt:.1%})",
                        "drift", drift, drift_threshold,
                        {"ticker": ticker},
                    )
                    if a:
                        alerts.append(a)

        # Cash
        max_cash = self._cfg.get("max_cash_pct", 0.20)
        min_cash = self._cfg.get("min_cash_pct", 0.02)
        if cash_pct > max_cash:
            a = self._fire(
                "portfolio", "WARNING",
                "Excessive cash",
                f"Cash at {cash_pct:.1%} exceeds {max_cash:.1%}",
                "cash_pct", cash_pct, max_cash,
            )
            if a:
                alerts.append(a)
        elif cash_pct < min_cash and cash_pct >= 0:
            a = self._fire(
                "portfolio", "WARNING",
                "Low cash",
                f"Cash at {cash_pct:.1%} below {min_cash:.1%} minimum",
                "cash_pct", cash_pct, min_cash,
            )
            if a:
                alerts.append(a)

        # Excessive turnover
        max_turn = self._cfg.get("excessive_turnover_monthly", 0.30)
        if monthly_turnover > max_turn:
            a = self._fire(
                "portfolio", "WARNING",
                "Excessive turnover",
                f"Monthly turnover {monthly_turnover:.1%} exceeds {max_turn:.1%}",
                "monthly_turnover", monthly_turnover, max_turn,
            )
            if a:
                alerts.append(a)

        return alerts

    # ── Risk Alerts ──────────────────────────────────────────────────────

    def check_risk(
        self,
        current_vol: float = 0.0,
        target_vol: float = 0.15,
        current_drawdown: float = 0.0,
        correlation_max: float = 0.0,
        cvar_95: float = 0.0,
    ) -> list[Alert]:
        """Run all risk alert checks."""
        alerts = []

        # Volatility spike
        vol_thresh = self._cfg.get("vol_spike_threshold", 1.5)
        if target_vol > 0:
            vol_ratio = current_vol / target_vol
            if vol_ratio > vol_thresh:
                sev = "CRITICAL" if vol_ratio > vol_thresh * 1.5 else "WARNING"
                a = self._fire(
                    "risk", sev,
                    "Volatility spike",
                    f"Portfolio vol ({current_vol:.1%}) is {vol_ratio:.1f}x "
                    f"target ({target_vol:.1%})",
                    "vol_ratio", vol_ratio, vol_thresh,
                )
                if a:
                    alerts.append(a)

        # Drawdown
        dd_warn = self._cfg.get("drawdown_warning", -0.10)
        dd_crit = self._cfg.get("drawdown_critical", -0.20)
        if current_drawdown < dd_crit:
            a = self._fire(
                "risk", "CRITICAL",
                "Severe drawdown",
                f"Drawdown at {current_drawdown:.1%} "
                f"breaches critical threshold ({dd_crit:.1%})",
                "drawdown", current_drawdown, dd_crit,
            )
            if a:
                alerts.append(a)
        elif current_drawdown < dd_warn:
            a = self._fire(
                "risk", "WARNING",
                "Drawdown warning",
                f"Drawdown at {current_drawdown:.1%} "
                f"approaching critical ({dd_crit:.1%})",
                "drawdown", current_drawdown, dd_warn,
            )
            if a:
                alerts.append(a)

        # Correlation stress
        corr_thresh = self._cfg.get("correlation_stress_threshold", 0.80)
        if correlation_max > corr_thresh:
            a = self._fire(
                "risk", "WARNING",
                "Correlation stress",
                f"Max pairwise correlation {correlation_max:.2f} "
                f"exceeds {corr_thresh:.2f}",
                "correlation_max", correlation_max, corr_thresh,
            )
            if a:
                alerts.append(a)

        # Tail risk (CVaR)
        cvar_thresh = self._cfg.get("tail_risk_cvar_threshold", -0.05)
        if cvar_95 < cvar_thresh:
            a = self._fire(
                "risk", "WARNING",
                "Elevated tail risk",
                f"CVaR(95%) at {cvar_95:.2%} exceeds {cvar_thresh:.2%}",
                "cvar_95", cvar_95, cvar_thresh,
            )
            if a:
                alerts.append(a)

        return alerts

    # ── Regime Alerts ────────────────────────────────────────────────────

    def check_regime(
        self,
        current_regime: str,
        regime_changed: bool = False,
        previous_regime: str | None = None,
        regime_confidence: float = 0.5,
        regime_history: list[str] | None = None,
    ) -> list[Alert]:
        """Run all regime alert checks."""
        alerts = []

        # Panic detection
        if current_regime == "panic":
            a = self._fire(
                "regime", "CRITICAL",
                "PANIC regime detected",
                f"Market in panic mode (confidence: {regime_confidence:.1%}). "
                f"Defensive positioning required.",
                "regime_panic", 1.0, 0.0,
                {"regime": current_regime, "confidence": regime_confidence},
            )
            if a:
                alerts.append(a)

        # Regime transition
        if regime_changed and previous_regime:
            sev = "CRITICAL" if current_regime == "panic" else "WARNING"
            a = self._fire(
                "regime", sev,
                f"Regime change: {previous_regime} → {current_regime}",
                f"Regime transitioned from {previous_regime} to {current_regime} "
                f"(confidence: {regime_confidence:.1%})",
                "regime_change", 1.0, 0.0,
                {"from": previous_regime, "to": current_regime},
            )
            if a:
                alerts.append(a)

        # Regime instability
        if regime_history:
            window = self._cfg.get("regime_instability_window", 10)
            max_changes = self._cfg.get("regime_instability_changes", 3)
            recent = regime_history[-window:]
            n_changes = sum(
                1 for i in range(1, len(recent))
                if recent[i] != recent[i - 1]
            )
            if n_changes >= max_changes:
                a = self._fire(
                    "regime", "WARNING",
                    "Unstable regime state",
                    f"{n_changes} regime changes in last {window} periods",
                    "regime_changes", float(n_changes), float(max_changes),
                )
                if a:
                    alerts.append(a)

        return alerts

    # ── ML Alerts ────────────────────────────────────────────────────────

    def check_ml(
        self,
        confidence: float = 0.5,
        rolling_ic: float | None = None,
        prediction_cv: float | None = None,
        feature_zscores: dict[str, float] | None = None,
    ) -> list[Alert]:
        """Run all ML model alert checks."""
        alerts = []

        # Confidence collapse
        conf_thresh = self._cfg.get("confidence_collapse_threshold", 0.3)
        if confidence < conf_thresh:
            a = self._fire(
                "ml", "CRITICAL",
                "ML confidence collapse",
                f"Model confidence at {confidence:.1%} "
                f"(threshold: {conf_thresh:.1%})",
                "ml_confidence", confidence, conf_thresh,
            )
            if a:
                alerts.append(a)

        # IC degradation
        if rolling_ic is not None:
            ic_thresh = self._cfg.get("ic_degradation_threshold", 0.02)
            if rolling_ic < ic_thresh:
                a = self._fire(
                    "ml", "WARNING",
                    "IC degradation",
                    f"Rolling IC at {rolling_ic:.4f} "
                    f"below {ic_thresh:.4f} threshold",
                    "rolling_ic", rolling_ic, ic_thresh,
                )
                if a:
                    alerts.append(a)

        # Prediction instability
        if prediction_cv is not None:
            cv_thresh = self._cfg.get("prediction_instability_cv", 0.50)
            if prediction_cv > cv_thresh:
                a = self._fire(
                    "ml", "WARNING",
                    "Prediction instability",
                    f"Prediction CV at {prediction_cv:.2f} "
                    f"exceeds {cv_thresh:.2f}",
                    "prediction_cv", prediction_cv, cv_thresh,
                )
                if a:
                    alerts.append(a)

        # Feature drift
        if feature_zscores:
            drift_thresh = self._cfg.get("feature_drift_zscore", 3.0)
            for feature, zscore in feature_zscores.items():
                if abs(zscore) > drift_thresh:
                    a = self._fire(
                        "ml", "WARNING",
                        f"Feature drift: {feature}",
                        f"{feature} z-score={zscore:.2f} "
                        f"(threshold: ±{drift_thresh:.1f})",
                        "feature_zscore", abs(zscore), drift_thresh,
                        {"feature": feature},
                    )
                    if a:
                        alerts.append(a)

        return alerts

    # ── Operational Alerts ───────────────────────────────────────────────

    def check_operational(
        self,
        pipeline_status: dict[str, dict] | None = None,
        api_latency_ms: float | None = None,
        stale_components: list[str] | None = None,
    ) -> list[Alert]:
        """Run all operational alert checks."""
        alerts = []

        # Pipeline failures
        if pipeline_status:
            max_failures = self._cfg.get("max_pipeline_failures", 3)
            for component, status in pipeline_status.items():
                errors = status.get("error_count", 0)
                if errors >= max_failures:
                    a = self._fire(
                        "operational", "CRITICAL",
                        f"Pipeline failure: {component}",
                        f"{component} has {errors} errors "
                        f"(max: {max_failures})",
                        "pipeline_errors", float(errors), float(max_failures),
                        {"component": component},
                    )
                    if a:
                        alerts.append(a)

        # API latency
        if api_latency_ms is not None:
            max_latency = self._cfg.get("max_api_latency_ms", 5000)
            if api_latency_ms > max_latency:
                a = self._fire(
                    "operational", "WARNING",
                    "High API latency",
                    f"API latency {api_latency_ms:.0f}ms "
                    f"exceeds {max_latency}ms threshold",
                    "api_latency_ms", api_latency_ms, float(max_latency),
                )
                if a:
                    alerts.append(a)

        # Stale data
        if stale_components:
            stale_hours = self._cfg.get("stale_data_hours", 24)
            for component in stale_components:
                a = self._fire(
                    "operational", "WARNING",
                    f"Stale data: {component}",
                    f"{component} data is stale (>{stale_hours}h since update)",
                    "stale_data", 1.0, 0.0,
                    {"component": component},
                )
                if a:
                    alerts.append(a)

        return alerts

    # ── Run All Checks ───────────────────────────────────────────────────

    def run_all_checks(
        self,
        weights: dict[str, float] | None = None,
        target_weights: dict[str, float] | None = None,
        cash_pct: float = 0.0,
        monthly_turnover: float = 0.0,
        current_vol: float = 0.0,
        target_vol: float = 0.15,
        current_drawdown: float = 0.0,
        correlation_max: float = 0.0,
        cvar_95: float = 0.0,
        current_regime: str = "risk_on",
        regime_changed: bool = False,
        previous_regime: str | None = None,
        regime_confidence: float = 0.5,
        ml_confidence: float = 0.5,
        rolling_ic: float | None = None,
    ) -> list[Alert]:
        """Run all alert checks and return fired alerts."""
        all_alerts = []

        if weights:
            all_alerts.extend(
                self.check_portfolio(weights, target_weights, cash_pct, monthly_turnover)
            )

        all_alerts.extend(
            self.check_risk(current_vol, target_vol, current_drawdown, correlation_max, cvar_95)
        )

        all_alerts.extend(
            self.check_regime(current_regime, regime_changed, previous_regime, regime_confidence)
        )

        all_alerts.extend(
            self.check_ml(ml_confidence, rolling_ic)
        )

        if all_alerts:
            logger.info(f"  Alert check: {len(all_alerts)} alerts fired")
        return all_alerts

    # ── Query & Export ───────────────────────────────────────────────────

    def recent(self, n: int = 20) -> list[Alert]:
        """Get most recent alerts."""
        return sorted(self._alerts, key=lambda a: a.timestamp, reverse=True)[:n]

    def by_category(self, category: str) -> list[Alert]:
        """Filter alerts by category."""
        return [a for a in self._alerts if a.category == category]

    def by_severity(self, severity: str) -> list[Alert]:
        """Filter alerts by severity."""
        return [a for a in self._alerts if a.severity == severity]

    def unacknowledged(self) -> list[Alert]:
        """Get unacknowledged alerts."""
        return [a for a in self._alerts if not a.acknowledged]

    def acknowledge(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        for a in self._alerts:
            if a.alert_id == alert_id:
                a.acknowledged = True
                return True
        return False

    def to_dataframe(self) -> pd.DataFrame:
        """Export all alerts as DataFrame."""
        if not self._alerts:
            return pd.DataFrame()
        rows = [
            {
                "alert_id": a.alert_id,
                "timestamp": a.timestamp,
                "category": a.category,
                "severity": a.severity,
                "title": a.title,
                "message": a.message,
                "metric_name": a.metric_name,
                "metric_value": a.metric_value,
                "threshold": a.threshold,
                "acknowledged": a.acknowledged,
            }
            for a in self._alerts
        ]
        return pd.DataFrame(rows)

    def summary(self) -> dict:
        """Alert summary statistics."""
        from collections import Counter
        cats = Counter(a.category for a in self._alerts)
        sevs = Counter(a.severity for a in self._alerts)
        return {
            "total_alerts": len(self._alerts),
            "unacknowledged": len(self.unacknowledged()),
            "by_category": dict(cats),
            "by_severity": dict(sevs),
            "critical_count": sevs.get("CRITICAL", 0),
        }

    def save(self, path: str = "data/exports/alerts.parquet") -> None:
        """Save alerts to parquet."""
        df = self.to_dataframe()
        if not df.empty:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(path, index=False)
            logger.info(f"  Saved {len(df)} alerts to {path}")


# ── Legacy backward-compat re-exports from monitoring/alerts.py ──────────────
# These functions are used by older tests and code.

def check_drift(
    current_weights: dict[str, float],
    target_weights: dict[str, float],
    threshold: float = 0.05,
):
    """Legacy: check position drift (delegates to monitoring.alerts.py logic)."""
    from contracts import AlertLevel, DriftAlert

    alerts = []
    now = datetime.now(timezone.utc)
    for ticker in set(current_weights) | set(target_weights):
        current = current_weights.get(ticker, 0.0)
        target = target_weights.get(ticker, 0.0)
        drift = abs(current - target)
        if drift > threshold:
            level = AlertLevel.CRITICAL if drift > threshold * 2 else AlertLevel.WARNING
            alerts.append(DriftAlert(
                timestamp=now, level=level, ticker=ticker,
                current_weight=current, target_weight=target, drift=drift,
                message=f"{ticker} drifted {drift:.1%} (current={current:.1%}, target={target:.1%})",
            ))
    return alerts


def check_drawdown(nav_series, threshold: float = -0.15):
    """Legacy: check drawdown breach."""
    import pandas as pd
    from contracts import AlertLevel, DrawdownAlert

    if nav_series.empty:
        return None
    peak = nav_series.cummax()
    drawdown = (nav_series - peak) / peak
    current_dd = drawdown.iloc[-1]
    if current_dd < threshold:
        return DrawdownAlert(
            timestamp=datetime.now(timezone.utc),
            level=AlertLevel.CRITICAL if current_dd < threshold * 1.5 else AlertLevel.WARNING,
            current_drawdown=float(current_dd),
            threshold=threshold,
            message=f"Drawdown at {current_dd:.1%} breaches {threshold:.1%} threshold",
        )
    return None


def check_concentration(weights: dict[str, float], hhi_threshold: float = 0.25):
    """Legacy: check portfolio concentration."""
    from contracts import AlertLevel, DriftAlert

    hhi = sum(w ** 2 for w in weights.values())
    if hhi > hhi_threshold:
        top_ticker = max(weights, key=weights.get)
        return DriftAlert(
            timestamp=datetime.now(timezone.utc),
            level=AlertLevel.WARNING,
            ticker=top_ticker,
            current_weight=weights[top_ticker],
            target_weight=0.0,
            drift=hhi,
            message=f"Portfolio HHI={hhi:.3f} exceeds {hhi_threshold:.3f}. Largest position: {top_ticker} at {weights[top_ticker]:.1%}",
        )
    return None
