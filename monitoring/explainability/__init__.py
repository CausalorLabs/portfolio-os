"""
Monitoring — Decision Explainability & Trade Narrative Engine.

Explains WHY the portfolio changed:
  - Allocation change explanations (combining SHAP, regime, risk, utility)
  - Asset overweight/underweight rationale
  - Trade decision narratives (human-readable reasoning)
  - Decision timeline construction

Every trade produces human-readable reasoning for:
  - Advisors
  - Compliance
  - Client reporting
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger
from omegaconf import OmegaConf


def _load_config() -> dict:
    cfg_path = Path("configs/monitoring.yaml")
    if cfg_path.exists():
        return OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    return {}


# ══════════════════════════════════════════════════════════════════════════════
# Decision Explanation
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class DecisionExplanation:
    """Full explanation for a portfolio decision."""
    timestamp: datetime
    decision_id: str
    decision_type: str  # rebalance | regime_shift | risk_adjustment | tax_harvest
    summary: str
    drivers: list[str]
    regime_context: str
    risk_context: str
    utility_context: str
    confidence: float
    weight_changes: dict[str, float] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


def explain_allocation_change(
    current_weights: dict[str, float],
    target_weights: dict[str, float],
    regime: str = "unknown",
    regime_changed: bool = False,
    confidence: float = 0.5,
    alpha_scores: dict[str, float] | None = None,
    shap_explanations: dict[str, dict] | None = None,
    risk_contributions: dict[str, float] | None = None,
    utility_estimate: Any | None = None,
    volatility_state: dict[str, float] | None = None,
) -> DecisionExplanation:
    """
    Explain why the portfolio allocation changed.

    Combines signals from:
      - SHAP model explanations
      - Regime context
      - Risk contribution shifts
      - Utility engine decisions
      - Volatility changes
    """
    cfg = _load_config().get("explainability", {})
    min_change = cfg.get("min_weight_change_to_explain", 0.02)

    drivers = []
    weight_changes = {}
    increases = []
    decreases = []

    all_assets = set(current_weights) | set(target_weights)
    for asset in sorted(all_assets):
        curr = current_weights.get(asset, 0.0)
        tgt = target_weights.get(asset, 0.0)
        change = tgt - curr
        if abs(change) >= min_change:
            weight_changes[asset] = round(change, 4)
            if change > 0:
                increases.append((asset, change))
            else:
                decreases.append((asset, change))

    # Regime context
    regime_ctx = f"Current regime: {regime}"
    if regime_changed:
        regime_ctx += " (REGIME CHANGE detected)"
        drivers.append(f"Regime transition to {regime}")

    # Risk context
    risk_ctx = "No risk signals"
    if risk_contributions:
        top_risk = sorted(
            risk_contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:3]
        risk_ctx = "Top risk contributors: " + ", ".join(
            f"{t}={v:.1%}" for t, v in top_risk
        )

    # Alpha / SHAP drivers
    if alpha_scores:
        for asset, change in increases:
            score = alpha_scores.get(asset, 0)
            if score > 0.5:
                drivers.append(f"+ {asset}: strong alpha score ({score:.2f})")
        for asset, change in decreases:
            score = alpha_scores.get(asset, 0)
            if score < 0.3:
                drivers.append(f"- {asset}: weak alpha score ({score:.2f})")

    if shap_explanations:
        for asset in weight_changes:
            shap_data = shap_explanations.get(asset, {})
            top_pos = shap_data.get("top_positive", [])
            if top_pos:
                top = top_pos[0] if isinstance(top_pos[0], str) else top_pos[0].get("feature", "")
                drivers.append(f"  {asset} driven by: {top}")

    # Volatility context
    if volatility_state:
        high_vol_assets = [
            a for a, v in volatility_state.items() if v > 0.30
        ]
        if high_vol_assets:
            drivers.append(
                f"Elevated volatility in: {', '.join(high_vol_assets[:3])}"
            )

    # Utility context
    utility_ctx = "No utility analysis"
    if utility_estimate is not None:
        net_u = getattr(utility_estimate, "net_utility", None)
        if net_u is not None:
            utility_ctx = (
                f"Net utility: {net_u:.4f} "
                f"(alpha={getattr(utility_estimate, 'alpha_improvement', 0):.4f}, "
                f"risk_red={getattr(utility_estimate, 'risk_reduction', 0):.4f}, "
                f"friction={getattr(utility_estimate, 'total_friction', 0):.4f})"
            )

    # Build summary
    if not weight_changes:
        summary = "No significant allocation changes."
    else:
        parts = []
        if increases:
            inc_str = ", ".join(f"{a} +{c:.1%}" for a, c in increases[:3])
            parts.append(f"Increased: {inc_str}")
        if decreases:
            dec_str = ", ".join(f"{a} {c:.1%}" for a, c in decreases[:3])
            parts.append(f"Decreased: {dec_str}")
        summary = ". ".join(parts) + "."

    if not drivers:
        drivers.append("Routine portfolio optimization")

    now = datetime.now(timezone.utc)
    explanation = DecisionExplanation(
        timestamp=now,
        decision_id=f"exp_{now.strftime('%Y%m%d_%H%M%S')}",
        decision_type="rebalance",
        summary=summary,
        drivers=drivers,
        regime_context=regime_ctx,
        risk_context=risk_ctx,
        utility_context=utility_ctx,
        confidence=confidence,
        weight_changes=weight_changes,
    )

    logger.info(f"  Explanation: {summary}")
    return explanation


def explain_regime_shift(
    old_regime: str,
    new_regime: str,
    regime_confidence: float,
    regime_features: dict[str, float] | None = None,
) -> DecisionExplanation:
    """Explain a regime transition."""
    drivers = [f"Regime shifted: {old_regime} → {new_regime}"]

    if regime_features:
        for feature, value in sorted(
            regime_features.items(), key=lambda x: abs(x[1]), reverse=True,
        )[:5]:
            drivers.append(f"  {feature}: {value:.4f}")

    severity_map = {
        "panic": "CRITICAL — defensive positioning required",
        "risk_off": "Elevated caution — reducing equity exposure",
        "high_vol": "Increased volatility — tightening risk controls",
        "risk_on": "Favorable conditions — full allocation enabled",
    }
    summary = severity_map.get(new_regime, f"Regime changed to {new_regime}")

    now = datetime.now(timezone.utc)
    return DecisionExplanation(
        timestamp=now,
        decision_id=f"regime_{now.strftime('%Y%m%d_%H%M%S')}",
        decision_type="regime_shift",
        summary=summary,
        drivers=drivers,
        regime_context=f"{old_regime} → {new_regime}",
        risk_context="Regime-driven risk adjustment",
        utility_context="N/A — regime override",
        confidence=regime_confidence,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Trade Decision Narratives
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class TradeNarrative:
    """Human-readable narrative for a single trade decision."""
    decision_id: str
    action: str  # trade | no_trade | harvest
    narrative: str
    bullet_points: list[str]
    confidence: float
    regime: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


def generate_trade_narrative(
    action: str,
    trades: list[dict] | None = None,
    utility_estimate: Any | None = None,
    regime: str = "unknown",
    regime_changed: bool = False,
    confidence: float = 0.5,
    trigger: str = "scheduled",
    turnover_budget_remaining: float | None = None,
) -> TradeNarrative:
    """
    Generate human-readable narrative for a trade or no-trade decision.

    Combines utility, regime, confidence, and cost context into
    a narrative suitable for advisors, compliance, and client reporting.
    """
    bullets = []
    decision_id = f"narr_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    if action == "trade":
        n_trades = len(trades) if trades else 0
        buys = [t for t in (trades or []) if t.get("action") == "BUY"]
        sells = [t for t in (trades or []) if t.get("action") == "SELL"]

        narrative = f"Executed {n_trades} trades"
        if sells:
            narrative += f" ({len(sells)} sells, {len(buys)} buys)"

        bullets.append(f"Trigger: {trigger}")

        if utility_estimate is not None:
            net_u = getattr(utility_estimate, "net_utility", 0)
            alpha = getattr(utility_estimate, "alpha_improvement", 0)
            friction = getattr(utility_estimate, "total_friction", 0)
            bullets.append(
                f"Expected net utility: {net_u:.4f} "
                f"(alpha improvement: {alpha:.4f}, "
                f"estimated friction: {friction:.4f})"
            )

        if regime_changed:
            bullets.append(f"Regime change to {regime} triggered rebalancing")
        else:
            bullets.append(f"Regime: {regime} (stable)")

        bullets.append(f"Confidence: {confidence:.1%}")

        # Trade details
        for t in (trades or [])[:5]:  # Top 5 trades
            ticker = t.get("ticker", "?")
            t_action = t.get("action", "?")
            qty = t.get("quantity", 0)
            notional = t.get("notional", 0)
            bullets.append(
                f"  {t_action} {qty} {ticker} "
                f"(notional: ₹{notional:,.0f})"
            )

    elif action == "no_trade":
        narrative = "Trade evaluation completed — no trades executed"
        bullets.append(f"Trigger: {trigger}")

        if utility_estimate is not None:
            net_u = getattr(utility_estimate, "net_utility", 0)
            friction = getattr(utility_estimate, "total_friction", 0)
            rationale = getattr(utility_estimate, "rationale", "insufficient utility")
            bullets.append(f"Net utility: {net_u:.4f} (below threshold)")
            bullets.append(f"Estimated friction: {friction:.4f}")
            bullets.append(f"Reason: {rationale}")
        else:
            bullets.append("Insufficient drift to trigger rebalancing")

        if turnover_budget_remaining is not None:
            bullets.append(
                f"Turnover budget remaining: {turnover_budget_remaining:.1%}"
            )

    elif action == "harvest":
        narrative = "Tax-loss harvesting opportunity identified"
        n_trades = len(trades) if trades else 0
        bullets.append(f"Harvesting {n_trades} positions with unrealized losses")
        bullets.append(f"Regime: {regime}")

        for t in (trades or [])[:3]:
            ticker = t.get("ticker", "?")
            loss = t.get("unrealized_loss", 0)
            bullets.append(f"  {ticker}: unrealized loss ₹{abs(loss):,.0f}")

    else:
        narrative = f"Decision: {action}"
        bullets.append(f"Trigger: {trigger}")

    result = TradeNarrative(
        decision_id=decision_id,
        action=action,
        narrative=narrative,
        bullet_points=bullets,
        confidence=confidence,
        regime=regime,
    )

    logger.info(f"  Narrative: {narrative}")
    return result


def format_narrative_text(narrative: TradeNarrative) -> str:
    """Format a trade narrative as plain text."""
    lines = [
        f"Decision: {narrative.narrative}",
        f"Time: {narrative.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"ID: {narrative.decision_id}",
        "",
    ]
    for bullet in narrative.bullet_points:
        lines.append(f"• {bullet}")
    return "\n".join(lines)


def format_narrative_markdown(narrative: TradeNarrative) -> str:
    """Format a trade narrative as Markdown."""
    lines = [
        f"## {narrative.narrative}",
        "",
        f"**Time:** {narrative.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}  ",
        f"**Decision ID:** `{narrative.decision_id}`  ",
        f"**Confidence:** {narrative.confidence:.1%}  ",
        f"**Regime:** {narrative.regime}  ",
        "",
    ]
    for bullet in narrative.bullet_points:
        if bullet.startswith("  "):
            lines.append(f"  - {bullet.strip()}")
        else:
            lines.append(f"- {bullet}")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Decision Timeline
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class TimelineEvent:
    """Single event on the decision timeline."""
    timestamp: datetime
    event_type: str  # regime_change | rebalance | alert | anomaly
    title: str
    description: str
    severity: str  # INFO | WARNING | CRITICAL
    decision_id: str | None = None
    metadata: dict = field(default_factory=dict)


class DecisionTimeline:
    """
    Ordered timeline of all portfolio decisions and events.

    Provides a glass-box view of system behavior over time.
    """

    def __init__(self):
        self._events: list[TimelineEvent] = []

    def add_event(
        self,
        event_type: str,
        title: str,
        description: str = "",
        severity: str = "INFO",
        decision_id: str | None = None,
        metadata: dict | None = None,
    ) -> TimelineEvent:
        """Add an event to the timeline."""
        event = TimelineEvent(
            timestamp=datetime.now(timezone.utc),
            event_type=event_type,
            title=title,
            description=description,
            severity=severity,
            decision_id=decision_id,
            metadata=metadata or {},
        )
        self._events.append(event)
        return event

    def add_explanation(self, explanation: DecisionExplanation) -> TimelineEvent:
        """Add a decision explanation as a timeline event."""
        return self.add_event(
            event_type=explanation.decision_type,
            title=explanation.summary,
            description="; ".join(explanation.drivers),
            severity="WARNING" if "regime" in explanation.decision_type.lower() else "INFO",
            decision_id=explanation.decision_id,
            metadata={"weight_changes": explanation.weight_changes},
        )

    def add_narrative(self, narrative: TradeNarrative) -> TimelineEvent:
        """Add a trade narrative as a timeline event."""
        return self.add_event(
            event_type=f"trade_{narrative.action}",
            title=narrative.narrative,
            description="\n".join(narrative.bullet_points),
            severity="INFO" if narrative.action == "no_trade" else "WARNING",
            decision_id=narrative.decision_id,
        )

    def recent(self, n: int = 20) -> list[TimelineEvent]:
        """Get most recent events."""
        return sorted(self._events, key=lambda e: e.timestamp, reverse=True)[:n]

    def by_type(self, event_type: str) -> list[TimelineEvent]:
        """Filter events by type."""
        return [e for e in self._events if e.event_type == event_type]

    def by_severity(self, severity: str) -> list[TimelineEvent]:
        """Filter events by severity."""
        return [e for e in self._events if e.severity == severity]

    def to_dataframe(self) -> pd.DataFrame:
        """Export timeline as DataFrame."""
        if not self._events:
            return pd.DataFrame()
        rows = [
            {
                "timestamp": e.timestamp,
                "event_type": e.event_type,
                "title": e.title,
                "description": e.description,
                "severity": e.severity,
                "decision_id": e.decision_id or "",
            }
            for e in self._events
        ]
        return pd.DataFrame(rows).sort_values("timestamp", ascending=False)

    def summary(self) -> dict:
        """Timeline summary statistics."""
        from collections import Counter
        types = Counter(e.event_type for e in self._events)
        severities = Counter(e.severity for e in self._events)
        return {
            "total_events": len(self._events),
            "by_type": dict(types),
            "by_severity": dict(severities),
        }
