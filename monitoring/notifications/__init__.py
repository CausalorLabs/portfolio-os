"""
Monitoring — Notification System.

Dispatches alerts to configured channels:
  - Telegram
  - Slack
  - Email

With rate limiting, severity filtering, and digest mode.
"""

from __future__ import annotations

import os
import smtplib
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

from loguru import logger
from omegaconf import OmegaConf


def _load_config() -> dict:
    cfg_path = Path("configs/monitoring.yaml")
    if cfg_path.exists():
        return OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    return {}


SEVERITY_ORDER = {"INFO": 0, "WARNING": 1, "CRITICAL": 2}


# ══════════════════════════════════════════════════════════════════════════════
# Channel Abstraction
# ══════════════════════════════════════════════════════════════════════════════


class NotificationChannel(ABC):
    """Abstract notification channel."""

    def __init__(self, name: str, min_severity: str = "WARNING"):
        self.name = name
        self.min_severity = min_severity
        self._sent_count = 0
        self._last_sent: dict[str, datetime] = {}

    def should_send(self, severity: str) -> bool:
        """Check if alert meets minimum severity for this channel."""
        return SEVERITY_ORDER.get(severity, 0) >= SEVERITY_ORDER.get(
            self.min_severity, 1
        )

    @abstractmethod
    def send(self, title: str, message: str, severity: str) -> bool:
        """Send a notification. Returns True on success."""

    @property
    def stats(self) -> dict:
        return {"channel": self.name, "sent": self._sent_count}


class TelegramChannel(NotificationChannel):
    """Telegram notification channel."""

    def __init__(self, bot_token: str, chat_id: str, min_severity: str = "WARNING"):
        super().__init__("telegram", min_severity)
        self._bot_token = bot_token
        self._chat_id = chat_id

    def send(self, title: str, message: str, severity: str) -> bool:
        if not self.should_send(severity):
            return False

        try:
            import urllib.request
            import json

            emoji = {"INFO": "ℹ️", "WARNING": "⚠️", "CRITICAL": "🚨"}.get(severity, "📌")
            text = f"{emoji} *{severity}: {title}*\n\n{message}"

            url = f"https://api.telegram.org/bot{self._bot_token}/sendMessage"
            data = json.dumps({
                "chat_id": self._chat_id,
                "text": text,
                "parse_mode": "Markdown",
            }).encode()

            req = urllib.request.Request(
                url, data=data,
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=10)
            self._sent_count += 1
            logger.debug(f"  Telegram notification sent: {title}")
            return True

        except Exception as e:
            logger.error(f"  Telegram send failed: {e}")
            return False


class SlackChannel(NotificationChannel):
    """Slack notification channel via webhook."""

    def __init__(self, webhook_url: str, channel: str = "#portfolio-alerts",
                 min_severity: str = "WARNING"):
        super().__init__("slack", min_severity)
        self._webhook_url = webhook_url
        self._channel = channel

    def send(self, title: str, message: str, severity: str) -> bool:
        if not self.should_send(severity):
            return False

        try:
            import urllib.request
            import json

            color = {"INFO": "#36a64f", "WARNING": "#ff9900", "CRITICAL": "#ff0000"}.get(severity, "#cccccc")

            payload = {
                "channel": self._channel,
                "attachments": [{
                    "color": color,
                    "title": f"{severity}: {title}",
                    "text": message,
                    "footer": "Portfolio OS Alerts",
                    "ts": int(datetime.now(timezone.utc).timestamp()),
                }],
            }

            data = json.dumps(payload).encode()
            req = urllib.request.Request(
                self._webhook_url, data=data,
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=10)
            self._sent_count += 1
            logger.debug(f"  Slack notification sent: {title}")
            return True

        except Exception as e:
            logger.error(f"  Slack send failed: {e}")
            return False


class EmailChannel(NotificationChannel):
    """Email notification channel via SMTP."""

    def __init__(
        self,
        smtp_host: str,
        smtp_port: int,
        sender: str,
        password: str,
        recipients: list[str],
        min_severity: str = "CRITICAL",
    ):
        super().__init__("email", min_severity)
        self._smtp_host = smtp_host
        self._smtp_port = smtp_port
        self._sender = sender
        self._password = password
        self._recipients = recipients

    def send(self, title: str, message: str, severity: str) -> bool:
        if not self.should_send(severity):
            return False

        try:
            msg = MIMEMultipart()
            msg["From"] = self._sender
            msg["To"] = ", ".join(self._recipients)
            msg["Subject"] = f"[Portfolio OS {severity}] {title}"

            body = f"""
Portfolio OS Alert
==================
Severity: {severity}
Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}

{title}
------
{message}
"""
            msg.attach(MIMEText(body, "plain"))

            with smtplib.SMTP(self._smtp_host, self._smtp_port) as server:
                server.starttls()
                server.login(self._sender, self._password)
                server.sendmail(
                    self._sender, self._recipients, msg.as_string(),
                )

            self._sent_count += 1
            logger.debug(f"  Email notification sent: {title}")
            return True

        except Exception as e:
            logger.error(f"  Email send failed: {e}")
            return False


class LogChannel(NotificationChannel):
    """Local log-only channel (always available, no external deps)."""

    def __init__(self, min_severity: str = "INFO"):
        super().__init__("log", min_severity)

    def send(self, title: str, message: str, severity: str) -> bool:
        if not self.should_send(severity):
            return False
        logger.log(severity, f"  NOTIFICATION [{severity}] {title}: {message}")
        self._sent_count += 1
        return True


# ══════════════════════════════════════════════════════════════════════════════
# Notification Dispatcher
# ══════════════════════════════════════════════════════════════════════════════


class NotificationDispatcher:
    """
    Routes alerts to appropriate notification channels.

    Supports rate limiting and digest mode.
    """

    def __init__(self):
        cfg = _load_config().get("notifications", {})
        self._channels: list[NotificationChannel] = []
        self._rate_limit: dict[str, list[datetime]] = defaultdict(list)
        self._max_per_hour = cfg.get("max_alerts_per_hour", 20)
        self._cooldown_minutes = cfg.get("cooldown_minutes", 15)
        self._digest_mode = cfg.get("digest_mode", False)
        self._digest_buffer: list[tuple[str, str, str]] = []

        # Always add log channel
        self._channels.append(LogChannel())

        # Configure channels from env vars
        channels_cfg = cfg.get("channels", {})
        self._configure_channels(channels_cfg)

    def _configure_channels(self, channels_cfg: dict) -> None:
        """Configure notification channels from config + env vars."""
        # Telegram
        tg = channels_cfg.get("telegram", {})
        if tg.get("enabled", False):
            token = os.environ.get(tg.get("bot_token_env", ""), "")
            chat_id = os.environ.get(tg.get("chat_id_env", ""), "")
            if token and chat_id:
                self._channels.append(
                    TelegramChannel(token, chat_id, tg.get("min_severity", "WARNING"))
                )
                logger.info("  Telegram notifications enabled")

        # Slack
        slack = channels_cfg.get("slack", {})
        if slack.get("enabled", False):
            webhook = os.environ.get(slack.get("webhook_url_env", ""), "")
            if webhook:
                self._channels.append(
                    SlackChannel(
                        webhook,
                        slack.get("channel", "#portfolio-alerts"),
                        slack.get("min_severity", "WARNING"),
                    )
                )
                logger.info("  Slack notifications enabled")

        # Email
        email_cfg = channels_cfg.get("email", {})
        if email_cfg.get("enabled", False):
            sender = os.environ.get(email_cfg.get("sender_env", ""), "")
            password = os.environ.get(email_cfg.get("password_env", ""), "")
            recipients_str = os.environ.get(email_cfg.get("recipients_env", ""), "")
            if sender and password and recipients_str:
                self._channels.append(
                    EmailChannel(
                        email_cfg.get("smtp_host", "smtp.gmail.com"),
                        email_cfg.get("smtp_port", 587),
                        sender, password,
                        [r.strip() for r in recipients_str.split(",")],
                        email_cfg.get("min_severity", "CRITICAL"),
                    )
                )
                logger.info("  Email notifications enabled")

    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits."""
        now = datetime.now(timezone.utc)
        hour_ago = now.timestamp() - 3600

        # Clean old entries
        recent = [t for t in self._rate_limit.get("global", [])
                  if t.timestamp() > hour_ago]
        self._rate_limit["global"] = recent

        return len(recent) < self._max_per_hour

    def dispatch(self, title: str, message: str, severity: str) -> int:
        """
        Dispatch a notification to all configured channels.

        Returns number of channels that successfully sent.
        """
        if self._digest_mode:
            self._digest_buffer.append((title, message, severity))
            return 0

        if not self._check_rate_limit():
            logger.warning("  Notification rate limit reached")
            return 0

        sent = 0
        for channel in self._channels:
            if channel.send(title, message, severity):
                sent += 1

        self._rate_limit["global"].append(datetime.now(timezone.utc))
        return sent

    def dispatch_alert(self, alert) -> int:
        """Dispatch an Alert object."""
        return self.dispatch(alert.title, alert.message, alert.severity)

    def flush_digest(self) -> int:
        """Send buffered digest notifications."""
        if not self._digest_buffer:
            return 0

        # Group by severity
        by_severity: dict[str, list] = defaultdict(list)
        for title, message, severity in self._digest_buffer:
            by_severity[severity].append(f"• {title}: {message}")

        sent = 0
        for severity in ["CRITICAL", "WARNING", "INFO"]:
            items = by_severity.get(severity, [])
            if items:
                digest_title = f"Alert Digest: {len(items)} {severity} alerts"
                digest_message = "\n".join(items[:20])
                sent += self.dispatch(digest_title, digest_message, severity)

        self._digest_buffer.clear()
        return sent

    def add_channel(self, channel: NotificationChannel) -> None:
        """Add a notification channel."""
        self._channels.append(channel)

    def summary(self) -> dict:
        """Notification dispatcher summary."""
        return {
            "channels": [c.stats for c in self._channels],
            "rate_limit_remaining": max(
                0,
                self._max_per_hour - len(self._rate_limit.get("global", [])),
            ),
            "digest_buffered": len(self._digest_buffer),
        }
