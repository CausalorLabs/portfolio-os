"""
Mutual Fund NAV loader using mftool.
"""

from pathlib import Path

import pandas as pd
from loguru import logger
from mftool import Mftool


RAW_DIR = Path("data/raw")

mf = Mftool()


def download_mf_data(
    scheme_code: str,
    save: bool = True,
) -> pd.DataFrame:
    """
    Download historical NAV data for an Indian mutual fund scheme.

    Parameters
    ----------
    scheme_code : str
        AMFI scheme code (e.g. "119551" for SBI Bluechip Fund).
    save : bool
        If True, persist to data/raw/MF_<scheme_code>.parquet.

    Returns
    -------
    pd.DataFrame
        Columns: date, nav, ticker
    """
    logger.info(f"Downloading MF scheme {scheme_code}")

    try:
        history = mf.get_scheme_historical_nav(scheme_code, as_Dataframe=True)
    except Exception as e:
        logger.error(f"Failed to download MF {scheme_code}: {e}")
        return pd.DataFrame()

    if history is None or history.empty:
        logger.warning(f"No data returned for MF {scheme_code}")
        return pd.DataFrame()

    df = _normalize(history, scheme_code)
    df = _clean(df)

    logger.info(f"MF {scheme_code}: {len(df)} rows, {df['date'].min()} → {df['date'].max()}")

    if save:
        _save_parquet(df, scheme_code)

    return df


def get_scheme_name(scheme_code: str) -> str:
    """Look up the human-readable name for a scheme code."""
    try:
        details = mf.get_scheme_details(scheme_code)
        return details.get("scheme_name", f"MF_{scheme_code}")
    except Exception:
        return f"MF_{scheme_code}"


# ── internal helpers ─────────────────────────────────────────────────────────


def _normalize(raw: pd.DataFrame, scheme_code: str) -> pd.DataFrame:
    """Standardize mftool output to (date, nav, ticker)."""
    df = raw.copy().reset_index()

    # mftool returns columns like 'date' and 'nav' but naming can vary
    col_map = {}
    for col in df.columns:
        lower = col.strip().lower()
        if lower == "date":
            col_map[col] = "date"
        elif lower in ("nav", "net asset value"):
            col_map[col] = "nav"

    df = df.rename(columns=col_map)

    if "nav" not in df.columns:
        # try the first numeric-looking column
        for col in df.columns:
            if col != "date":
                df = df.rename(columns={col: "nav"})
                break

    df["ticker"] = f"MF_{scheme_code}"

    return df[["date", "nav", "ticker"]]


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce types, remove bad rows, sort."""
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
    df = df.dropna(subset=["date", "nav"])
    df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df


def _save_parquet(df: pd.DataFrame, scheme_code: str) -> Path:
    """Write MF NAV data to parquet."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"MF_{scheme_code}.parquet"
    df.to_parquet(path, index=False, engine="pyarrow")
    logger.info(f"Saved → {path}")
    return path
