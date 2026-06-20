"""
utils.py — Shared utilities for FinTrust Search.

Provides:
- Min-Max normalization for pandas Series
- Pickle serialization / deserialization helpers
- Logging setup
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DIR = BASE_DIR / "data/processed"
RAW_DIR = BASE_DIR / "data/raw"
INDEX_DIR = BASE_DIR / "index"
RESULTS_DIR = BASE_DIR / "results"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str) -> logging.Logger:
    """Return a consistently formatted logger for the given module name."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s — %(message)s",
                              datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def min_max_normalize(series: pd.Series) -> pd.Series:
    """
    Apply Min-Max normalization to scale values to [0, 1].

    If all values are identical (zero range), returns a zero-filled Series
    to avoid division by zero.

    Args:
        series: Numeric pandas Series to normalize.

    Returns:
        Normalized Series with values in [0, 1].
    """
    s_min = series.min()
    s_max = series.max()
    if s_max - s_min < 1e-9:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - s_min) / (s_max - s_min)


def log1p_normalize(series: pd.Series) -> pd.Series:
    """
    Apply log1p transformation followed by Min-Max normalization.

    Useful for heavy-tailed distributions such as post scores and
    comment counts, where raw values have large outliers.

    Args:
        series: Numeric pandas Series (non-negative values expected).

    Returns:
        Normalized Series with values in [0, 1].
    """
    return min_max_normalize(np.log1p(series))


# ---------------------------------------------------------------------------
# Pickle I/O
# ---------------------------------------------------------------------------

def save_pkl(obj: object, path: Path | str) -> None:
    """
    Serialize an object to a pickle file, creating parent directories
    if they do not exist.

    Args:
        obj:  Any Python object to serialize.
        path: Destination file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    logging.getLogger(__name__).info("Saved → %s", path)


def load_pkl(path: Path | str) -> object:
    """
    Deserialize an object from a pickle file.

    Args:
        path: Source file path.

    Returns:
        The deserialized Python object.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")
    with open(path, "rb") as f:
        obj = pickle.load(f)
    logging.getLogger(__name__).info("Loaded ← %s", path)
    return obj


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    log = get_logger("utils.smoke_test")
    log.info("Running smoke test...")

    # --- normalization ---
    s = pd.Series([0.0, 10.0, 100.0, 1000.0])
    normed = min_max_normalize(s)
    assert normed.min() == 0.0 and normed.max() == 1.0, "min_max_normalize failed"
    log.info("min_max_normalize ✓  %s", normed.tolist())

    log_normed = log1p_normalize(s)
    assert log_normed.min() == 0.0 and log_normed.max() == 1.0, "log1p_normalize failed"
    log.info("log1p_normalize    ✓  %s", log_normed.tolist())

    # --- edge case: zero-range series ---
    flat = pd.Series([5.0, 5.0, 5.0])
    assert (min_max_normalize(flat) == 0.0).all(), "zero-range handling failed"
    log.info("zero-range edge case ✓")

    # --- pickle round-trip ---
    test_path = PROCESSED_DIR / "_smoke_test.pkl"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"key": [1, 2, 3]}
    save_pkl(payload, test_path)
    loaded = load_pkl(test_path)
    assert loaded == payload, "pickle round-trip failed"
    test_path.unlink()  # clean up
    log.info("pickle round-trip  ✓")

    log.info("All smoke tests passed.")
