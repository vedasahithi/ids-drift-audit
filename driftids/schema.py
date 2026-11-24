"""Column roles for the NetFlow v3 datasets.

The exact column list is checked against each file (`phase0.py describe`); a file that lacks the
time or label columns is rejected.
"""
from __future__ import annotations

TIME_START = "FLOW_START_MILLISECONDS"
TIME_END = "FLOW_END_MILLISECONDS"
LABEL_BIN = "Label"
LABEL_CLASS = "Attack"
BENIGN = "Benign"

# Identifiers: never features (network-specific; leak the host or tool). Read is skipped to save memory.
IDENTIFIER_COLS = ("IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT", "L4_DST_PORT")
# Random transaction identifiers: no meaning, but a model can memorise values
TRANSACTION_ID_COLS = ("DNS_QUERY_ID",)
# Absolute time: used only for ordering, never as features.
TIME_COLS = (TIME_START, TIME_END)
LABEL_COLS = (LABEL_BIN, LABEL_CLASS)
EXCLUDED_FROM_FEATURES = IDENTIFIER_COLS + TRANSACTION_ID_COLS + TIME_COLS + LABEL_COLS

REQUIRED_COLS = TIME_COLS + LABEL_COLS

# The 10 temporal features added in v3 (Luay et al. 2025); checked against the files.
EXPECTED_V3_TEMPORAL = (
    TIME_START, TIME_END,
    "SRC_TO_DST_IAT_MIN", "SRC_TO_DST_IAT_MAX", "SRC_TO_DST_IAT_AVG", "SRC_TO_DST_IAT_STDDEV",
    "DST_TO_SRC_IAT_MIN", "DST_TO_SRC_IAT_MAX", "DST_TO_SRC_IAT_AVG", "DST_TO_SRC_IAT_STDDEV",
)


def is_excluded(col: str) -> bool:
    """True for identifier, absolute-time and label columns (also catches renamed address columns)."""
    c = col.upper()
    if col in EXCLUDED_FROM_FEATURES:
        return True
    return "ADDR" in c or c.endswith("_PORT") or c in ("FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS")


def feature_columns(columns) -> list[str]:
    return [c for c in columns if not is_excluded(c)]
