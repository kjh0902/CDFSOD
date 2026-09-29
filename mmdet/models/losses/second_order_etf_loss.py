"""Backward-compatible names for the raw mean ETF used by FE experiments."""
from .raw_mean_etf_loss import (
    _raw_mean_prototypes as _second_order_representations,
    raw_mean_etf_loss as second_order_etf_loss,
)
