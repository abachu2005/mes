"""EEG preprocessing pipeline (filters, ICA, epoching, montage mapping)."""

from mes_core.preprocessing.montage import map_to_openbci_16
from mes_core.preprocessing.pipeline import PreprocessConfig, epoch_raw, preprocess_raw

__all__ = ["PreprocessConfig", "epoch_raw", "map_to_openbci_16", "preprocess_raw"]
