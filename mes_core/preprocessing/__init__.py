"""EEG preprocessing pipeline (filters, ICA, epoching, montage mapping)."""

from mes_core.preprocessing.pipeline import preprocess_raw, epoch_raw, PreprocessConfig
from mes_core.preprocessing.montage import map_to_openbci_16

__all__ = ["preprocess_raw", "epoch_raw", "PreprocessConfig", "map_to_openbci_16"]
