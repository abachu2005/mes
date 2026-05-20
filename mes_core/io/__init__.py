"""I/O loaders for EEG sources used by MES."""

from mes_core.io.generic import load_eeg
from mes_core.io.openbci import load_openbci_txt

__all__ = ["load_eeg", "load_openbci_txt"]
