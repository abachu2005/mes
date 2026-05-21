"""Motor Engagement Signal (MES).

Quantifies neural drive for movement recovery from EEG.
"""

__version__ = "0.2.1"

from mes_core.config import OPENBCI_MONTAGE_16, TARGET_SFREQ

__all__ = ["OPENBCI_MONTAGE_16", "TARGET_SFREQ", "__version__"]
