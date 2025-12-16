"""
vanilla_structures package
--------------------------
Contains elementary and commonly-used equity option structures:
- Single vanilla legs (Call, Put)
- Multi-leg combinations (Straddle, Strangle, Spreads, Butterflies, Ratio Spreads, Calendars)
- Risk Reversals
- Custom user-defined strategy container
"""

# --- Naked Vanillas ---
from .vanilla import Call, Put

# --- Custom free-form strategy ---
from .custom import CustomStrategy

# --- Spreads ---
from .spreads import (
    CallSpread,
    PutSpread
)

# --- Straddle & Strangle ---
from .straddle_strangle import (
    Straddle,
    Strangle
)

# --- Risk Reversal ---
from .risk_reversals import RiskReversal

# --- Ratio Spreads ---
from .ratio_spreads import (
    CallRatioSpread,
    PutRatioSpread
)

# --- Butterflies ---
from .butterflies import (
    CallButterfly,
    PutButterfly
)

# --- Calendars ---
from .calendars import (
    CallCalendar,
    PutCalendar
)

__all__ = [
    # Naked
    "Call", "Put",

    # Custom
    "CustomStrategy",

    # Spreads
    "CallSpread", "PutSpread",

    # Straddle / Strangle
    "Straddle", "Strangle",

    # Risk Reversal
    "RiskReversal",

    # Ratio Spreads
    "CallRatioSpread", "PutRatioSpread",

    # Butterflies
    "CallButterfly", "PutButterfly",

    # Calendars
    "CallCalendar", "PutCalendar",
]
