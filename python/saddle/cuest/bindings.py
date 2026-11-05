"""The single import point for the cuEST Python bindings (spec §14, ADR 0001).

No other Saddle module may import `cuest`; they use `saddle.cuest.bindings.cb`.
"""

from typing import Any

import cuest as _cuest
import cuest.bindings as _bindings

cb: Any = _bindings
CUEST_VERSION: str = str(_cuest.__version__)
