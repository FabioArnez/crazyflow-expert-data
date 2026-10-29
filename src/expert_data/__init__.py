"""crazyflow-expert-data: velocity-setpoint PPO expert and demonstration pipeline (UAVLA)."""

import os

# crazyflow requires scipy's array-API mode and refuses to import if scipy was imported without it.
os.environ.setdefault("SCIPY_ARRAY_API", "1")

__version__ = "0.1.0"
