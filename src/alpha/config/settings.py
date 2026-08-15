"""
Central configuration for Alpha. Every layer reads constants from here
instead of hardcoding them — this is the one file that changes if, say,
the table position or IK tolerance needs tuning after real physics runs.
"""

from alpha.core.entities import BlockColor

# --- URDF assets (pybullet_data built-ins; swap for custom URDFs later if needed) ---
ARM_URDF = "kuka_iiwa/model.urdf"
TABLE_URDF = "table/table.urdf"
BLOCK_URDF = "cube_small.urdf"

# --- Scene layout ---
BLOCK_SPAWN_POSITIONS: dict[BlockColor, tuple[float, float, float]] = {
    BlockColor.RED: (0.5, -0.15, 0.68),
    BlockColor.GREEN: (0.5, 0.0, 0.68),
    BlockColor.BLUE: (0.5, 0.15, 0.68),
}
BLOCK_RGBA: dict[BlockColor, tuple[float, float, float, float]] = {
    BlockColor.RED: (0.8, 0.1, 0.1, 1),
    BlockColor.GREEN: (0.1, 0.7, 0.1, 1),
    BlockColor.BLUE: (0.1, 0.1, 0.8, 1),
}
PLACE_ZONE_POSITION = (0.6, 0.35, 0.65)
TABLE_BASE_POSITION = (0.6, 0.0, 0.0)
ARM_BASE_POSITION = (0.0, 0.0, 0.0)

# --- Physics ---
GRAVITY = (0, 0, -9.8)
SETTLE_STEPS = 120
MOVE_STEPS = 240

# --- Control ---
POSITION_TOLERANCE = 0.02
PLACE_SUCCESS_TOLERANCE = 0.08  # loose: "landed in the zone", not exact
PLACE_RELEASE_HEIGHT_OFFSET = 0.06  # release slightly above table so the block settles naturally

# --- Metrics / persistence (used from Week 5 onward, referenced by repos/ now) ---
RESULTS_DIR = "results"
EPISODES_CSV = f"{RESULTS_DIR}/episodes.csv"
EXPERIMENTS_CSV = f"{RESULTS_DIR}/experiments_log.csv"
DEV_SMOKE_DIR = f"{RESULTS_DIR}/dev_smoke"
PILOT_RESULTS_DIR = f"{RESULTS_DIR}/pilot"
FINAL_RESULTS_DIR = f"{RESULTS_DIR}/final"
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"

# --- Week 5 telemetry / safety thresholds ---
MAX_POST_FAULT_COMPLETION_SIM_STEPS = 10_000
MAX_POST_FAULT_COMPLETION_WALL_SECONDS = 3_600.0
WORKSPACE_X_BOUNDS = (0.15, 0.95)
WORKSPACE_Y_BOUNDS = (-0.45, 0.55)
WORKSPACE_Z_BOUNDS = (0.55, 1.25)
CONTACT_FORCE_THRESHOLD = 3000.0  # Newtons; above this counts as violent collision (normal grip forces are 2000-2600N)
PLACE_ZONE_RADIUS = 0.18  # blocks dropped farther than this from place_zone are OOW violations
