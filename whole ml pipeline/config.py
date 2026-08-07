"""
General configuration definitions not specific to environment variables.
"""
from pathlib import Path

# Project root resolution
PROJECT_ROOT = Path(__file__).resolve().parent

# Directory paths
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
WEIGHTS_DIR = MODELS_DIR / "weights"

# YOLO Class mapping (12 HVAC Classes)
YOLO_CLASSES = {
    0: "Supply_Diffuser",
    1: "Return_Register",
    2: "Exhaust_Grille",
    3: "VAV_Box",
    4: "FCU",
    5: "AHU",
    6: "Fire_Damper",
    7: "Volume_Damper",
    8: "Thermostat",
    9: "Sensor",
    10: "Flex_Duct",
    11: "Rigid_Duct"
}

# Embedding Model configs
EMBEDDING_DIM = 512
