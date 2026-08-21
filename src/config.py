"""Shared paths and constants for the email reply system."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

DEFAULT_MODEL_NAME = "t5-small"
PROCESSED_FILENAME = "email_pairs.csv"
TRAIN_FILENAME = "train.csv"
VAL_FILENAME = "val.csv"
TEST_FILENAME = "test.csv"
MODEL_CHECKPOINT_DIR = MODELS_DIR / "t5-email-final"
MODEL_TRAINING_DIR = MODELS_DIR / "t5-email"
