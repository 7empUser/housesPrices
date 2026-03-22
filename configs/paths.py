from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
CONFIG_DIR = BASE_DIR / "configs"
TEST_DIR = BASE_DIR / "tests"
ARTIFACTS_DIR = BASE_DIR / "artifacts"