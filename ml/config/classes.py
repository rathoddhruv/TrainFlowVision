from ml.config.paths import ML_DIR

CLASS_FILE = ML_DIR / "class_names.txt"

if not CLASS_FILE.exists():
    CLASS_FILE.write_text("default_class\n", encoding="utf-8")

with CLASS_FILE.open("r", encoding="utf-8") as f:
    CLASS_NAMES = [line.strip() for line in f if line.strip()]

CLASS_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}
CLASS_MAP_REVERSE = {idx: name for name, idx in CLASS_MAP.items()}
