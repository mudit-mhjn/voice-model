from pydantic import BaseModel
from pathlib import Path

class Settings(BaseModel):
    app_name: str = "voice-model"
    sample_rate: int = 22050
    segment_seconds: float = 3.0
    hop_seconds: float = 1.5

    root: Path = Path(__file__).resolve().parent.parent
    artifacts_dir: Path = root / "artifacts"
    data_dir: Path = root / "data"
    uploads_dir: Path = data_dir / "uploads"

    model_path: Path = artifacts_dir / "xgb_all.joblib"
    feature_schema_path: Path = artifacts_dir / "feature_schema.json"
    report_path: Path = artifacts_dir / "classification_report.json"
    cm_path: Path = artifacts_dir / "confusion_matrix.json"

settings = Settings()
settings.uploads_dir.mkdir(parents = True, exist_ok = True)
settings.data_dir.mkdir(parents = True, exist_ok = True)
