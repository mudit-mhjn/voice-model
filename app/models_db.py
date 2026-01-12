from sqlmodel import Relationship, SQLModel, Field
from datetime import datetime
from typing import Optional

class Patients(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key = True)
    name: str
    age: int
    gender: str  # "male" / "female"
    created_at: datetime = Field(default_factory=datetime.now)
    audio_path: str
    duration_sec: float
    actual_label: Optional[str] = None
    predicted_label: str
    predicted_proba_json: str
    segments: list["SegmentPredictions"] = Relationship(back_populates = "patient")

class SegmentPredictions(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key = True)
    patient_id: int = Field(foreign_key = "patients.id")
    segment_index: int
    start_time_sec: float
    end_time_sec: float
    predicted_label: str
    predicted_proba_json: str
    features_json: str

    patient: Optional["Patients"] = Relationship(back_populates = "segments")
