from .settings import settings
from sqlmodel import SQLModel, create_engine, Session

from .models_db import Patients, SegmentPredictions

sqllite_url = f"sqlite:///{settings.data_dir / 'app.db'}"
engine = create_engine(sqllite_url, echo = False)

def init_db():
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session
