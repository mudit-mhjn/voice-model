from .settings import settings
from sqlmodel import SQLModel, create_engine, Session

from .models_db import Patients, SegmentPredictions


db_path = settings.data_dir / "app.db"
sqlite_url = f"sqlite:///{db_path.as_posix()}"

engine = create_engine(sqlite_url, echo = False)

def init_db():
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session
