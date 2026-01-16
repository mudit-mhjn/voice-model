import sqlite3
import logging
import pathlib as Path
import soundfile as sf
import boto3
from datetime import datetime
import librosa
import os

from app.settings import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

session = boto3.Session(
    region_name=os.getenv("AWS_REGION"),
)

S3_BUCKET = "voice-model-uploads"
LABEL_DIRS = {
    "Benign": "benign",
    "Malignant": "malignant",
    "Normal": "normal",
}

def parse_filename(filepath: Path):
    stem = filepath.stem
    parts = stem.split('-')
    if len(parts) != 3:
        logging.warning(f"Skipping {filepath} - invalid filename format")
        return None
    name, gender, age = parts
    return name, gender.lower(), int(age)

def get_duration_seconds(filepath: Path) -> float:
    try:
        return float(librosa.get_duration(path=str(filepath)))
    except Exception:
        info = sf.info(str(filepath))
        return float(info.duration)

def upload_to_s3(s3_client, filepath: Path, key: str) -> str:
    s3_client.upload_file(str(filepath), S3_BUCKET, key)
    return f"s3://{S3_BUCKET}/{key}"

def backfill_audio_data():
    conn = None
    try:
        conn = sqlite3.connect(str(settings.data_dir / 'app.db'))
        cursor = conn.cursor()
        s3_client = session.client('s3')
        
        base_dir = settings.data_dir / 'db_ops'
        logging.info(f"Starting backfill of audio data from {base_dir}")

        for folder_name, label in LABEL_DIRS.items():
            folder_path = base_dir / folder_name
            if not folder_path.exists():
                logging.warning("Folder missing: %s", folder_path)
                continue
            
            try:
                for filepath in folder_path.iterdir():
                    if not filepath.is_file():
                        continue

                    parsed = parse_filename(filepath)
                    if not parsed:
                        continue

                    name, gender, age = parsed
                    duration = get_duration_seconds(filepath)
                    key = f"uploads/{label}/{filepath.name}"
                    audio_path = upload_to_s3(s3_client, filepath, key)

                    cursor.execute("""
                        INSERT INTO patients (name, age, gender, audio_path, duration_sec, actual_label, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (name, age, gender, audio_path, duration, label, datetime.now()))
                    logging.info("Inserted %s (%s)", filepath.name, label)
            except Exception as e:
                logging.exception("Error processing %s", filepath)

        conn.commit()
        logging.info("Backfill completed")
    except Exception as e:
        logging.exception("Backfill failed:", e)
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    backfill_audio_data()






