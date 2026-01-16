from contextlib import asynccontextmanager
import json
import uuid
import os
import boto3
from botocore.utils import ClientError
import requests
import logging
from pathlib import Path
from fastapi import FastAPI, Request, UploadFile, File, Form, Depends, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from numpy import extract
import numpy as np
from sqlmodel import select, Session
from datetime import datetime

from .settings import settings
from .db import init_db, get_session
from .models_db import Patients, SegmentPredictions

from .services.audio_io import read_wav_bytes_to_mono_float32, write_wav
from .services.segmenter import segment_audio, has_voice_activity
from .services.features import extract_features_one_segment, aggregate_segment_features
from .services.model_infer import ModelBundle, infer_recording, align_features_to_model_schema
from .services.report_store import load_json_if_exists

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # on_startup
    global model_bundle
    init_db()
    if not settings.model_path.exists():
        # Let app boot, but inference will error until model is added.
        model_bundle = None
    else:
        model_bundle = ModelBundle(settings.model_path)
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")

model_bundle = None

def upload_audio_via_presigned_url(file_bytes: bytes, key: str, content_type: str) -> str:
    try:
        presigned_url = s3_client.generate_presigned_url(
            "put_object",
            Params={
                "Bucket": S3_BUCKET,
                "Key": key,
                "ContentType": content_type,
            },
            ExpiresIn=3600,
            HttpMethod = "PUT",
        )
    except ClientError as e:
        logger.exception("Failed to generate presigned URL: %s", e)
        raise HTTPException(status_code = 502, detail = "Failed to generate presigned URL")
    try:
        headers = {
            "Content-Type": content_type
        }
        resp = requests.put(presigned_url, data = file_bytes, headers = headers)
        if resp.status_code not in (200, 204):
            logger.error("Failed to upload audio to S3: %s", 
            extra={
                    "status_code": resp.status_code,
                    "response_text": resp.text,
                    "bucket": S3_BUCKET,
                    "key": key,
                })
            raise HTTPException(status_code = 502, detail = "Failed to upload audio to S3")
    except Exception as e:
        logger.exception("Failed to upload audio to S3: %s", e)
        raise HTTPException(status_code = 502, detail = "Failed to upload audio to S3")
    return f"s3://{S3_BUCKET}/{key}"

def parse_s3_uri(s3_uri: str) -> tuple[str, str]:
    if not s3_uri.startswith("s3://"):
        raise ValueError("Invalid S3 URI: %s", s3_uri)
    bucket, key = s3_uri.replace("s3://", "").split("/", 1)
    return bucket, key

S3_BUCKET = "voice-model-uploads"
S3_UPLOAD_PREFIX = "uploads/"
s3_client = boto3.client('s3', region_name = os.getenv("AWS_REGION"))

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.get("/record", response_class=HTMLResponse)
def record_page(request: Request):
    return templates.TemplateResponse("record.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    name: str = Form(...),
    age: int = Form(...),
    gender: str = Form(...),
    audio: UploadFile = File(...),
    session: Session = Depends(get_session),
):
    if model_bundle is None:
        raise HTTPException(status_code=500, detail="Model not found. Put a joblib model file in place.")
    
    allowed_types = [
        "audio/wav", "audio/x-wav", "audio/wave",
        "audio/mpeg", "audio/mp3", "audio/mpeg3"
    ]

    if audio.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Please upload WAV audio (the recorder uses WAV).")

    raw = await audio.read()
    file_ext = None
    if audio.content_type in ["audio/mpeg", "audio/mp3", "audio/mpeg3"]:
        file_ext = ".mp3"
    elif audio.filename and audio.filename.lower().endswith('.mp3'):
        file_ext = ".mp3"
    elif audio.filename and audio.filename.lower().endswith('.wav'):
        file_ext = ".wav"
    else:
        raise HTTPException(status_code=400, detail="Unsupported audio format. Please upload WAV audio.")

    y, sr = read_wav_bytes_to_mono_float32(raw, settings.sample_rate)
    duration_sec = float(len(y) / sr)

    # Segment into 3s windows
    segments = segment_audio(y, sr, settings.segment_seconds, settings.hop_seconds)

    # Find valid segments, and filter out silence
    valid_segments = []
    valid_segment_indices = []
    per_seg_features = []

    for i, seg in enumerate(segments):
        # check for valid voice activity
        if not has_voice_activity(seg, sr):
            continue
        
        try:
            feats = extract_features_one_segment(seg, sr)
            per_seg_features.append(feats)
            valid_segments.append(seg)
            valid_segment_indices.append(i)
        except Exception as e:
            print(f"Error while extracting features for segment {i}: {e}")
            continue
    
    if not valid_segments:
        raise HTTPException(
            status_code = 400,
            detail = "No valid segments found. Please try again with a clearer recording."
        )

    agg_feats = aggregate_segment_features(per_seg_features)

    segment_predictions = []
    segment_probabilities = []

    for segidx, (feats, orig_idx) in enumerate(zip(per_seg_features, valid_segment_indices)):
        x = align_features_to_model_schema(feats, settings.feature_schema_path)
        # get probability and prediction
        proba = model_bundle.predict_proba(x)[0]
        idx = int(np.argmax(proba))
        label = model_bundle.classes[idx]
        proba_dict = {model_bundle.classes[i]: float(proba[i]) for i in range(len(model_bundle.classes))}

        segment_predictions.append({
            "segment_index": orig_idx,
            "label": label,
            "proba": proba_dict,
            "features": feats,
        })
        segment_probabilities.append(proba_dict)
    
    # combine probabilities
    combined_proba = model_bundle.combine_segment_probabilities(segment_probabilities, model_bundle.classes)
    combined_label = max(combined_proba, key = combined_proba.get)

    label_map = {
        "0": "Benign",
        "1": "Malignant",
        "2": "Normal",
    }
    class_name_map = {}
    for key in combined_proba.keys():
        if key in label_map:
            class_name_map[label_map[key]] = combined_proba[key]
        else:
            class_name_map[key.capitalize()] = combined_proba[key]  

    if combined_label in label_map:
        combined_label_display = label_map[combined_label]
    else:
        combined_label_display = combined_label.capitalize()
    
    print(f"DEBUG: combined_label={combined_label}, type={type(combined_label)}")
    print(f"DEBUG: combined_proba keys={list(combined_proba.keys())}")
    print(f"DEBUG: class_name_map={class_name_map}")
    print(f"DEBUG: combined_label_display={combined_label_display}")


    # Save audio
    file_id = uuid.uuid4().hex
    key = f"{S3_UPLOAD_PREFIX}{file_id}{file_ext}"
    audio_path = upload_audio_via_presigned_url(raw, key, audio.content_type)

    # Persist
    rec = Patients(
        name=name.strip(),
        age=int(age),
        gender=gender.strip().lower(),
        audio_path=audio_path,
        duration_sec=duration_sec,
        actual_label=None,    # not known, to be set after medical diagnosis
        predicted_label=combined_label,
        predicted_proba_json=json.dumps(combined_proba),
        segments=[],
        created_at=datetime.now()
    )
    session.add(rec)
    session.commit()
    session.refresh(rec)

    for seg_pred in segment_predictions:
        start_time = seg_pred["segment_index"] * settings.hop_seconds
        end_time = start_time + settings.segment_seconds

        seg_rec = SegmentPredictions(
            patient_id = rec.id,
            segment_index = seg_pred["segment_index"],
            start_time_sec = start_time,
            end_time_sec = end_time,
            predicted_label = seg_pred["label"],
            predicted_proba_json = json.dumps(seg_pred["proba"]),
            features_json = json.dumps(seg_pred["features"])
        )
        session.add(seg_rec)
    session.commit()

    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "rec": rec,
            "predicted_label_display": combined_label_display,
            "proba": class_name_map,
            "features": {"aggregate": agg_feats, "per_segment": per_seg_features},
        },
    )

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request, page: int = 1, session: Session = Depends(get_session)):
    per_page = 10
    offset = (page - 1) * per_page
    total_count = session.exec(select(Patients)).all()
    total_pages = (len(total_count) + per_page - 1) // per_page

    rows = session.exec(
        select(Patients)
        .order_by(Patients.created_at.desc())
        .offset(offset)
        .limit(per_page)
        ).all()
    
    label_map = {
        "0": "Benign",
        "1": "Malignant",
        "2": "Normal",
    }

    mapped_rows = []
    for row in rows:
        predicted_label_display = label_map.get(
            row.predicted_label, 
            row.predicted_label.capitalize() if row.predicted_label else 'N/A'
        )
        if predicted_label_display not in label_map:
            predicted_label_display = predicted_label_display.capitalize()
        actual_display = None
        if row.actual_label:
            actual_display = label_map.get(row.actual_label, row.actual_label.capitalize())
        mapped_rows.append({
            "id": row.id,
            "name": row.name,
            "age": row.age,
            "gender": row.gender,
            "duration_sec": row.duration_sec,
            "predicted_label": predicted_label_display,
            "actual_label": actual_display,
        })
    return templates.TemplateResponse("dashboard.html", 
        {
            "request": request, 
            "rows": mapped_rows, 
            "page": page, 
            "total_pages": total_pages, 
            "total_count": len(total_count),
        },
    )


@app.get("/dashboard/patient/{pid}/audio")
def get_patient_audio(pid: int, session: Session = Depends(get_session)):
    rec = session.get(Patients, pid)
    if not rec:
        raise HTTPException(status_code = 404, detail = "Not found")
    try:
        bucket, key = parse_s3_uri(rec.audio_path)
        url = s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            HttpMethod = "GET"
        )
        return {"url": url}
    except Exception as e:
        logger.exception("Failed to get patient audio: %s", e)
        raise HTTPException(status_code = 502, detail = "Failed to get patient audio")

@app.get("/dashboard/patient/{pid}", response_class=HTMLResponse)
def patient_detail(pid: int, request: Request, session: Session = Depends(get_session)):
    rec = session.get(Patients, pid)
    if not rec:
        raise HTTPException(status_code=404, detail="Not found")

    proba = json.loads(rec.predicted_proba_json)
    label_map = {
        "0": "Benign",
        "1": "Malignant",
        "2": "Normal",
    }

    proba_mapped = {}
    for key in proba.keys():
        if key in label_map:
            proba_mapped[label_map[key]] = proba[key]
        else:
            proba_mapped[key.capitalize()] = proba[key]
    
    predicted_label_display = label_map.get(rec.predicted_label, rec.predicted_label.capitalize())
    if predicted_label_display not in label_map:
        predicted_label_display = predicted_label_display.capitalize()
    
    seg_records = session.exec(
        select(SegmentPredictions)
        .where(SegmentPredictions.patient_id == pid)
        .order_by(SegmentPredictions.segment_index.asc())
    ).all()

    seg_preds = []
    all_features = []
    for seg_rec in seg_records:
        seg_proba = json.loads(seg_rec.predicted_proba_json)
        seg_features = json.loads(seg_rec.features_json)

        seg_proba_mapped = {}
        for key in seg_proba.keys():
            if key in label_map:
                seg_proba_mapped[label_map[key]] = seg_proba[key]
            else:
                seg_proba_mapped[key.capitalize()] = seg_proba[key]
        
        seg_preds.append({
            "segment_index": seg_rec.segment_index,
            "start_time": seg_rec.start_time_sec,
            "end_time": seg_rec.end_time_sec,
            "label": label_map.get(seg_rec.predicted_label, seg_rec.predicted_label.capitalize()),
            "proba": seg_proba_mapped,
        })
        all_features.append(seg_features)
    
    
    if all_features:
        agg_feats = aggregate_segment_features(all_features)
    else:
        agg_feats = {}
    
    return templates.TemplateResponse(
        "patient_details.html",
        {
            "request": request, 
            "rec": rec, 
            "predicted_label_display": predicted_label_display,
            "proba": proba_mapped,
            "seg_preds": seg_preds, 
            "features": {"aggregate": agg_feats},
            "label_map": label_map,
        },
    )

@app.post("/dashboard/patient/{pid}/update-label", response_class=HTMLResponse)
def update_patient_label(
    pid: int,
    request: Request,
    actual_label: str = Form(...),
    session: Session = Depends(get_session)
):
    rec = session.get(Patients, pid)
    if not rec:
        raise HTTPException(status_code=404, detail="Not found")
    
    reverse_label_map = {
        "Benign": "0",
        "Malignant": "1",
        "Normal": "2",
    }
    rec.actual_label = reverse_label_map.get(actual_label, actual_label)
    session.add(rec)
    session.commit()

    return RedirectResponse(url=f"/dashboard/patient/{pid}", status_code = 303)
    


@app.get("/dashboard/model-report", response_class=HTMLResponse)
def model_report(request: Request):
    report = load_json_if_exists(settings.report_path, default={"note": "No report found yet"})
    cm = load_json_if_exists(settings.cm_path, default={"labels": [], "matrix": []})

    # You said: “accuracy around 69%” – we will DISPLAY what is in the report JSON.
    return templates.TemplateResponse(
        "model_report.html",
        {"request": request, "report": report, "cm": cm},
    )
