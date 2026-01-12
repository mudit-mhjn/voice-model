import json
import numpy as np
import joblib
from pathlib import Path

class ModelBundle:
    def __init__(self, model_path: Path):
        self.model = joblib.load(model_path)

        # Try to infer class labels
        if hasattr(self.model, "classes_"):
            self.classes = [str(c) for c in self.model.classes_]
        else:
            # fallback
            self.classes = ["benign", "malignant", "normal"]

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(x)
        # fallback: if only decision function exists
        scores = self.model.decision_function(x)
        scores = np.atleast_2d(scores)
        e = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        return e / np.sum(e, axis=1, keepdims=True)
    
    def combine_segment_probabilities(self, segment_probabilities: list[dict], class_names: list[str]) -> dict:
        """
        combine probabilites by averaging
        """
        if not segment_probabilities:
            return {name: 1.0 / len(class_names) for name in class_names}
        combined = {name: 0.0 for name in class_names}
        for seg_proba in segment_probabilities:
            for name in class_names:
                combined[name] += seg_proba.get(name, 0.0)
        n = len(segment_probabilities)
        return {name: float(combined[name] / n) for name in class_names}

def align_features_to_model_schema(feature_vector: dict, schema_path: Path) -> np.ndarray:
    """
    Ensures feature ordering matches training.
    artifacts/feature_schema.json should be:
      {"feature_names": ["f0_mean", "cpp_proxy", ...]}
    """
    schema = json.loads(schema_path.read_text())
    names = schema["feature_names"]
    x = np.array([[float(feature_vector.get(n, 0.0)) for n in names]], dtype=np.float32)
    return x

def infer_recording(model: ModelBundle, agg_features: dict, per_seg_features: list[dict], schema_path: Path):
    # aggregate prediction
    x = align_features_to_model_schema(agg_features, schema_path)
    proba = model.predict_proba(x)[0]
    idx = int(np.argmax(proba))
    label = model.classes[idx]

    # optional: segment-wise prediction for explainability
    seg_preds = []
    for sf in per_seg_features:
        sx = align_features_to_model_schema(sf, schema_path)
        sp = model.predict_proba(sx)[0]
        seg_preds.append({
            "label": model.classes[int(np.argmax(sp))],
            "proba": {model.classes[i]: float(sp[i]) for i in range(len(model.classes))}
        })

    return label, {model.classes[i]: float(proba[i]) for i in range(len(model.classes))}, seg_preds
