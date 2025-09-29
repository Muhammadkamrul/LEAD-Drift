# FILE: simulate/common.py (REVISED)
import numpy as np
import json
import tensorflow as tf

def write_json(path, obj):
    with open(path, 'w') as f: json.dump(obj, f, indent=2)

def read_json(path):
    with open(path, 'r') as f: return json.load(f)

def calculate_risk_from_model(model, kpi_vectors):
    """Calculates risk by running inference with the trained Keras model."""
    kpi_vectors = np.array(kpi_vectors, dtype='float32')
    if kpi_vectors.ndim == 1:
        kpi_vectors = kpi_vectors.reshape(1, -1)
    risk_scores = model.predict(kpi_vectors, verbose=0)
    return risk_scores.flatten()

def availability(series_kHs):
    tdown = float(np.sum(series_kHs < 100.0))
    T = float(len(series_kHs))
    if T == 0: return 1.0, 0
    return 1.0 - tdown / T, int(tdown)