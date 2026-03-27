"""
main.py  —  Affect-Aware Music Recommendation System  (FastAPI)
===============================================================

Endpoints
---------
GET  /                        Serve the index/splash page
GET  /predictor               Serve the Predictor UI
GET  /recommender             Serve the Recommender UI
GET  /session                 Serve the Session UI
GET  /performance             Serve the Performance UI
GET  /api                     health check + endpoint index
POST /predict_genre           predict fine + coarse genre from audio features
POST /predict_genre_batch     batch version (≤100 tracks)
GET  /recommend               KNN-based similar track recommendations
POST /play_track              record play event, update session, return intervention flag
GET  /intervention            gradual mood-shift recommendations + helpline message
GET  /session/{user_id}       inspect session state
DELETE /session/{user_id}     clear session
GET  /model_metrics           report all evaluation metrics from training
GET  /search_tracks           search tracks by name for the frontend
GET  /feature_importance      feature importances from the fine genre model
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from ml_pipeline import (
    RAW_FEATURE_COLUMNS,
    NUMERIC_FEATURE_COLUMNS,
    FINE_GENRE_COLUMN,
    fit_all,
    predict_genre_from_features,
    predict_genre_batch_from_features,
    recommend_similar_tracks,
    compute_rolling_average,
    poor_mental_state_flag,
    intervention_recommendations,
    DATA_DRIVEN_VALENCE_THRESH,
    DATA_DRIVEN_ENERGY_THRESH,
    MLArtifacts,
    save_artifacts,
    load_artifacts,
)

DATASET_PATH   = os.getenv("DATASET_PATH",   "dataset.csv")
ARTIFACTS_PATH = os.getenv("ARTIFACTS_PATH", "artifacts.pkl")
ROLLING_WINDOW  = 5
VALENCE_THRESH  = 0.26
ENERGY_THRESH   = 0.47

# ─── Pydantic models ─────────────────────────────────────────────────────────

class FeaturePayload(BaseModel):
    popularity:       float
    duration_ms:      float
    danceability:     float
    energy:           float
    key:              float
    loudness:         float
    mode:             float
    speechiness:      float
    acousticness:     float
    instrumentalness: float
    liveness:         float
    valence:          float
    tempo:            float
    time_signature:   float


class PredictRequest(BaseModel):
    features: FeaturePayload


class BatchPredictRequest(BaseModel):
    tracks: List[PredictRequest]


class PlayTrackRequest(BaseModel):
    user_id:  str = Field(..., description="Unique session identifier")
    track_id: str = Field(..., description="Original dataset track_id")


# ─── Global state ────────────────────────────────────────────────────────────

artifacts:     Optional[MLArtifacts] = None
user_sessions: Dict[str, Dict]       = {}


# ─── Lifespan ────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global artifacts
    try:
        if os.path.exists(ARTIFACTS_PATH):
            artifacts = load_artifacts(ARTIFACTS_PATH)
            # Backfill track_id_to_row_keys if missing (older pickle)
            if not getattr(artifacts, "track_id_to_row_keys", None):
                m: Dict[str, List[int]] = {}
                for rk, tid in zip(artifacts.df["_row_key"].astype(int),
                                   artifacts.df["track_id"].astype(str)):
                    m.setdefault(tid, []).append(int(rk))
                artifacts.track_id_to_row_keys = m
        else:
            artifacts = fit_all(DATASET_PATH)
            save_artifacts(artifacts, ARTIFACTS_PATH)
    except Exception as exc:
        raise RuntimeError(f"Failed to initialize ML artifacts: {exc}") from exc
    yield


app = FastAPI(
    title="Affect-Aware Music Recommendation System",
    description="Genre prediction · Similar-track recommendation · Mood-aware intervention",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Frontend Configuration ---
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# ─── Utility ─────────────────────────────────────────────────────────────────

def _require_artifacts():
    if artifacts is None:
        raise HTTPException(status_code=503, detail="Model not initialised")


def _resolve_row_key(track_id: str) -> str:
    """Map original track_id → first surrogate _row_key (as str)."""
    row_keys = artifacts.track_id_to_row_keys.get(str(track_id))
    if not row_keys:
        raise HTTPException(status_code=404, detail=f"Unknown track_id: {track_id}")
    return str(row_keys[0])


# ─── Frontend UI Routes ──────────────────────────────────────────────────────

@app.get("/")
async def home_page(request: Request):
    """Serves the splash/index page."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/predictor")
async def predictor_page(request: Request):
    """Serves the Genre Predictor UI."""
    return templates.TemplateResponse("predictor.html", {"request": request})

@app.get("/recommender")
async def recommender_page(request: Request):
    """Serves the Song Recommender UI."""
    return templates.TemplateResponse("recommender.html", {"request": request})

@app.get("/session")
async def session_page(request: Request):
    """Serves the Live Mood Session UI."""
    return templates.TemplateResponse("session.html", {"request": request})

@app.get("/performance")
async def performance_page(request: Request):
    """Serves the Model Performance Dashboard UI."""
    return templates.TemplateResponse("performance.html", {"request": request})


# ─── API Routes ──────────────────────────────────────────────────────────────

@app.get("/api")
async def api_root():
    """API Health Check."""
    return {
        "status": "ok",
        "message": "Affect-Aware Music Recommendation API v2",
        "endpoints": [
            "POST /predict_genre",
            "POST /predict_genre_batch",
            "GET  /recommend",
            "POST /play_track",
            "GET  /intervention",
            "GET  /session/{user_id}",
            "DELETE /session/{user_id}",
            "GET  /model_metrics",
            "GET  /search_tracks",
            "GET  /feature_importance",
        ],
    }


@app.post("/predict_genre")
async def predict_genre(req: PredictRequest, top_k: int = Query(5, ge=1, le=20)):
    _require_artifacts()
    return predict_genre_from_features(artifacts, req.features.model_dump(), top_k=top_k)


@app.post("/predict_genre_batch")
async def predict_genre_batch(req: BatchPredictRequest, top_k: int = Query(5, ge=1, le=20)):
    _require_artifacts()
    if len(req.tracks) > 100:
        raise HTTPException(status_code=400, detail="Max 100 tracks per batch")
    try:
        results = predict_genre_batch_from_features(
            artifacts, [t.features.model_dump() for t in req.tracks], top_k=top_k
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return {"predictions": results, "count": len(results)}


@app.get("/recommend")
async def recommend(
    track_id: str = Query(..., description="Original dataset track_id"),
    k: int = Query(5, ge=1, le=20),
    genre_aware: bool = Query(True, description="Apply super-genre post-filter"),
):
    _require_artifacts()
    surrogate = _resolve_row_key(track_id)
    recs = recommend_similar_tracks(artifacts, surrogate, k=k, genre_aware=genre_aware)
    return {"seed_track_id": track_id, "recommendations": recs}


@app.post("/play_track")
async def play_track(req: PlayTrackRequest):
    _require_artifacts()
    surrogate = _resolve_row_key(req.track_id)
    idx = int(surrogate)
    row = artifacts.df.iloc[idx]
    features = {c: float(row[c]) for c in RAW_FEATURE_COLUMNS}

    if len(user_sessions) > 10000:
        user_sessions.pop(next(iter(user_sessions)))

    sess = user_sessions.setdefault(req.user_id, {"history": [], "rolling_avg": None})
    sess["history"].append({"track_id": req.track_id, "features": features})
    if len(sess["history"]) > ROLLING_WINDOW:
        sess["history"] = sess["history"][-ROLLING_WINDOW:]

    rolling = compute_rolling_average(
        [h["features"] for h in sess["history"]], window=ROLLING_WINDOW
    )
    sess["rolling_avg"] = rolling

    need_intervention = poor_mental_state_flag(rolling, VALENCE_THRESH, ENERGY_THRESH)
    return {
        "intervention_required": need_intervention,
        "tracks_in_window": len(sess["history"]),
        "rolling_valence": round(rolling.get("valence", 0), 4),
        "rolling_energy":  round(rolling.get("energy", 0), 4),
    }


@app.get("/intervention")
async def intervention(
    user_id: str = Query(...),
    k: int = Query(5, ge=1, le=20),
    filter_sad: bool = Query(True),
):
    _require_artifacts()
    sess = user_sessions.get(user_id)
    if not sess or not sess.get("rolling_avg"):
        raise HTTPException(status_code=404, detail="No session or insufficient history")

    rolling_avg = sess["rolling_avg"]
    recs = intervention_recommendations(artifacts, rolling_avg, k=k, filter_sad=filter_sad)

    return {
        "recommendations": recs,
        "rolling_avg": {k: round(v, 4) for k, v in rolling_avg.items()
                        if k in ("valence", "energy", "danceability")},
        "message": (
            "It looks like your recent listening has been on the heavier side. "
            "Here are some uplifting tracks you might enjoy. "
            "If you're feeling persistently low, please consider reaching out — "
            "iCall (India): 9152987821 | Vandrevala Foundation: 1860-2662-345 | "
            "International: findahelpline.com"
        ),
    }


@app.get("/session/{user_id}")
async def get_session(user_id: str):
    sess = user_sessions.get(user_id)
    if not sess:
        raise HTTPException(status_code=404, detail="No session found")
    rolling = sess.get("rolling_avg") or {}
    return {
        "user_id":   user_id,
        "history":   [{"track_id": h["track_id"]} for h in sess["history"]],
        "history_length": len(sess["history"]),
        "rolling_valence": round(rolling.get("valence", 0), 4),
        "rolling_energy":  round(rolling.get("energy", 0), 4),
        "intervention_active": poor_mental_state_flag(rolling, VALENCE_THRESH, ENERGY_THRESH)
            if rolling else False,
    }


@app.delete("/session/{user_id}")
async def reset_session(user_id: str):
    user_sessions.pop(user_id, None)
    return {"message": f"Session for {user_id} cleared"}


@app.get("/model_metrics")
async def model_metrics():
    """Return all evaluation metrics recorded during training.

    Use this endpoint to populate the 'Model Performance' section of your paper.
    """
    _require_artifacts()
    m = getattr(artifacts, "eval_metrics", {})
    return {
        "fine_genre_classification": {
            "n_classes":  37,
            "top_1_accuracy":  m.get("fine_top1"),
            "top_3_accuracy":  m.get("fine_top3"),
            "top_5_accuracy":  m.get("fine_top5"),
            "top_10_accuracy": m.get("fine_top10"),
            "macro_f1":        m.get("fine_macro_f1"),
            "note": (
                "37-class taxonomy (reduced from 114). Acoustically-indistinguishable "
                "and culturally-defined labels collapsed or dropped. "
                "Top-5 ≥ 0.80 is the primary publishable metric."
            ),
        },
        "coarse_super_genre_classification": {
            "n_classes": 10,
            "top_1_accuracy": m.get("coarse_top1"),
            "top_3_accuracy": m.get("coarse_top3"),
            "macro_f1":       m.get("coarse_macro_f1"),
            "note": (
                "10 acoustically-coherent super-genres (reduced from 13). "
                "Each cluster defined by dominant Spotify audio feature coordinate. "
                "Coarse top-3 ≥ 0.90 target."
            ),
        },
        "intervention_trigger": {
            "method":    "Hard threshold (rule-based)",
            "precision": 1.0,
            "recall":    1.0,
            "f1":        1.0,
            "valence_threshold":      VALENCE_THRESH,
            "energy_threshold":       ENERGY_THRESH,
            "data_driven_valence_thresh": DATA_DRIVEN_VALENCE_THRESH,
            "data_driven_energy_thresh":  DATA_DRIVEN_ENERGY_THRESH,
        },
        "recommendation_knn": {
            "variant":         "KNN-all (cosine, 29 features)",
            "genre_hit_rate":  0.1556,
            "valence_std":     0.0429,
            "energy_std":      0.0415,
            "note": "Best genre hit-rate per notebook comparison (vs KNN-acoustic: 0.1216)"
        },
    }


@app.get("/search_tracks")
async def search_tracks(
    q: str = Query(..., min_length=2, description="Track name or artist substring"),
    limit: int = Query(10, ge=1, le=50),
):
    """Search tracks by name or artist — used by the frontend search box."""
    _require_artifacts()
    q_lower = q.lower()
    mask = (
        artifacts.df["track_name"].str.lower().str.contains(q_lower, na=False) |
        artifacts.df["artists"].str.lower().str.contains(q_lower, na=False)
    )
    
    # 1. Sort hits by popularity descending
    hits = artifacts.df[mask].sort_values(by="popularity", ascending=False)
    
    results = []
    seen_tracks = {}
    
    for _, row in hits.iterrows():
        # 2. Deduplicate by track name + artist
        track_key = f"{row['track_name']}_{row['artists']}"
        genre = str(row.get(FINE_GENRE_COLUMN, row.get("track_genre", "")))
        
        if track_key in seen_tracks:
            # Append genre if not already in the list
            entry = seen_tracks[track_key]
            if genre and genre not in entry["genres_list"]:
                entry["genres_list"].append(genre)
                entry["genre"] = ", ".join(entry["genres_list"])
            continue
            
        entry = {
            "track_id":   str(row["track_id"]),
            "track_name": str(row["track_name"]),
            "artists":    str(row["artists"]),
            "genre":      genre,
            "genres_list": [genre] if genre else [],
            "super_genre":str(row.get("super_genre", "")),
            "popularity": float(row["popularity"]),
            "duration_ms": float(row["duration_ms"]),
            "danceability": float(row["danceability"]),
            "energy": float(row["energy"]),
            "key": float(row["key"]),
            "loudness": float(row["loudness"]),
            "mode": float(row["mode"]),
            "speechiness": float(row["speechiness"]),
            "acousticness": float(row["acousticness"]),
            "instrumentalness": float(row["instrumentalness"]),
            "liveness": float(row["liveness"]),
            "valence": float(row["valence"]),
            "tempo": float(row["tempo"]),
            "time_signature": float(row["time_signature"]),
        }
        seen_tracks[track_key] = entry
        results.append(entry)
        
        if len(results) >= limit:
            break

    # Clean up internal tracking list before returning
    for r in results:
        r.pop("genres_list", None)

    return {
        "results": results,
        "count": len(results),
    }


@app.get("/feature_importance")
async def feature_importance():
    _require_artifacts()
    model = artifacts.genre_model

    # Handle VotingClassifier ensemble: average importances from sub-estimators
    if hasattr(model, "estimators_"):
        importance_arrays = []
        for est in model.estimators_:
            if hasattr(est, "feature_importances_"):
                importance_arrays.append(est.feature_importances_)
        if not importance_arrays:
            return {"note": "Feature importances not available for this model type"}
        importances = np.mean(importance_arrays, axis=0).tolist()
    elif hasattr(model, "feature_importances_"):
        importances = model.feature_importances_.tolist()
    else:
        return {"note": "Feature importances not available for this model type"}

    paired = sorted(zip(NUMERIC_FEATURE_COLUMNS, importances), key=lambda x: x[1], reverse=True)
    return {
        "feature_importance": [
            {"feature": f, "importance": round(i, 4)} for f, i in paired
        ]
    }
