"""
main.py

FastAPI application exposing endpoints for an Affect-Aware Music Recommendation System.

Endpoints
---------
- POST /predict_genre: Predict track genre from provided audio features
- GET  /recommend: Recommend similar tracks given a seed track_id
- POST /play_track: Record that a user played a track; update rolling stats and report intervention flag
- GET  /intervention: Return gradual mood-shift recommendations and a helpline message

Startup
-------
- On startup, the application loads the dataset and trains models using ml_pipeline.fit_all.

Session tracking
----------------
- In-memory dict maps user_id -> {
    "history": list of dicts with {track_id, features}, bounded to last N entries
    "rolling_avg": cached rolling averages dict
  }
- Rolling average window is 5 (per business rule)

Notes
-----
- This prototype uses in-memory state and trains models at startup for simplicity.
- In production, you would persist trained models and session state to a database or cache.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from ml_pipeline import (
    fit_all,
    NUMERIC_FEATURE_COLUMNS,
    predict_genre_from_features,
    recommend_similar_tracks,
    compute_rolling_average,
    poor_mental_state_flag,
    intervention_recommendations,
    MLArtifacts,
)


DATASET_PATH = os.getenv("DATASET_PATH", "dataset.csv")
ROLLING_WINDOW = 5
VALENCE_THRESH = 0.26
ENERGY_THRESH = 0.47


class FeaturePayload(BaseModel):
    """Pydantic model for audio feature payloads.

    All features are floats. This model enforces presence of the required fields.
    """

    popularity: float
    duration_ms: float
    danceability: float
    energy: float
    key: float
    loudness: float
    mode: float
    speechiness: float
    acousticness: float
    instrumentalness: float
    liveness: float
    valence: float
    tempo: float
    time_signature: float


class PredictRequest(BaseModel):
    features: FeaturePayload


class PlayTrackRequest(BaseModel):
    user_id: str = Field(..., description="Unique identifier for a user session")
    track_id: str = Field(..., description="Track ID as present in the dataset")


app = FastAPI(title="Affect-Aware Music Recommendation System")


# Global state
artifacts: Optional[MLArtifacts] = None
user_sessions: Dict[str, Dict[str, object]] = {}


@app.on_event("startup")
async def startup_event():
    global artifacts
    try:
        artifacts = fit_all(DATASET_PATH)
    except Exception as e:
        # Surface error early to indicate missing or malformed dataset
        raise RuntimeError(f"Failed to initialize ML artifacts from {DATASET_PATH}: {e}")


@app.get("/")
async def root():
    return {
        "status": "ok",
        "message": "Affect-Aware Music Recommendation API",
        "endpoints": [
            "/predict_genre",
            "/recommend",
            "/play_track",
            "/intervention",
        ],
    }


@app.post("/predict_genre")
async def predict_genre(req: PredictRequest):
    if artifacts is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    result = predict_genre_from_features(artifacts, req.features.dict())
    return result


@app.get("/recommend")
async def recommend(track_id: str = Query(..., description="Seed track_id to find similar tracks"), k: int = 5):
    if artifacts is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    try:
        recs = recommend_similar_tracks(artifacts, track_id, k=k)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown track_id: {track_id}")
    return {"recommendations": recs}


@app.post("/play_track")
async def play_track(req: PlayTrackRequest):
    if artifacts is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    # Lookup the track in our dataset and pull raw features
    tid = str(req.track_id)
    if tid not in artifacts.id_to_index:
        raise HTTPException(status_code=404, detail=f"Unknown track_id: {tid}")
    idx = artifacts.id_to_index[tid]

    row = artifacts.df.iloc[idx]
    features = {c: float(row[c]) for c in NUMERIC_FEATURE_COLUMNS}

    # Update session state
    sess = user_sessions.setdefault(req.user_id, {"history": [], "rolling_avg": None})

    # Maintain a fixed-size history (keep last ROLLING_WINDOW items)
    sess["history"].append({"track_id": tid, "features": features})
    if len(sess["history"]) > ROLLING_WINDOW:
        sess["history"] = sess["history"][-ROLLING_WINDOW:]

    # Compute rolling average from session history features
    feature_list = [h["features"] for h in sess["history"]]  # type: ignore[index]
    rolling = compute_rolling_average(feature_list, window=ROLLING_WINDOW)
    sess["rolling_avg"] = rolling

    # Check mental state
    need_intervention = poor_mental_state_flag(rolling, VALENCE_THRESH, ENERGY_THRESH)
    return {"intervention_required": need_intervention}


@app.get("/intervention")
async def intervention(user_id: str = Query(..., description="User ID to compute intervention for"), k: int = 5):
    if artifacts is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    sess = user_sessions.get(user_id)
    if not sess or not sess.get("rolling_avg"):
        raise HTTPException(status_code=404, detail="No session or insufficient history for user")

    rolling_avg = sess["rolling_avg"]  # type: ignore[assignment]
    recs = intervention_recommendations(artifacts, rolling_avg, k=k)

    message = (
        "Take a deep breath. If you need support, please call your local mental health helpline "
        "(e.g., 988 in the U.S., 116 123 in parts of the EU/UK). If you are in immediate danger, call emergency services."
    )

    return {"recommendations": recs, "message": message, "rolling_avg": rolling_avg}
