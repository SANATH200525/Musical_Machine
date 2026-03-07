"""
ml_pipeline.py

End-to-end ML utilities for an Affect-Aware Music Recommendation System.

This module provides:
- Data loading and preprocessing (MinMax normalization for continuous features)
- Supervised genre prediction using RandomForestClassifier
- Content-based recommendation using NearestNeighbors with cosine similarity
- Intervention (Gradual Mood-Shift) recommendation that nudges valence/energy upward

Design choices:
- Scikit-learn pipeline-style separation for scaler and models
- The dataset is expected to be a CSV called `dataset.csv` in the working directory
- For reproducibility and deployment simplicity, models are trained on startup

Mathematical notes:
- MinMaxScaler maps each continuous feature x to (x - min) / (max - min) using the
  empirical min/max from the dataset. This keeps all features in [0, 1] to make
  cosine similarity meaningful across heterogeneous scales.
- NearestNeighbors with metric='cosine' finds points minimizing cosine distance
  d(u, v) = 1 - cos(theta) where cos(theta) = (u·v) / (||u|| ||v||). With normalized
  features, the direction of the feature vector dominates similarity.
- Rolling average for valence and energy is a simple arithmetic mean across the
  last k songs (k=5 in our business rule). Given values v_i, the rolling average
  is (1/k) * sum_{i=1..k} v_i.
- Gradual Mood-Shift: we construct a synthetic feature vector equal to the user's
  rolling average across acoustic features, but with valence and energy nudged by
  +delta (clipped to [0,1]). We then query nearest neighbors in the normalized
  feature space to find gentle positive-shift recommendations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Columns used as numerical audio features for modeling/recommendations.
NUMERIC_FEATURE_COLUMNS = [
    "popularity",
    "duration_ms",
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "time_signature",
]

ID_COLUMN = "track_id"
GENRE_COLUMN = "track_genre"
TRACK_NAME_COLUMN = "track_name"
ARTISTS_COLUMN = "artists"


@dataclass
class MLArtifacts:
    """Container holding preprocessed data and trained models.

    Attributes
    ----------
    df : pd.DataFrame
        Full dataset including metadata and features.
    feature_matrix : np.ndarray
        Scaled numeric features (shape: [n_samples, n_features]).
    scaler : MinMaxScaler
        Fitted scaler used to transform numeric features.
    genre_model : RandomForestClassifier
        Trained classifier to predict track_genre.
    knn : NearestNeighbors
        Trained nearest-neighbor index over feature_matrix.
    id_to_index : Dict[str, int]
        Map from surrogate _row_key (str) to positional row index in
        df/feature_matrix. Using _row_key instead of track_id supports
        rows where the same track_id appears under multiple genres.
    index_to_id : List[str]
        Reverse map from positional index to surrogate _row_key string.
        Values are _row_key strings, not original track_ids.
    label_encoder : Dict[str, int]
        String label to integer class mapping for genres.
    label_decoder : Dict[int, str]
        Integer class to string label mapping for genres.
    track_id_to_row_keys : Dict[str, List[int]]
        Maps each original track_id string to a list of all integer _row_key
        values that share that track_id (handles one-to-many duplicates).
        Built once at startup for O(1) lookup in API endpoints.
    """

    df: pd.DataFrame
    feature_matrix: np.ndarray
    scaler: MinMaxScaler
    genre_model: RandomForestClassifier
    knn: NearestNeighbors
    id_to_index: Dict[str, int]
    index_to_id: List[str]
    label_encoder: Dict[str, int]
    label_decoder: Dict[int, str]
    track_id_to_row_keys: Dict[str, List[int]]


def load_and_preprocess(csv_path: str) -> Tuple[pd.DataFrame, np.ndarray, MinMaxScaler]:
    """Load dataset and apply MinMax scaling to numeric features.

    Parameters
    ----------
    csv_path : str
        Path to the dataset CSV.

    Returns
    -------
    (df, X_scaled, scaler)
        df: the raw dataframe with minimal cleanup (drops rows with NA in needed columns),
        X_scaled: MinMax-scaled numeric feature matrix,
        scaler: fitted MinMaxScaler.

    Notes
    -----
    - Missing numeric values are imputed via dropna for simplicity in prototype.
    - Categorical columns are kept as-is for metadata; we model on numeric features only.
    """
    # [FIX-1] Use index_col=0 so a saved CSV index like "Unnamed: 0" becomes the DataFrame index and does not pollute columns
    df = pd.read_csv(csv_path, index_col=0)

    # Ensure required columns exist
    required_cols = set([ID_COLUMN, GENRE_COLUMN] + NUMERIC_FEATURE_COLUMNS)
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    # Drop rows with NA in numeric or id/genre
    df = df.dropna(subset=[ID_COLUMN, GENRE_COLUMN] + NUMERIC_FEATURE_COLUMNS).copy()

    # [FIX-2] Do NOT drop duplicate track_ids; instead, create a surrogate row key after reset_index
    df = df.reset_index(drop=True)
    # [FIX-2] Introduce a stable surrogate integer key to uniquely reference rows across duplicate track_ids
    # [FIX-2] This avoids discarding valid rows when the same track appears under multiple genres.
    df["_row_key"] = df.index.astype(int)

    X = df[NUMERIC_FEATURE_COLUMNS].values.astype(float)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return df, X_scaled, scaler


def train_genre_model(df: pd.DataFrame, X_scaled: np.ndarray) -> Tuple[RandomForestClassifier, Dict[str, int], Dict[int, str], float]:
    """Train a Random Forest classifier to predict track_genre.

    [FIX-5] Prototype-optimised hyperparameters
    -------------------------------------------
    The classifier hyperparameters are set to balance startup speed and memory
    usage vs. accuracy in this prototype. For production, tune via cross-validation.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame including the target column `track_genre`.
    X_scaled : np.ndarray
        Normalized numeric features.

    Returns
    -------
    (model, label_encoder, label_decoder, val_acc)
        Trained model and mapping dicts, along with a validation accuracy estimate.

    Notes
    -----
    - For speed and robustness, we use a moderately sized RandomForest.
    - Labels are integer-encoded via a simple Python dict (no external encoders needed).
    - We perform a single train/validation split to report a quick accuracy figure.
    """
    y_str = df[GENRE_COLUMN].astype(str).values
    # Build simple encoders
    unique_labels = np.unique(y_str)
    label_encoder: Dict[str, int] = {label: i for i, label in enumerate(unique_labels)}
    label_decoder: Dict[int, str] = {i: label for label, i in label_encoder.items()}
    y = np.array([label_encoder[s] for s in y_str], dtype=int)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y if len(unique_labels) > 1 else None
    )

    # [FIX-5] Reduce estimators, cap depth, and increase min_samples_leaf to speed up startup and reduce memory
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=2,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_tr, y_tr)

    # Quick validation
    y_pred = model.predict(X_val)
    val_acc = accuracy_score(y_val, y_pred) if len(np.unique(y_val)) > 1 else 1.0

    return model, label_encoder, label_decoder, float(val_acc)


def build_knn_index(X_scaled: np.ndarray, n_neighbors: int = 50, metric: str = "cosine") -> NearestNeighbors:
    """Build a NearestNeighbors index over the normalized features.

    Parameters
    ----------
    X_scaled : np.ndarray
        Scaled feature matrix.
    n_neighbors : int
        Upper bound for neighbors to query at once. This can be larger than the
        request-time K to ensure enough candidates.
    metric : str
        Distance metric for neighbor search. Cosine is appropriate for normalized data.
    """
    knn = NearestNeighbors(n_neighbors=min(n_neighbors, max(1, len(X_scaled))), metric=metric, algorithm="auto")
    knn.fit(X_scaled)
    return knn


def build_index_maps(df: pd.DataFrame) -> Tuple[Dict[str, int], List[str]]:
    """Create bi-directional mappings between surrogate _row_key and dataset row index.

    [FIX-2] Surrogate key rationale
    -------------------------------
    The dataset contains many duplicate track_id values (same song under multiple genres).
    Instead of dropping duplicates, we create a surrogate integer key `_row_key` equal to
    the DataFrame's integer index and use it internally for lookups to avoid silent data loss.
    """
    # [FIX-2] Use _row_key as the lookup ID
    keys = df["_row_key"].astype(int).astype(str).tolist()
    id_to_index = {kid: i for i, kid in enumerate(keys)}
    index_to_id = keys
    return id_to_index, index_to_id


def _get_seed_vector_by_track_id(artifacts: MLArtifacts, track_id: str) -> np.ndarray:
    """Return the scaled feature vector for a given track key.

    [FIX-2] Note: `track_id` parameter now refers to the surrogate `_row_key` used internally.
    Raises KeyError if the id is unknown.
    """
    idx = artifacts.id_to_index[str(track_id)]
    return artifacts.feature_matrix[idx]


def recommend_similar_tracks(
    artifacts: MLArtifacts,
    seed_track_id: str,
    k: int = 5,
) -> List[Dict[str, object]]:
    """Recommend K tracks most similar to a seed track using cosine KNN.

    Parameters
    ----------
    artifacts : MLArtifacts
        Trained artifacts including KNN and dataframe.
    seed_track_id : str
        The track id used as the seed for similarity search.
    k : int
        Number of tracks to return.

    Returns
    -------
    List[dict]
        Each dict includes: track_id, track_name, artists, distance (cosine distance).
        [FIX-2] The returned `track_id` here is the original dataset track_id for display; lookups
        internally use the surrogate `_row_key`.

    Notes
    -----
    - The seed track itself is excluded from results.
    - Distances are cosine distances in [0, 2] but in normalized non-negative settings,
      typically [0, 1]. Lower is better (more similar).
    """
    seed_vec = _get_seed_vector_by_track_id(artifacts, seed_track_id).reshape(1, -1)
    n_query = min(artifacts.knn.n_neighbors, len(artifacts.df))
    distances, indices = artifacts.knn.kneighbors(seed_vec, n_neighbors=n_query)
    distances = distances[0].tolist()
    indices = indices[0].tolist()

    results: List[Dict[str, object]] = []
    for dist, idx in zip(distances, indices):
        if artifacts.index_to_id[idx] == str(seed_track_id):
            continue  # exclude the seed itself
        row = artifacts.df.iloc[idx]
        results.append({
            # [FIX-2] Expose original track_id for display; internal routing uses surrogate keys
            "track_id": row.get(ID_COLUMN, None),
            "track_name": row.get(TRACK_NAME_COLUMN, None),
            "artists": row.get(ARTISTS_COLUMN, None),
            "distance": float(dist),
        })
        if len(results) >= k:
            break
    return results


def predict_genre_from_features(
    artifacts: MLArtifacts,
    features: Dict[str, float],
) -> Dict[str, object]:
    """Predict a track genre from raw numeric audio features.

    Parameters
    ----------
    artifacts : MLArtifacts
        Trained artifacts including scaler and classifier.
    features : Dict[str, float]
        Mapping from feature name to numeric value for all NUMERIC_FEATURE_COLUMNS.

    Returns
    -------
    dict
        {"predicted_genre": str, "confidence": float} where confidence is the
        probability of the predicted class under the RF model.

    Implementation details
    ----------------------
    - The function expects all numeric features; missing ones will raise.
    - Features are organized in the canonical column order and transformed with
      the fitted MinMaxScaler before classification.
    """
    missing = [c for c in NUMERIC_FEATURE_COLUMNS if c not in features]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    x = np.array([[float(features[c]) for c in NUMERIC_FEATURE_COLUMNS]], dtype=float)
    x_scaled = artifacts.scaler.transform(x)

    proba = artifacts.genre_model.predict_proba(x_scaled)[0]
    pred_idx = int(np.argmax(proba))
    pred_genre = artifacts.label_decoder[pred_idx]
    confidence = float(proba[pred_idx])

    return {"predicted_genre": pred_genre, "confidence": confidence}


def compute_rolling_average(features_list: List[Dict[str, float]], window: int = 5) -> Dict[str, float]:
    """Compute rolling average across numeric features for the last `window` items.

    Parameters
    ----------
    features_list : List[Dict[str, float]]
        List of feature dictionaries (raw, unscaled), each expected to include the
        keys in NUMERIC_FEATURE_COLUMNS.
    window : int
        The number of most recent items to average.

    Returns
    -------
    Dict[str, float]
        Mapping from feature name to the arithmetic mean across the last `window` elements.

    Mathematical definition
    -----------------------
    Given k=min(n, window) latest items with feature f values f_1..f_k,
    rolling_mean(f) = (1/k) * sum_{i=1..k} f_i.
    """
    if not features_list:
        raise ValueError("features_list cannot be empty")
    recent = features_list[-window:]

    means: Dict[str, float] = {}
    for col in NUMERIC_FEATURE_COLUMNS:
        vals = [float(item[col]) for item in recent if col in item]
        if not vals:
            continue
        means[col] = float(np.mean(vals))
    return means


def poor_mental_state_flag(rolling_avg: Dict[str, float], valence_thresh: float = 0.26, energy_thresh: float = 0.47) -> bool:
    """Return True if rolling averages indicate a poor mental state.

    Business rule
    -------------
    Flag if rolling average valence < 0.26 AND energy < 0.47.
    """
    # [FIX-4] Safe-fail-low: treat missing values as low (0.0) so we err on recommending intervention
    valence = rolling_avg.get("valence", 0.0)
    energy = rolling_avg.get("energy", 0.0)
    return (valence < valence_thresh) and (energy < energy_thresh)


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def intervention_recommendations(
    artifacts: MLArtifacts,
    rolling_avg_features: Dict[str, float],
    delta_low: float = 0.10,
    delta_high: float = 0.15,
    k: int = 5,
) -> List[Dict[str, object]]:
    """Recommend K tracks that gently increase mood (valence/energy) while staying similar.

    Parameters
    ----------
    artifacts : MLArtifacts
        Trained artifacts including scaler and KNN index.
    rolling_avg_features : Dict[str, float]
        Rolling average of recent tracks' raw features.
    delta_low : float
        Lower bound of the mood uplift to apply to valence and energy.
    delta_high : float
        Upper bound of the mood uplift to apply to valence and energy.
    k : int
        Number of tracks to return.

    Algorithm
    ---------
    1. Construct a synthetic feature vector equal to the rolling average for all
       numeric features.
    2. Increase valence and energy by a random delta in [delta_low, delta_high],
       clipping to [0, 1].
    3. Transform this vector using the fitted MinMaxScaler.
    4. Query KNN for nearest neighbors to this adjusted point.

    Rationale
    ---------
    By querying neighbors around a point that is slightly happier/more energetic,
    we maintain acoustic similarity (through the other features) while nudging
    mood upwards.
    """
    rng = np.random.default_rng(42)  # deterministic uplift for reproducibility
    uplift = float(rng.uniform(delta_low, delta_high))

    # Build synthetic vector
    syn = {c: float(rolling_avg_features.get(c, 0.0)) for c in NUMERIC_FEATURE_COLUMNS}
    syn["valence"] = _clip01(syn.get("valence", 0.0) + uplift)
    syn["energy"] = _clip01(syn.get("energy", 0.0) + uplift)

    x = np.array([[syn[c] for c in NUMERIC_FEATURE_COLUMNS]], dtype=float)
    x_scaled = artifacts.scaler.transform(x)

    n_query = min(artifacts.knn.n_neighbors, len(artifacts.df))
    distances, indices = artifacts.knn.kneighbors(x_scaled, n_neighbors=n_query)
    distances = distances[0].tolist()
    indices = indices[0].tolist()

    recs: List[Dict[str, object]] = []
    for dist, idx in zip(distances, indices):
        row = artifacts.df.iloc[idx]
        recs.append({
            # [FIX-2] Expose original track_id for display; internal routing uses surrogate keys
            "track_id": row.get(ID_COLUMN, None),
            "track_name": row.get(TRACK_NAME_COLUMN, None),
            "artists": row.get(ARTISTS_COLUMN, None),
            "distance": float(dist),
        })
        if len(recs) >= k:
            break
    return recs


def fit_all(csv_path: str) -> MLArtifacts:
    """Convenience helper to train all components from a CSV.

    Steps
    -----
    - Load and preprocess (MinMax scaling)
    - Train RandomForest genre classifier
    - Build cosine KNN index for content-based recommendations
    - Build ID<->index maps for quick lookup

    Returns
    -------
    MLArtifacts
        A dataclass bundling the dataset, models, scaler, and lookup maps.
    """
    df, X_scaled, scaler = load_and_preprocess(csv_path)
    genre_model, enc, dec, _ = train_genre_model(df, X_scaled)
    knn = build_knn_index(X_scaled)
    id_to_index, index_to_id = build_index_maps(df)
    # Build O(1) track_id -> list of _row_key ints lookup to avoid per-request O(N) scans
    track_id_to_row_keys: Dict[str, List[int]] = {}
    for rk, tid in zip(df["_row_key"].astype(int), df["track_id"].astype(str)):
        track_id_to_row_keys.setdefault(tid, []).append(int(rk))

    return MLArtifacts(
        df=df,
        feature_matrix=X_scaled,
        scaler=scaler,
        genre_model=genre_model,
        knn=knn,
        id_to_index=id_to_index,
        index_to_id=index_to_id,
        label_encoder=enc,
        label_decoder=dec,
        track_id_to_row_keys=track_id_to_row_keys,
    )
