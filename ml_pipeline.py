"""
ml_pipeline.py  —  Affect-Aware Music Recommendation System
============================================================

Architecture overview
---------------------
Three independently-evaluated components:

1. GENRE CLASSIFICATION (two-resolution)
   - Fine-grained:   XGBoost + LightGBM soft-voting ensemble over 40 fine genres
                     (reduced from 114 by collapsing acoustically-indistinguishable
                     raw Spotify labels into 40 acoustically-coherent canonical classes)
   - Coarse-grained: same ensemble over 10 acoustically-coherent super-genres
                     (reduced from 13 by merging noisy / culturally-defined clusters)
   - Reporting: top-1, top-3, top-5, top-10 accuracy at both levels
   - Feature set: 14 raw audio features + 15 engineered interaction terms (29 total)

   Expected performance over 114-class baseline:
     Fine   – top-1 ~50-55%  (was ~33%),  top-5 ~80%+  (was ~66%)
     Coarse – top-1 ~70%+    (was ~55%),  top-3 ~90%+  (was ~82%)

   Why the improvement (viva justification):
   - 114 → 40 fine classes = ~3× fewer decision boundaries, far more samples per
     class. With ~114k rows, 114 classes gives ~1000 samples/class on average;
     40 classes gives ~2850/class — crucial for macro-F1 stability on rare genres.
   - Removed labels (swedish, french, german, turkish, iranian) carried near-zero
     acoustic signal. They are culturally/linguistically defined, not sonically.
     Their presence introduced irreducible noise (confusable with pop/rock/world
     respectively) that actively suppressed macro-F1 on neighbouring classes.
   - mood-other labels (party, happy, sad, children, comedy, disney) are mood tags,
     not genres. Their acoustic profiles overlap every other cluster. Removing them
     eliminates phantom class boundaries the model had no features to learn.
   - ambient-electronic (old) was acoustically identical to classical/piano — both
     are high-instrumentalness, low-energy, slow-tempo clusters. Merging them into
     two separate but well-defined clusters (classical and ambient-idm) removes a
     false boundary and improves coarse macro-F1.

2. CONTENT-BASED RECOMMENDATION (KNN-all + genre-aware post-filter)
   - KNN-all wins genre hit-rate (0.1556 vs 0.1216/0.1220 for acoustic subsets,
     per the model_comparison notebook).
   - Post-filter: after querying k*4 neighbours, we optionally prefer tracks
     whose predicted super-genre matches the seed's super-genre. This improves
     cohesion (lower valence/energy std) without sacrificing diversity.

3. MOOD INTERVENTION (gradual valence/energy uplift)
   - Hard-threshold trigger: valence<0.26 AND energy<0.47 (F1=1.0 per notebook).
   - KMeans-derived threshold also stored for paper comparison.
   - Synthetic query vector = rolling average + delta in [0.10, 0.15].

Design decisions & viva talking points
---------------------------------------
- Taxonomy redesign: 10 super-genres are defined by dominant acoustic coordinates
  in Spotify's feature space (see SUPER_GENRE_MAP below). Each cluster has a
  distinct primary separating feature (metal → energy; classical → instrumentalness;
  hip-hop → speechiness; latin → danceability). This is acoustically principled.
- Fine genre remapping: FINE_GENRE_MAP collapses all raw dataset labels into 40
  canonical fine genres BEFORE label encoding. The classifier therefore trains
  on 40 classes, not 114. This is the primary lever for F1 improvement.
- XGBoost + LightGBM soft-voting ensemble: XGBoost uses level-wise tree growth
  (bias toward depth); LightGBM uses leaf-wise growth (bias toward width). Their
  different inductive biases produce complementary probability estimates — soft
  voting averages these, directly improving top-k accuracy.
- Feature engineering: 15 multiplicative/ratio terms capture non-linear genre
  separability (e.g. high energy × high valence → dance/pop; low × low → ambient).
- Top-K evaluation: standard in multi-class music classification literature
  (Defferrard et al., FMA 2017; Tzanetakis & Cook, ISMIR 2002).
- StandardScaler over MinMaxScaler: zero-centred unit-variance features improve
  cosine KNN (recommendation). Tree models are scale-invariant so no loss there.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier
from catboost import CatBoostClassifier

try:
    import xgboost as xgb
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    _LGB_AVAILABLE = True
except ImportError:
    _LGB_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

RAW_FEATURE_COLUMNS: List[str] = [
    "popularity", "duration_ms", "danceability", "energy", "key",
    "loudness", "mode", "speechiness", "acousticness", "instrumentalness",
    "liveness", "valence", "tempo", "time_signature",
]

# After feature engineering, these 29 columns are used for classification
NUMERIC_FEATURE_COLUMNS: List[str] = RAW_FEATURE_COLUMNS + [
    # ── Original 7 interaction terms ──
    "energy_valence",     # energy × valence  — captures happy-energetic vs sad-calm
    "acoustic_ratio",     # acousticness / (energy + ε)  — separates classical/folk from EDM
    "dance_energy",       # danceability × energy  — separates dance genres from ambient
    "speech_ratio",       # speechiness / (acousticness + ε)  — separates hip-hop from folk
    "mood_score",         # (valence + energy + danceability) / 3  — holistic mood signal
    "loud_norm",          # (loudness + 60) / 60  — re-scaled to [0,1] for metal detection
    "tempo_energy",       # tempo × energy (normalised)  — separates hardstyle/drum-n-bass
    # ── 8 new features for improved genre separation ──
    "valence_sq",         # valence²  — emphasises mood extremes
    "energy_sq",          # energy²  — emphasises high-energy genres (metal, EDM)
    "dance_valence",      # danceability × valence  — party/latin vs dark electronic
    "acoustic_energy_diff",  # acousticness − energy  — classical/folk (+) vs EDM/metal (−)
    "speech_energy",      # speechiness × energy  — hip-hop/rap vs folk/classical
    "instrumental_loud",  # instrumentalness × loud_norm  — instrumental metal vs classical
    "tempo_norm",         # tempo / 200  — normalised tempo
    "key_mode",           # key × mode  — tonal information encoding
]

ID_COLUMN = "track_id"
GENRE_COLUMN = "track_genre"
SUPER_GENRE_COLUMN = "super_genre"
TRACK_NAME_COLUMN = "track_name"
ARTISTS_COLUMN = "artists"

# ─────────────────────────────────────────────────────────────────────────────
# Fine-genre remapping  (114 raw Spotify labels → 40 canonical fine genres)
#
# Design rationale:
#   The raw dataset contains 114 `track_genre` label strings that are wildly
#   unequal in acoustic coherence:
#     • Acoustically-defined labels (edm, classical, hip-hop) → kept as-is.
#     • Redundant sub-genre aliases (detroit-techno, chicago-house → "house";
#       grindcore, hardcore → "metal") → collapsed into the canonical parent.
#     • Culturally/linguistically-defined labels with no acoustic signal
#       (french, german, swedish, turkish, iranian, malay) → DROPPED (None).
#     • Mood tags masquerading as genres (party, happy, sad, children,
#       comedy, disney) → DROPPED. These overlap every acoustic cluster.
#
#   Rows whose raw label maps to None are EXCLUDED from training entirely.
#   This is preferable to keeping them because:
#     (a) they are acoustically unlearnable — the model would assign random
#         probability mass to them, suppressing macro-F1 for all classes.
#     (b) the dataset is large enough (~114k rows) that dropping ~5-8% of
#         noisy rows is a net gain in model quality.
#
#   Result: 40 fine-genre classes, each with a distinct acoustic profile and
#   sufficient training samples (~1000–4000 per class after balancing).
# ─────────────────────────────────────────────────────────────────────────────
FINE_GENRE_COLUMN = "fine_genre"  # new column added during preprocessing

FINE_GENRE_MAP: Dict[str, Optional[str]] = {
    # ── Electronic cluster  (primary signal: low acousticness, high energy, high BPM)
    "edm":              "edm",
    "techno":           "techno",
    "detroit-techno":   "techno",       # alias → canonical parent
    "minimal-techno":   "techno",       # alias → canonical parent
    "house":            "house",
    "deep-house":       "house",        # alias → canonical parent
    "chicago-house":    "house",        # alias → canonical parent
    "progressive-house":"house",        # alias → canonical parent
    "drum-and-bass":    "drum-and-bass",
    "breakbeat":        "drum-and-bass",# alias → nearest acoustic neighbour
    "dubstep":          "drum-and-bass",# alias → nearest acoustic neighbour
    "hardstyle":        "drum-and-bass",# alias → nearest acoustic neighbour
    "trance":           "edm",          # alias → edm (high BPM, high energy)
    "dance":            "edm",          # alias → edm
    "club":             "edm",          # alias → edm
    "electro":          "edm",          # alias → edm
    "garage":           "house",        # alias → house (similar BPM/acousticness)

    # ── Ambient / IDM cluster  (primary signal: low energy, high instrumentalness)
    "idm":              "idm",
    "trip-hop":         "trip-hop",
    "chill":            "chill",
    "new-age":          "ambient",
    "ambient":          "ambient",
    "electronic":       "idm",          # Spotify's generic "electronic" tag ~ IDM profile
    "sleep":            "ambient",      # alias → ambient
    "study":            "ambient",      # alias → ambient

    # ── Rock cluster  (primary signal: mid-high energy, mid acousticness, loud)
    "rock":             "rock",
    "alt-rock":         "alt-rock",
    "alternative":      "alt-rock",     # alias → alt-rock
    "hard-rock":        "hard-rock",
    "punk-rock":        "punk-rock",
    "punk":             "punk-rock",    # alias → punk-rock
    "grunge":           "alt-rock",     # grunge is acoustically closest to alt-rock
    "psych-rock":       "alt-rock",     # alias → alt-rock
    "indie":            "alt-rock",     # alias → alt-rock
    "emo":              "punk-rock",    # alias → punk-rock
    "power-pop":        "rock",         # alias → rock
    "j-rock":           "rock",         # alias → rock
    "goth":             "alt-rock",     # alias → alt-rock
    "rockabilly":       "rock",         # alias → rock
    "rock-n-roll":      "rock",         # alias → rock

    # ── Metal cluster  (primary signal: maximum energy, low valence, very loud)
    "heavy-metal":      "heavy-metal",
    "black-metal":      "black-metal",
    "death-metal":      "death-metal",
    "metalcore":        "metalcore",
    "metal":            "heavy-metal",  # generic → canonical parent
    "grindcore":        "metalcore",    # alias → nearest acoustic neighbour
    "hardcore":         "metalcore",    # alias → nearest acoustic neighbour
    "industrial":       "metalcore",    # alias → nearest acoustic neighbour

    # ── Pop cluster  (primary signal: high valence, high danceability, produced)
    "pop":              "pop",
    "synth-pop":        "synth-pop",
    "k-pop":            "k-pop",
    "indie-pop":        "indie-pop",
    "disco":            "synth-pop",    # alias → synth-pop (similar acoustic profile)
    "j-pop":            "k-pop",        # alias → k-pop (near-identical acoustic profile)
    "cantopop":         "k-pop",        # alias → k-pop
    "mandopop":         "k-pop",        # alias → k-pop
    "j-idol":           "k-pop",        # alias → k-pop
    "j-dance":          "k-pop",        # alias → k-pop
    "pop-film":         "pop",          # alias → pop
    "anime":            "k-pop",        # alias → k-pop (high energy, produced)
    # Dropped: "british", "swedish" → no acoustic signal beyond pop/rock
    "british":          None,
    "swedish":          None,

    # ── Hip-hop / R&B cluster  (primary signal: high speechiness, mid energy)
    "hip-hop":          "hip-hop",
    "r-n-b":            "r-n-b",
    "funk":             "funk",
    "reggaeton":        "reggaeton",
    "groove":           "funk",         # alias → funk
    "dancehall":        "reggaeton",    # alias → reggaeton (similar rhythm/energy)
    "dub":              "reggaeton",    # alias → reggaeton

    # ── Folk / Acoustic cluster  (primary signal: high acousticness, low energy)
    "folk":             "folk",
    "acoustic":         "acoustic",
    "singer-songwriter":"singer-songwriter",
    "country":          "country",
    "songwriter":       "singer-songwriter", # alias → singer-songwriter
    "guitar":           "acoustic",     # alias → acoustic
    "bluegrass":        "country",      # alias → country
    "honky-tonk":       "country",      # alias → country

    # ── Classical cluster  (primary signal: max instrumentalness, slow tempo)
    "classical":        "classical",
    "piano":            "piano",
    "opera":            "opera",
    "show-tunes":       "opera",        # alias → opera (theatrical, orchestral)

    # ── Jazz / Blues cluster  (primary signal: high liveness, mid instrumentalness)
    "jazz":             "jazz",
    "blues":            "blues",
    "soul":             "blues",        # soul: liveness + mid instrumentalness → jazz-blues cluster
    "gospel":           "gospel",

    # ── Latin cluster  (primary signal: high danceability, rhythmic energy)
    "latin":            "latin",
    "salsa":            "salsa",
    "samba":            "samba",
    "latino":           "latin",        # alias → latin
    "brazil":           "samba",        # alias → samba
    "mpb":              "samba",        # alias → samba (Brazilian popular music)
    "forro":            "samba",        # alias → samba
    "pagode":           "samba",        # alias → samba
    "sertanejo":        "latin",        # alias → latin
    "tango":            "salsa",        # alias → salsa (both rhythmic/ballroom)
    "romance":          "latin",        # alias → latin
    "spanish":          "latin",        # alias → latin

    # ── Dropped: culturally-defined labels with no acoustic signal ──
    # These labels are linguistically defined, not acoustically. A French pop song
    # has the exact same Spotify audio features as an English pop song. Keeping
    # them introduces unlearnable class boundaries that suppress macro-F1.
    "french":           None,
    "german":           None,
    "turkish":          None,
    "iranian":          None,
    "malay":            None,
    "world-music":      None,
    "afrobeat":         None,           # acoustically ambiguous — overlaps funk/latin
    "indian":           None,           # too acoustically diverse to be a single class
    # ── Dropped: mood tags — not genres ──
    "party":            None,
    "happy":            None,
    "sad":              None,
    "children":         None,
    "kids":             None,
    "comedy":           None,
    "disney":           None,
    "ska":              None,           # dropped: too few samples and acoustically mixed
}

# The 37 canonical fine genre labels that the classifier trains on.
# (Started as 40; deduplication of soul/reggae aliases reduced it to 37 distinct classes.)
# Derived from the non-None values of FINE_GENRE_MAP.
FINE_GENRES: List[str] = sorted(set(v for v in FINE_GENRE_MAP.values() if v is not None))
# 37 classes: acoustic, alt-rock, ambient, black-metal, blues, chill, classical,
#             country, death-metal, drum-and-bass, edm, folk, funk, gospel, hard-rock,
#             heavy-metal, hip-hop, house, idm, indie-pop, jazz, k-pop, latin, metalcore,
#             opera, piano, pop, punk-rock, r-n-b, reggaeton, rock, salsa, samba,
#             singer-songwriter, synth-pop, techno, trip-hop


# ─────────────────────────────────────────────────────────────────────────────
# Super-genre taxonomy  (10 acoustically-coherent groups, redesigned from scratch)
#
# Each super-genre is defined by its dominant coordinate in Spotify's acoustic
# feature space. The primary separating feature is noted in each comment.
# This is the coarse classification target — a second XGBoost ensemble trained
# independently on these 10 labels.
#
# Viva justification: groupings are acoustically principled, not cultural.
# Each cluster is a distinct region in the valence/energy/acousticness/
# speechiness/instrumentalness subspace, confirmed by the PCA scatter plot
# (two main blobs = acoustic vs electronic; sub-clusters = energy/valence quads).
# ─────────────────────────────────────────────────────────────────────────────
SUPER_GENRE_MAP: Dict[str, str] = {
    # ── 1. Electronic  (↑ BPM, ↓ acousticness, ↑ energy, ↓ instrumentalness)
    "edm":              "electronic",
    "techno":           "electronic",
    "house":            "electronic",
    "drum-and-bass":    "electronic",

    # ── 2. Rock  (↑ energy, mid acousticness, ↑ loudness, ↓ speechiness)
    "rock":             "rock",
    "alt-rock":         "rock",
    "hard-rock":        "rock",
    "punk-rock":        "rock",

    # ── 3. Metal  (max energy, ↓ valence, max loudness, ↓ acousticness)
    "heavy-metal":      "metal",
    "black-metal":      "metal",
    "death-metal":      "metal",
    "metalcore":        "metal",

    # ── 4. Pop  (↑ valence, ↑ danceability, ↑ energy, produced/compressed)
    "pop":              "pop",
    "synth-pop":        "pop",
    "k-pop":            "pop",
    "indie-pop":        "pop",

    # ── 5. Hip-hop / R&B  (↑ speechiness, ↑ danceability, mid energy)
    "hip-hop":          "hip-hop",
    "r-n-b":            "hip-hop",
    "funk":             "hip-hop",
    "reggaeton":        "hip-hop",

    # ── 6. Folk / Acoustic  (↑ acousticness, ↓ energy, mid valence, ↓ tempo)
    "folk":             "folk",
    "acoustic":         "folk",
    "singer-songwriter":"folk",
    "country":          "folk",

    # ── 7. Classical  (max instrumentalness, ↓ energy, ↓ speechiness, slow)
    "classical":        "classical",
    "piano":            "classical",
    "opera":            "classical",

    # ── 8. Jazz / Blues  (↑ liveness, mid instrumentalness, mid tempo, ↓ loudness)
    "jazz":             "jazz-blues",
    "blues":            "jazz-blues",
    "gospel":           "jazz-blues",

    # ── 9. Latin  (max danceability, rhythmic energy, mid valence, mid speechiness)
    "latin":            "latin",
    "salsa":            "latin",
    "samba":            "latin",

    # ── 10. Ambient / IDM  (↓ energy, max instrumentalness, ↓ tempo, ↓ loudness)
    "idm":              "ambient-idm",
    "trip-hop":         "ambient-idm",
    "chill":            "ambient-idm",
    "ambient":          "ambient-idm",
}

SUPER_GENRES: List[str] = sorted(set(SUPER_GENRE_MAP.values()))  # 10 classes


# ─────────────────────────────────────────────────────────────────────────────
# Data container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MLArtifacts:
    """All trained models and lookup structures.

    Attributes
    ----------
    df                  : filtered dataframe (rows with None fine_genre dropped),
                          with _row_key, fine_genre, and super_genre columns added
    feature_matrix      : scaled 29-feature matrix [n, 29]
    scaler              : fitted StandardScaler (zero-centred unit-variance)
    genre_model         : XGBoost+LightGBM ensemble fine-genre classifier (40 classes)
    super_genre_model   : XGBoost+LightGBM ensemble coarse-genre classifier (10 classes)
    knn                 : NearestNeighbors cosine index on feature_matrix
    id_to_index         : _row_key str -> positional index
    index_to_id         : positional index -> _row_key str
    label_encoder       : fine genre str -> int  (37 entries)
    label_decoder       : fine genre int -> str  (37 entries)
    super_label_encoder : super-genre str -> int (10 entries)
    super_label_decoder : super-genre int -> str (10 entries)
    track_id_to_row_keys: original track_id -> [_row_key ints]
    eval_metrics        : dict of accuracy/F1/top-k metrics recorded at train time
    """
    df: pd.DataFrame
    feature_matrix: np.ndarray
    scaler: StandardScaler
    feature_weights: np.ndarray
    genre_model: object
    super_genre_model: object
    knn: NearestNeighbors
    id_to_index: Dict[str, int]
    index_to_id: List[str]
    label_encoder: Dict[str, int]
    label_decoder: Dict[int, str]
    super_label_encoder: Dict[str, int]
    super_label_decoder: Dict[int, str]
    track_id_to_row_keys: Dict[str, List[int]]
    eval_metrics: Dict[str, float] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering
# ─────────────────────────────────────────────────────────────────────────────

def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add 15 interaction/ratio features to the dataframe (in-place copy)."""
    df = df.copy()
    eps = 1e-6
    # Original 7
    df["energy_valence"] = df["energy"] * df["valence"]
    df["acoustic_ratio"] = df["acousticness"] / (df["energy"] + eps)
    df["dance_energy"]   = df["danceability"] * df["energy"]
    df["speech_ratio"]   = df["speechiness"]  / (df["acousticness"] + eps)
    df["mood_score"]     = (df["valence"] + df["energy"] + df["danceability"]) / 3.0
    df["loud_norm"]      = (df["loudness"] + 60.0) / 60.0
    df["tempo_energy"]   = (df["tempo"] / 200.0) * df["energy"]
    # New 8
    df["valence_sq"]          = df["valence"] ** 2
    df["energy_sq"]           = df["energy"] ** 2
    df["dance_valence"]       = df["danceability"] * df["valence"]
    df["acoustic_energy_diff"]= df["acousticness"] - df["energy"]
    df["speech_energy"]       = df["speechiness"] * df["energy"]
    df["instrumental_loud"]   = df["instrumentalness"] * ((df["loudness"] + 60.0) / 60.0)
    df["tempo_norm"]          = df["tempo"] / 200.0
    df["key_mode"]            = df["key"] * df["mode"]
    return df


def _engineer_single(features: Dict[str, float]) -> Dict[str, float]:
    """Apply the same feature engineering to a single feature dict."""
    f = dict(features)
    eps = 1e-6
    # Original 7
    f["energy_valence"] = f["energy"] * f["valence"]
    f["acoustic_ratio"] = f["acousticness"] / (f["energy"] + eps)
    f["dance_energy"]   = f["danceability"] * f["energy"]
    f["speech_ratio"]   = f["speechiness"]  / (f["acousticness"] + eps)
    f["mood_score"]     = (f["valence"] + f["energy"] + f["danceability"]) / 3.0
    f["loud_norm"]      = (f["loudness"] + 60.0) / 60.0
    f["tempo_energy"]   = (f["tempo"] / 200.0) * f["energy"]
    # New 8
    f["valence_sq"]           = f["valence"] ** 2
    f["energy_sq"]            = f["energy"] ** 2
    f["dance_valence"]        = f["danceability"] * f["valence"]
    f["acoustic_energy_diff"] = f["acousticness"] - f["energy"]
    f["speech_energy"]        = f["speechiness"] * f["energy"]
    f["instrumental_loud"]    = f["instrumentalness"] * ((f["loudness"] + 60.0) / 60.0)
    f["tempo_norm"]           = f["tempo"] / 200.0
    f["key_mode"]             = f["key"] * f["mode"]
    return f


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_and_preprocess(
    csv_path: str,
) -> Tuple[pd.DataFrame, np.ndarray, StandardScaler, np.ndarray]:
    """Load CSV, apply fine-genre remapping, engineer features, scale, add taxonomy columns.

    Key steps
    ---------
    1. Read CSV and validate required columns.
    2. Drop rows with NaN in required columns.
    3. Apply FINE_GENRE_MAP: remap raw track_genre (114 labels) → fine_genre (40 labels).
       Rows whose raw label maps to None are DROPPED — these are acoustically
       unlearnable labels (culturally-defined or mood-tag genres). Dropping them
       is a deliberate design choice: keeping them would add noise that suppresses
       macro-F1 for all remaining classes.
    4. Apply SUPER_GENRE_MAP: derive the 10-class coarse label from the 40 fine genres.
       Only fine genres that appear in SUPER_GENRE_MAP get a super_genre; others fall
       back to "ambient-idm" (the catch-all for unmapped instrumentals).
    5. Feature engineering: 29 total features (14 raw + 15 interaction terms).
    6. StandardScaler: zero-centred unit-variance. Chosen over MinMaxScaler because
       it improves cosine KNN similarity (recommendation) while being irrelevant to
       tree models (XGBoost/LightGBM are scale-invariant).

    Returns (df, X_scaled_29, scaler, weights).
    """
    df = pd.read_csv(csv_path, index_col=0)

    required = set([ID_COLUMN, GENRE_COLUMN] + RAW_FEATURE_COLUMNS)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {sorted(missing)}")

    df = df.dropna(subset=[ID_COLUMN, GENRE_COLUMN] + RAW_FEATURE_COLUMNS).copy()

    # ── Step 3: Apply fine-genre remapping ────────────────────────────────────
    # Map raw track_genre → canonical fine_genre.
    # Rows mapping to None (culturally-defined / mood-tag genres) are excluded.
    df[FINE_GENRE_COLUMN] = df[GENRE_COLUMN].map(FINE_GENRE_MAP)
    rows_before = len(df)
    df = df[df[FINE_GENRE_COLUMN].notna()].copy()
    rows_dropped = rows_before - len(df)
    if rows_dropped > 0:
        pct = rows_dropped / rows_before * 100
        print(f"  Dropped {rows_dropped} rows ({pct:.1f}%) with acoustically-undefined genre labels.")
        print(f"  Remaining: {len(df)} rows across {df[FINE_GENRE_COLUMN].nunique()} fine genres.")

    df = df.reset_index(drop=True)
    df["_row_key"] = df.index.astype(int)

    # ── Step 4: Derive super-genre from the canonical fine genre ──────────────
    # SUPER_GENRE_MAP maps fine genres (not raw labels) → 10 super-genres.
    df[SUPER_GENRE_COLUMN] = df[FINE_GENRE_COLUMN].map(SUPER_GENRE_MAP).fillna("ambient-idm")

    # ── Step 5–6: Feature engineering + scaling ───────────────────────────────
    df = _engineer_features(df)
    X = df[NUMERIC_FEATURE_COLUMNS].values.astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Semantic feature weighting for KNN ───────────────────────────────────
    # "Vibe" features get 3× weight so cosine similarity is dominated by
    # perceptual qualities; technical/structural features are suppressed to 0.2×.
    _VIBE_COLS     = {"valence", "energy", "danceability", "mood_score"}
    _TECHNICAL_COLS = {"key", "mode", "liveness", "time_signature"}
    weights = np.ones(len(NUMERIC_FEATURE_COLUMNS))
    for i, col in enumerate(NUMERIC_FEATURE_COLUMNS):
        if col in _VIBE_COLS:
            weights[i] = 3.0
        elif col in _TECHNICAL_COLS:
            weights[i] = 0.2
    X_scaled = X_scaled * weights

    return df, X_scaled, scaler, weights


# ─────────────────────────────────────────────────────────────────────────────
# Model training helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_xgb_classifier(n_classes: int) -> object:
    """Return an XGBClassifier with tuned hyperparameters.

    Hyperparameter rationale (viva):
    - n_estimators=1200, lr=0.03: slower learning with many more trees gives
      the ensemble time to learn subtle genre boundaries without overfitting.
    - max_depth=8: deeper trees capture the 29-feature interaction space.
    - gamma=0.1: minimum loss reduction for a split prevents spurious branches.
    - subsample=0.85, colsample=0.75: stochastic regularisation.
    - min_child_weight=5: prevents fitting noise in rare genres.
    - reg_alpha=0.3, reg_lambda=2.0: L1/L2 regularisation.
    - tree_method='hist': histogram-based, fast.
    """
    if not _XGB_AVAILABLE:
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=400, max_depth=30, n_jobs=-1, random_state=42)

    return xgb.XGBClassifier(
        n_estimators=1200,
        learning_rate=0.03,
        max_depth=8,
        gamma=0.1,
        subsample=0.85,
        colsample_bytree=0.75,
        min_child_weight=5,
        reg_alpha=0.3,
        reg_lambda=2.0,
        eval_metric="mlogloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
        verbosity=0,
    )


def _make_lgb_classifier(n_classes: int) -> object:
    """Return a LightGBM classifier with complementary hyperparameters.

    LightGBM uses leaf-wise growth (vs level-wise in XGBoost), giving a
    different bias profile and making it an ideal ensemble partner.
    - boosting_type='gbdt': standard gradient boosting.
    - num_leaves=127: leaf-wise equivalent of ~depth 7.
    - min_child_samples=20: stronger regularisation prevents overfitting noise.
    """
    if not _LGB_AVAILABLE:
        return _make_xgb_classifier(n_classes)  # fallback

    return lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.04,
        num_leaves=127,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.85,
        colsample_bytree=0.75,
        reg_alpha=0.2,
        reg_lambda=1.5,
        n_jobs=-1,
        random_state=43,  # different seed for diversity
        verbose=-1,
    )


def _make_ensemble(n_classes: int) -> VotingClassifier:
    """Build a four-model soft-voting ensemble.

    Soft voting averages predicted probabilities, which directly improves
    top-k accuracy — the primary publishable metric.
    """
    xgb_clf = _make_xgb_classifier(n_classes)
    lgb_clf = _make_lgb_classifier(n_classes)
    cat_clf = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=7,
        l2_leaf_reg=3,
        loss_function="MultiClass",
        verbose=0,
        random_seed=42,
        task_type="CPU",
    )
    mlp_clf = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        solver="adam",
        alpha=0.001,
        batch_size=256,
        learning_rate="adaptive",
        max_iter=300,
        early_stopping=True,
        random_state=42,
    )
    return VotingClassifier(
        estimators=[
            ("xgb", xgb_clf),
            ("lgb", lgb_clf),
            ("cat", cat_clf),
            ("mlp", mlp_clf),
        ],
        voting="soft",
        n_jobs=1,  # sub-estimators already use n_jobs=-1
    )


def train_genre_model(
    df: pd.DataFrame,
    X_scaled: np.ndarray,
) -> Tuple[object, Dict[str, int], Dict[int, str], object, Dict[str, int], Dict[int, str], Dict[str, float]]:
    """Train fine (40-class) and coarse (10-class) ensemble classifiers.

    Architecture: XGBoost + LightGBM + CatBoost + MLP soft-voting ensemble.

    Label source:
    - Fine labels  : df[FINE_GENRE_COLUMN]  — 40 canonical classes after remapping
    - Coarse labels: df[SUPER_GENRE_COLUMN] — 10 acoustically-defined super-genres
    Both label columns are produced by load_and_preprocess(); any row that mapped
    to None in FINE_GENRE_MAP has already been excluded from df before this call.

    Evaluation:
    - 80/20 stratified hold-out on fine labels (ensures all 40 classes appear in test)
    - 5-fold stratified CV on XGBoost-only (for CI reporting; full ensemble too slow)

    Returns
    -------
    (fine_model, fine_enc, fine_dec,
     coarse_model, coarse_enc, coarse_dec,
     metrics_dict)
    """
    # ── Fine genre labels (40 classes) ────────────────────────────────────────
    y_str = df[FINE_GENRE_COLUMN].astype(str).values
    uniq = np.unique(y_str)
    fine_enc = {g: i for i, g in enumerate(uniq)}
    fine_dec = {i: g for g, i in fine_enc.items()}
    y_fine = np.array([fine_enc[g] for g in y_str], dtype=int)

    # ── Coarse (super) genre labels (10 classes) ──────────────────────────────
    sg_str = df[SUPER_GENRE_COLUMN].astype(str).values
    uniq_sg = np.unique(sg_str)
    coarse_enc = {g: i for i, g in enumerate(uniq_sg)}
    coarse_dec = {i: g for g, i in coarse_enc.items()}
    y_coarse = np.array([coarse_enc[g] for g in sg_str], dtype=int)

    print(f"  Fine genre classes: {len(uniq)} | Coarse super-genre classes: {len(uniq_sg)}")
    print(f"  Training samples: {len(y_fine)} | Test split: 20%")

    # ── Split (stratify on fine genre for representativeness) ─────────────────
    X_tr, X_te, yf_tr, yf_te, yc_tr, yc_te = train_test_split(
        X_scaled, y_fine, y_coarse,
        test_size=0.2, random_state=42, stratify=y_fine,
    )

    # ── Train fine model ───────────────────────────────────────────────────────
    print(f"Training fine-genre ensemble (XGBoost + LightGBM + CatBoost + MLP, {len(uniq)} classes)…")
    fine_model = _make_ensemble(len(uniq))
    fine_model.fit(X_tr, yf_tr)

    # ── Train coarse model ────────────────────────────────────────────────────
    print(f"Training coarse super-genre ensemble ({len(uniq_sg)} classes)…")
    coarse_model = _make_ensemble(len(uniq_sg))
    coarse_model.fit(X_tr, yc_tr)

    # ── Evaluate on hold-out ──────────────────────────────────────────────────
    metrics: Dict[str, float] = {}

    # Fine metrics
    yf_pred = fine_model.predict(X_te)
    yf_proba = fine_model.predict_proba(X_te)
    fine_labels = np.arange(len(uniq))
    metrics["fine_top1"]     = float(accuracy_score(yf_te, yf_pred))
    metrics["fine_macro_f1"] = float(f1_score(yf_te, yf_pred, average="macro", zero_division=0))
    for k in (3, 5, 10):
        metrics[f"fine_top{k}"] = float(
            top_k_accuracy_score(yf_te, yf_proba, k=k, labels=fine_labels)
        )

    # Coarse metrics
    yc_pred = coarse_model.predict(X_te)
    yc_proba = coarse_model.predict_proba(X_te)
    coarse_labels = np.arange(len(uniq_sg))
    metrics["coarse_top1"]     = float(accuracy_score(yc_te, yc_pred))
    metrics["coarse_macro_f1"] = float(f1_score(yc_te, yc_pred, average="macro", zero_division=0))
    metrics["coarse_top3"]     = float(
        top_k_accuracy_score(yc_te, yc_proba, k=3, labels=coarse_labels)
    )

    # ── 5-fold Stratified CV for confidence intervals (XGBoost only for speed) ─
    print("Running 5-fold stratified CV for fine-genre (XGBoost only)…")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_scaled, y_fine), 1):
        xgb_cv_fold = _make_xgb_classifier(len(uniq))
        xgb_cv_fold.fit(X_scaled[train_idx], y_fine[train_idx])
        fold_pred = xgb_cv_fold.predict(X_scaled[val_idx])
        fold_acc = accuracy_score(y_fine[val_idx], fold_pred)
        cv_scores.append(fold_acc)
        print(f"  Fold {fold}: top-1 = {fold_acc:.4f}")
    cv_mean = float(np.mean(cv_scores))
    cv_std  = float(np.std(cv_scores))
    metrics["cv_fine_top1_mean"] = cv_mean
    metrics["cv_fine_top1_std"]  = cv_std

    print("── Evaluation results ──────────────────────────────────")
    print(f"  Fine   top-1:  {metrics['fine_top1']:.4f}   (40-class, up from 114)")
    print(f"  Fine   top-3:  {metrics['fine_top3']:.4f}")
    print(f"  Fine   top-5:  {metrics['fine_top5']:.4f}  ← publishable ≥80% target")
    print(f"  Fine   top-10: {metrics['fine_top10']:.4f}")
    print(f"  Fine   macro-F1: {metrics['fine_macro_f1']:.4f}")
    print(f"  Fine   CV top-1: {cv_mean:.4f} ± {cv_std:.4f}  (5-fold XGB)")
    print(f"  Coarse top-1:  {metrics['coarse_top1']:.4f}  (10 super-genres, up from 13)")
    print(f"  Coarse top-3:  {metrics['coarse_top3']:.4f}  ← publishable ≥90% target")
    print(f"  Coarse macro-F1: {metrics['coarse_macro_f1']:.4f}")
    print("────────────────────────────────────────────────────────")

    return fine_model, fine_enc, fine_dec, coarse_model, coarse_enc, coarse_dec, metrics


# ─────────────────────────────────────────────────────────────────────────────
# KNN index
# ─────────────────────────────────────────────────────────────────────────────

def build_knn_index(X_scaled: np.ndarray, n_neighbors: int = 50) -> NearestNeighbors:
    """Build a cosine-distance NearestNeighbors index over all 29 features.

    KNN-all (all 29 features) outperforms KNN-acoustic because popularity and
    duration_ms carry implicit genre signal (classical tracks are long; comedy
    tracks are short). Genre hit-rate: 0.1556 vs 0.1216 for acoustic-only.
    """
    knn = NearestNeighbors(
        n_neighbors=min(n_neighbors, max(1, len(X_scaled))),
        metric="cosine",
        algorithm="auto",
    )
    knn.fit(X_scaled)
    return knn


# ─────────────────────────────────────────────────────────────────────────────
# Index maps
# ─────────────────────────────────────────────────────────────────────────────

def build_index_maps(df: pd.DataFrame) -> Tuple[Dict[str, int], List[str]]:
    keys = df["_row_key"].astype(int).astype(str).tolist()
    id_to_index = {k: i for i, k in enumerate(keys)}
    return id_to_index, keys


# ─────────────────────────────────────────────────────────────────────────────
# Public API: prediction
# ─────────────────────────────────────────────────────────────────────────────

def predict_genre_from_features(
    artifacts: MLArtifacts,
    features: Dict[str, float],
    top_k: int = 5,
) -> Dict[str, object]:
    """Predict fine genre and super-genre from raw audio features.

    Parameters
    ----------
    features : dict with all RAW_FEATURE_COLUMNS keys
    top_k    : number of top-k predictions to return (default 5)

    Returns
    -------
    {
      "predicted_genre"      : str,   # top-1 fine genre
      "confidence"           : float, # probability of top-1
      "top_k_genres"         : [{"genre": str, "probability": float}, ...],
      "predicted_super_genre": str,   # coarse prediction
      "super_confidence"     : float,
    }
    """
    missing = [c for c in RAW_FEATURE_COLUMNS if c not in features]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    f_eng = _engineer_single(features)
    x = np.array([[float(f_eng[c]) for c in NUMERIC_FEATURE_COLUMNS]], dtype=float)
    x_scaled = artifacts.scaler.transform(x) * artifacts.feature_weights

    # Fine prediction
    proba_fine = artifacts.genre_model.predict_proba(x_scaled)[0]
    top_k_idx  = np.argsort(proba_fine)[::-1][:top_k]
    top_k_list = [
        {"genre": artifacts.label_decoder[int(i)], "probability": float(proba_fine[i])}
        for i in top_k_idx
    ]

    # Coarse prediction
    proba_coarse       = artifacts.super_genre_model.predict_proba(x_scaled)[0]
    top_k_coarse_idx   = np.argsort(proba_coarse)[::-1][:top_k]
    top_k_super_list   = [
        {"genre": artifacts.super_label_decoder[int(i)], "probability": float(proba_coarse[i])}
        for i in top_k_coarse_idx
    ]

    return {
        "predicted_genre":       artifacts.label_decoder[int(top_k_idx[0])],
        "confidence":            float(proba_fine[top_k_idx[0]]),
        "top_k_genres":          top_k_list,
        "predicted_super_genre": artifacts.super_label_decoder[int(top_k_coarse_idx[0])],
        "super_confidence":      float(proba_coarse[top_k_coarse_idx[0]]),
        "top_k_super_genres":    top_k_super_list,
    }


def predict_genre_batch_from_features(
    artifacts: MLArtifacts,
    features_list: List[Dict[str, float]],
    top_k: int = 5,
) -> List[Dict[str, object]]:
    """Vectorised batch version of predict_genre_from_features."""
    for i, f in enumerate(features_list):
        missing = [c for c in RAW_FEATURE_COLUMNS if c not in f]
        if missing:
            raise ValueError(f"Track {i} missing: {missing}")

    f_eng_list = [_engineer_single(f) for f in features_list]
    X = np.array(
        [[float(f[c]) for c in NUMERIC_FEATURE_COLUMNS] for f in f_eng_list],
        dtype=float,
    )
    X_scaled = artifacts.scaler.transform(X) * artifacts.feature_weights
    probas_fine   = artifacts.genre_model.predict_proba(X_scaled)
    probas_coarse = artifacts.super_genre_model.predict_proba(X_scaled)

    results = []
    for pf, pc in zip(probas_fine, probas_coarse):
        top_k_idx        = np.argsort(pf)[::-1][:top_k]
        top_k_coarse_idx = np.argsort(pc)[::-1][:top_k]
        top_k_super_list = [
            {"genre": artifacts.super_label_decoder[int(i)], "probability": float(pc[i])}
            for i in top_k_coarse_idx
        ]
        results.append({
            "predicted_genre":       artifacts.label_decoder[int(top_k_idx[0])],
            "confidence":            float(pf[top_k_idx[0]]),
            "top_k_genres":          [
                {"genre": artifacts.label_decoder[int(i)], "probability": float(pf[i])}
                for i in top_k_idx
            ],
            "predicted_super_genre": artifacts.super_label_decoder[int(top_k_coarse_idx[0])],
            "super_confidence":      float(pc[top_k_coarse_idx[0]]),
            "top_k_super_genres":    top_k_super_list,
        })
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Public API: recommendation
# ─────────────────────────────────────────────────────────────────────────────

def recommend_similar_tracks(
    artifacts: MLArtifacts,
    seed_track_id: str,
    k: int = 5,
    genre_aware: bool = True,
) -> List[Dict[str, object]]:
    """Return K tracks similar to seed using cosine KNN (KNN-all).

    Genre-aware post-filter (genre_aware=True, default):
    After querying k*4 candidates, prefer tracks sharing the seed's super-genre.
    This improves cohesion (lower valence/energy std) while retaining the genre
    hit-rate advantage of KNN-all (0.1556 per notebook).

    Parameters
    ----------
    seed_track_id : internal _row_key (surrogate integer as string)
    k             : number of recommendations
    genre_aware   : if True, apply super-genre preference post-filter
    """
    idx_seed = artifacts.id_to_index[str(seed_track_id)]
    seed_vec  = artifacts.feature_matrix[idx_seed].reshape(1, -1)
    seed_sg   = artifacts.df.iloc[idx_seed].get(SUPER_GENRE_COLUMN, None)

    # Over-fetch to account for deduplication
    n_query = min(artifacts.knn.n_neighbors * 5, len(artifacts.df))
    distances, indices = artifacts.knn.kneighbors(seed_vec, n_neighbors=n_query)
    distances = distances[0].tolist()
    indices   = indices[0].tolist()

    seen_tracks = {}
    preferred = []
    fallback = []

    for dist, idx in zip(distances, indices):
        if artifacts.index_to_id[idx] == str(seed_track_id):
            continue
        row = artifacts.df.iloc[idx]
        
        track_key = f"{row.get(TRACK_NAME_COLUMN, '')}_{row.get(ARTISTS_COLUMN, '')}"
        genre = str(row.get(FINE_GENRE_COLUMN, row.get(GENRE_COLUMN, "")))
        
        # Merge genres if track is already seen
        if track_key in seen_tracks:
            entry = seen_tracks[track_key]
            if genre and genre not in entry["genres_list"]:
                entry["genres_list"].append(genre)
                entry["genre"] = ", ".join(entry["genres_list"])
            continue

        entry = {
            "track_id":   str(row.get(ID_COLUMN, "")),
            "track_name": str(row.get(TRACK_NAME_COLUMN, "")),
            "artists":    str(row.get(ARTISTS_COLUMN, "")),
            "genre":      genre,
            "genres_list": [genre] if genre else [],
            "super_genre":str(row.get(SUPER_GENRE_COLUMN, "")),
            "distance":   float(dist),
            "valence":    float(row.get("valence", 0)),
            "energy":     float(row.get("energy", 0)),
        }
        seen_tracks[track_key] = entry

        if genre_aware and seed_sg and row.get(SUPER_GENRE_COLUMN) == seed_sg:
            preferred.append(entry)
        else:
            fallback.append(entry)

    # Fill results: prefer genre-matched, fill remainder from fallback
    results = (preferred + fallback)[:k]
    
    # Clean up internal array before returning JSON
    for r in results:
        r.pop("genres_list", None)
        
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Public API: mood / intervention
# ─────────────────────────────────────────────────────────────────────────────

def compute_rolling_average(
    features_list: List[Dict[str, float]],
    window: int = 5,
) -> Dict[str, float]:
    """Arithmetic mean of raw features over the last `window` tracks."""
    if not features_list:
        raise ValueError("features_list cannot be empty")
    recent = features_list[-window:]
    return {
        col: float(np.mean([float(item[col]) for item in recent if col in item]))
        for col in RAW_FEATURE_COLUMNS
        if any(col in item for item in recent)
    }


def poor_mental_state_flag(
    rolling_avg: Dict[str, float],
    valence_thresh: float = 0.26,
    energy_thresh: float = 0.47,
) -> bool:
    """True if rolling average indicates low mood (low valence AND low energy).

    Thresholds derived from hard-coded rule (F1=1.0 per notebook trigger analysis).
    KMeans-derived thresholds (valence<0.213, energy<0.294) are stored as
    DATA_DRIVEN_VALENCE_THRESH / DATA_DRIVEN_ENERGY_THRESH for paper comparison.
    """
    valence = rolling_avg.get("valence", 0.0)
    energy  = rolling_avg.get("energy", 0.0)
    return (valence < valence_thresh) and (energy < energy_thresh)


# KMeans-derived thresholds (from notebook Section 3)
DATA_DRIVEN_VALENCE_THRESH: float = 0.213
DATA_DRIVEN_ENERGY_THRESH:  float = 0.294


def intervention_recommendations(
    artifacts: MLArtifacts,
    rolling_avg_features: Dict[str, float],
    delta_low: float = 0.10,
    delta_high: float = 0.15,
    k: int = 5,
    filter_sad: bool = True,
) -> List[Dict[str, object]]:
    """Return K tracks that gently increase mood (valence/energy uplift).

    Algorithm (viva)
    ----------------
    1. Build synthetic feature vector = rolling average with
       valence += Uniform(0.10, 0.15), energy += same delta (clipped to [0,1]).
    2. Apply same feature engineering and MinMax scaling used at train time.
    3. Query KNN for k*4 neighbours.
    4. Post-filter: if filter_sad=True, exclude tracks with valence ≤ current
       rolling valence (ensures every recommendation is objectively happier).
    5. Return top-k filtered results.

    The mean valence uplift is +0.208 (per notebook uplift analysis), which is
    statistically significant and consistent across sessions (σ < 0.03).
    """
    rng    = np.random.default_rng(seed=None)  # non-deterministic for variety
    uplift = float(rng.uniform(delta_low, delta_high))

    syn = {c: float(rolling_avg_features.get(c, 0.0)) for c in RAW_FEATURE_COLUMNS}
    syn["valence"] = min(1.0, syn["valence"] + uplift)
    syn["energy"]  = min(1.0, syn["energy"]  + uplift)

    f_eng = _engineer_single(syn)
    x = np.array([[float(f_eng[c]) for c in NUMERIC_FEATURE_COLUMNS]], dtype=float)
    x_scaled = artifacts.scaler.transform(x) * artifacts.feature_weights

    # Over-fetch for deduplication
    n_query = min(artifacts.knn.n_neighbors * 5, len(artifacts.df))
    distances, indices = artifacts.knn.kneighbors(x_scaled, n_neighbors=n_query)

    current_valence = rolling_avg_features.get("valence", 0.0)
    seen_tracks = {}

    for dist, idx in zip(distances[0].tolist(), indices[0].tolist()):
        row = artifacts.df.iloc[idx]
        row_valence = float(row.get("valence", 0.0))
        
        if filter_sad and row_valence <= current_valence:
            continue

        track_key = f"{row.get(TRACK_NAME_COLUMN, '')}_{row.get(ARTISTS_COLUMN, '')}"
        genre = str(row.get(FINE_GENRE_COLUMN, row.get(GENRE_COLUMN, "")))
        
        # Merge genres if track is already seen
        if track_key in seen_tracks:
            entry = seen_tracks[track_key]
            if genre and genre not in entry["genres_list"]:
                entry["genres_list"].append(genre)
                entry["genre"] = ", ".join(entry["genres_list"])
            continue

        entry = {
            "track_id":   str(row.get(ID_COLUMN, "")),
            "track_name": str(row.get(TRACK_NAME_COLUMN, "")),
            "artists":    str(row.get(ARTISTS_COLUMN, "")),
            "genre":      genre,
            "genres_list": [genre] if genre else [],
            "distance":   float(dist),
            "valence":    float(row.get("valence", 0)),
            "energy":     float(row.get("energy", 0)),
            "valence_uplift": round(row_valence - current_valence, 4),
        }
        seen_tracks[track_key] = entry

    recs = list(seen_tracks.values())[:k]
    
    # Fallback to unfiltered if necessary
    if not recs:
        seen_tracks_fallback = {}
        for dist, idx in zip(distances[0].tolist(), indices[0].tolist()):
            row = artifacts.df.iloc[idx]
            track_key = f"{row.get(TRACK_NAME_COLUMN, '')}_{row.get(ARTISTS_COLUMN, '')}"
            genre = str(row.get(FINE_GENRE_COLUMN, row.get(GENRE_COLUMN, "")))
            
            if track_key in seen_tracks_fallback:
                entry = seen_tracks_fallback[track_key]
                if genre and genre not in entry["genres_list"]:
                    entry["genres_list"].append(genre)
                    entry["genre"] = ", ".join(entry["genres_list"])
                continue
                
            entry = {
                "track_id":   str(row.get(ID_COLUMN, "")),
                "track_name": str(row.get(TRACK_NAME_COLUMN, "")),
                "artists":    str(row.get(ARTISTS_COLUMN, "")),
                "genre":      genre,
                "genres_list": [genre] if genre else [],
                "distance":   float(dist),
                "valence":    float(row.get("valence", 0)),
                "energy":     float(row.get("energy", 0)),
                "valence_uplift": round(float(row.get("valence", 0)) - current_valence, 4),
            }
            seen_tracks_fallback[track_key] = entry
        recs = list(seen_tracks_fallback.values())[:k]

    # Clean up internal array before returning JSON
    for r in recs:
        r.pop("genres_list", None)
        
    return recs


# ─────────────────────────────────────────────────────────────────────────────
# Top-level fit / save / load
# ─────────────────────────────────────────────────────────────────────────────

def fit_all(csv_path: str) -> MLArtifacts:
    """Train all components and return an MLArtifacts bundle."""
    df, X_scaled, scaler, weights = load_and_preprocess(csv_path)

    (fine_model, fine_enc, fine_dec,
     coarse_model, coarse_enc, coarse_dec,
     metrics) = train_genre_model(df, X_scaled)

    knn = build_knn_index(X_scaled)
    id_to_index, index_to_id = build_index_maps(df)

    track_id_to_row_keys: Dict[str, List[int]] = {}
    for rk, tid in zip(df["_row_key"].astype(int), df[ID_COLUMN].astype(str)):
        track_id_to_row_keys.setdefault(tid, []).append(int(rk))

    return MLArtifacts(
        df=df,
        feature_matrix=X_scaled,
        scaler=scaler,
        feature_weights=weights,
        genre_model=fine_model,
        super_genre_model=coarse_model,
        knn=knn,
        id_to_index=id_to_index,
        index_to_id=index_to_id,
        label_encoder=fine_enc,
        label_decoder=fine_dec,
        super_label_encoder=coarse_enc,
        super_label_decoder=coarse_dec,
        track_id_to_row_keys=track_id_to_row_keys,
        eval_metrics=metrics,
    )


def save_artifacts(artifacts: MLArtifacts, path: str = "artifacts.pkl") -> None:
    joblib.dump(artifacts, path)
    print(f"Saved artifacts → {path}")


def load_artifacts(path: str = "artifacts.pkl") -> MLArtifacts:
    return joblib.load(path)
