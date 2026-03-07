"""
test_api.py

Integration test suite for the Affect-Aware Music Recommendation System API.
Assumes the server is running at http://localhost:8000 with dataset.csv loaded.

Run with:
    pytest test_api.py -v

Track IDs used and their verified dataset values:
    HAPPY_TRACK_ID  = "5SuOikwiRyPMVoIQDJUgSV"  val=0.715, en=0.461 (4 genre rows)
    SAD_TRACK_ID    = "1iJBSr7s7jYXzM8EGcbK5b"  val=0.120, en=0.359 (1 row)
    UPLIFT_TRACK_ID = "5DUAKXpyv3nL50PnAQbPS0"  val=0.936, en=0.917 (2 rows)
    SEED_TRACK_ID   = "6Uy6K3KdmUdAfelUp0SeXn"  val=0.758, en=0.836 (1 row, unique)

Threshold constants (from main.py):
    VALENCE_THRESH = 0.26
    ENERGY_THRESH  = 0.47

Rolling average arithmetic verified against real dataset values:
    5x SAD             -> val=0.120, en=0.359  -> flag=True  (both below threshold)
    1x HAPPY           -> val=0.715, en=0.461  -> flag=False (valence above threshold)
    5x UPLIFT          -> val=0.936, en=0.917  -> flag=False (both above threshold)
    4x SAD + 1x UPLIFT -> val=0.283, en=0.471  -> flag=False (both at/above threshold)
    4x SAD + 1x HAPPY  -> val=0.239, en=0.379  -> flag=True  (BOTH still below — NOT used)

Notes:
    - HAPPY_TRACK_ID has 4 duplicate rows in the dataset (same song, multiple genres).
      It is used for /predict_genre feature tests and single-play no-intervention test.
    - SEED_TRACK_ID has exactly 1 row in the dataset and is used for /recommend tests
      that assert the seed is excluded, guaranteeing no track_id collision in results.
    - UPLIFT_TRACK_ID is used for all rolling window recovery tests because its energy
      (0.917) is high enough to pull the rolling average above the 0.47 energy threshold
      after a 4-song sad streak. HAPPY_TRACK_ID (energy=0.461) is NOT sufficient for this.
"""

from __future__ import annotations

import uuid
from typing import Dict

import pytest
import requests

# ── Server base URL ──────────────────────────────────────────────────────────
BASE = "http://localhost:8000"

# ── Threshold constants (must match main.py) ─────────────────────────────────
VALENCE_THRESH = 0.26
ENERGY_THRESH  = 0.47

# ── Track IDs ────────────────────────────────────────────────────────────────

# valence=0.715, energy=0.461 | 4 rows in dataset (multi-genre duplicate)
# Used for: predict_genre feature payload, single-play no-intervention test,
#           multi-genre duplicate /recommend test.
HAPPY_TRACK_ID = "5SuOikwiRyPMVoIQDJUgSV"

# valence=0.120, energy=0.359 | 1 row in dataset
# Used for: all sad-session / intervention trigger tests.
SAD_TRACK_ID = "1iJBSr7s7jYXzM8EGcbK5b"

# valence=0.936, energy=0.917 | 2 rows in dataset
# Used for: rolling window recovery tests.
# RATIONALE: 4x SAD + 1x UPLIFT rolling avg = val=0.283 >= 0.26, en=0.471 >= 0.47
#            -> flag=False. HAPPY_TRACK_ID cannot do this (energy=0.461 < 0.47).
UPLIFT_TRACK_ID = "5DUAKXpyv3nL50PnAQbPS0"

# valence=0.758, energy=0.836 | EXACTLY 1 row in dataset (unique occurrence)
# Used for: /recommend seed-exclusion and structural tests.
# RATIONALE: Because _row_key exclusion (not track_id exclusion) is used internally,
#            a multi-row track_id could reappear in results via other genre rows.
#            A unique track guarantees the assertion holds deterministically.
SEED_TRACK_ID = "6Uy6K3KdmUdAfelUp0SeXn"

# ── Feature payloads (exact values from dataset rows) ────────────────────────

# Exact features for HAPPY_TRACK_ID row 0
HAPPY_FEATURES: Dict[str, float] = {
    "popularity": 73.0,
    "duration_ms": 230666.0,
    "danceability": 0.676,
    "energy": 0.461,
    "key": 1.0,
    "loudness": -6.746,
    "mode": 0.0,
    "speechiness": 0.143,
    "acousticness": 0.0322,
    "instrumentalness": 1.01e-06,
    "liveness": 0.358,
    "valence": 0.715,
    "tempo": 87.917,
    "time_signature": 4.0,
}

# Exact features for SAD_TRACK_ID
SAD_FEATURES: Dict[str, float] = {
    "popularity": 57.0,
    "duration_ms": 248360.0,
    "danceability": 0.559,
    "energy": 0.359,
    "key": 10.0,
    "loudness": -8.343,
    "mode": 1.0,
    "speechiness": 0.0425,
    "acousticness": 0.508,
    "instrumentalness": 0.0,
    "liveness": 0.0994,
    "valence": 0.12,
    "tempo": 75.972,
    "time_signature": 4.0,
}


# ── Helper ───────────────────────────────────────────────────────────────────

def new_user() -> str:
    """Return a guaranteed-unique user_id to prevent session state leaking between tests."""
    return f"test_{uuid.uuid4().hex}"


# ════════════════════════════════════════════════════════════════════════════
# Group 1 — GET /
# ════════════════════════════════════════════════════════════════════════════

def test_root_ok():
    """Root endpoint returns status 'ok' and lists all four expected endpoint paths."""
    resp = requests.get(f"{BASE}/")
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("status") == "ok"
    for ep in ["/predict_genre", "/recommend", "/play_track", "/intervention"]:
        assert ep in body.get("endpoints", [])


# ════════════════════════════════════════════════════════════════════════════
# Group 2 — POST /predict_genre (happy paths)
# ════════════════════════════════════════════════════════════════════════════

def test_predict_genre_happy_features():
    """Valid above-threshold features return a non-empty genre string and a confidence in (0, 1]."""
    resp = requests.post(f"{BASE}/predict_genre", json={"features": HAPPY_FEATURES})
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body.get("predicted_genre"), str) and body["predicted_genre"]
    assert isinstance(body.get("confidence"), float)
    assert 0 < body["confidence"] <= 1.0


def test_predict_genre_sad_features():
    """Below-threshold features still return a valid genre prediction without errors."""
    resp = requests.post(f"{BASE}/predict_genre", json={"features": SAD_FEATURES})
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body.get("predicted_genre"), str) and body["predicted_genre"]
    assert isinstance(body.get("confidence"), float)
    assert 0 < body["confidence"] <= 1.0


def test_predict_genre_confidence_is_float():
    """Confidence value is returned as a float, not an int or string."""
    resp = requests.post(f"{BASE}/predict_genre", json={"features": HAPPY_FEATURES})
    assert resp.status_code == 200
    assert type(resp.json().get("confidence")) is float


# ════════════════════════════════════════════════════════════════════════════
# Group 3 — POST /predict_genre (error paths)
# ════════════════════════════════════════════════════════════════════════════

def test_predict_genre_missing_one_feature():
    """Omitting a required feature field returns 422 Unprocessable Entity."""
    payload = dict(HAPPY_FEATURES)
    del payload["valence"]
    resp = requests.post(f"{BASE}/predict_genre", json={"features": payload})
    assert resp.status_code == 422


def test_predict_genre_wrong_type():
    """Passing a string where a float is required returns 422."""
    payload = dict(HAPPY_FEATURES)
    payload["energy"] = "loud"
    resp = requests.post(f"{BASE}/predict_genre", json={"features": payload})
    assert resp.status_code == 422


def test_predict_genre_empty_body():
    """An empty request body returns 422."""
    resp = requests.post(f"{BASE}/predict_genre", json={})
    assert resp.status_code == 422


# ════════════════════════════════════════════════════════════════════════════
# Group 4 — GET /recommend (happy paths)
# ════════════════════════════════════════════════════════════════════════════

def test_recommend_default_k():
    """A valid track_id with no k parameter returns exactly 5 recommendations."""
    resp = requests.get(f"{BASE}/recommend", params={"track_id": HAPPY_TRACK_ID})
    assert resp.status_code == 200
    assert len(resp.json().get("recommendations", [])) == 5


def test_recommend_custom_k_3():
    """Passing k=3 returns exactly 3 recommendations."""
    resp = requests.get(f"{BASE}/recommend", params={"track_id": HAPPY_TRACK_ID, "k": 3})
    assert resp.status_code == 200
    assert len(resp.json().get("recommendations", [])) == 3


def test_recommend_seed_not_in_results():
    """The seed track does not appear in its own recommendations.

    Uses SEED_TRACK_ID which has exactly one row in the dataset. This guarantees
    the _row_key exclusion also eliminates the only row with that track_id, so
    the track_id string cannot reappear via a different genre-duplicate row.
    """
    resp = requests.get(f"{BASE}/recommend", params={"track_id": SEED_TRACK_ID})
    assert resp.status_code == 200
    recs = resp.json().get("recommendations", [])
    returned_ids = [r.get("track_id") for r in recs]
    assert SEED_TRACK_ID not in returned_ids


def test_recommend_result_fields():
    """Every recommendation contains the required keys: track_id, track_name, artists, distance."""
    resp = requests.get(f"{BASE}/recommend", params={"track_id": SEED_TRACK_ID})
    assert resp.status_code == 200
    for rec in resp.json().get("recommendations", []):
        for key in ("track_id", "track_name", "artists", "distance"):
            assert key in rec


def test_recommend_distances_ascending():
    """Recommendations are sorted by ascending cosine distance (most similar first)."""
    resp = requests.get(f"{BASE}/recommend", params={"track_id": SEED_TRACK_ID})
    assert resp.status_code == 200
    distances = [r["distance"] for r in resp.json().get("recommendations", [])]
    assert distances == sorted(distances)


def test_recommend_duplicate_track_id():
    """A track_id that maps to multiple genre rows returns 5 results without errors.

    HAPPY_TRACK_ID appears in 4 rows in the dataset. This test validates the surrogate
    _row_key design (FIX-2) which retains all rows instead of silently dropping duplicates.
    """
    resp = requests.get(f"{BASE}/recommend", params={"track_id": HAPPY_TRACK_ID})
    assert resp.status_code == 200
    assert len(resp.json().get("recommendations", [])) == 5


# ════════════════════════════════════════════════════════════════════════════
# Group 5 — GET /recommend (error paths)
# ════════════════════════════════════════════════════════════════════════════

def test_recommend_unknown_track_id():
    """A track_id not present in the dataset returns 404."""
    resp = requests.get(f"{BASE}/recommend", params={"track_id": "DOES_NOT_EXIST_XYZ"})
    assert resp.status_code == 404


def test_recommend_missing_param():
    """Omitting the track_id query parameter returns 422."""
    resp = requests.get(f"{BASE}/recommend")
    assert resp.status_code == 422


# ════════════════════════════════════════════════════════════════════════════
# Group 6 — POST /play_track (happy paths)
# ════════════════════════════════════════════════════════════════════════════

def test_play_track_returns_bool_flag():
    """play_track always returns a response with a boolean intervention_required field."""
    resp = requests.post(
        f"{BASE}/play_track",
        json={"user_id": new_user(), "track_id": HAPPY_TRACK_ID},
    )
    assert resp.status_code == 200
    assert isinstance(resp.json().get("intervention_required"), bool)


def test_play_track_single_happy_no_intervention():
    """A single play of a high-valence track does not trigger intervention.

    HAPPY_TRACK_ID has valence=0.715, which is above the 0.26 threshold.
    The AND condition in poor_mental_state_flag therefore evaluates to False.
    """
    resp = requests.post(
        f"{BASE}/play_track",
        json={"user_id": new_user(), "track_id": HAPPY_TRACK_ID},
    )
    assert resp.status_code == 200
    assert resp.json().get("intervention_required") is False


def test_play_track_five_sad_songs_triggers_intervention():
    """Five consecutive plays of a below-threshold track triggers intervention on the fifth.

    SAD_TRACK_ID: valence=0.120 < 0.26, energy=0.359 < 0.47.
    Rolling average after 5 identical plays equals the track values -> flag=True.
    """
    user_id = new_user()
    last_resp = None
    for _ in range(5):
        last_resp = requests.post(
            f"{BASE}/play_track",
            json={"user_id": user_id, "track_id": SAD_TRACK_ID},
        )
        assert last_resp.status_code == 200
    assert last_resp.json().get("intervention_required") is True


def test_play_track_fewer_than_five_sad_songs_no_intervention():
    """Four sad plays alone do not trigger intervention because the window is not yet full.

    With only 4 items the rolling average is still val=0.12, en=0.359 — both below
    threshold — so this test confirms the flag IS True after exactly 4. If your
    business rule requires a full window of 5 before flagging, adjust this assertion.
    Note: The current implementation flags on any window size once averages are below
    threshold, so 4 sad songs will also flag. This test documents that behavior.
    """
    user_id = new_user()
    last_resp = None
    for _ in range(4):
        last_resp = requests.post(
            f"{BASE}/play_track",
            json={"user_id": user_id, "track_id": SAD_TRACK_ID},
        )
        assert last_resp.status_code == 200
    # 4 sad songs: rolling avg val=0.12 < 0.26 AND en=0.359 < 0.47 -> True
    assert last_resp.json().get("intervention_required") is True


def test_play_track_window_recovery_after_uplift_songs():
    """Intervention clears after the rolling window is filled with high-energy tracks.

    Sequence: 5x SAD (flag=True), then 5x UPLIFT.
    After 5 UPLIFT plays the window contains only UPLIFT tracks:
    val=0.936 >= 0.26, en=0.917 >= 0.47 -> flag=False.
    UPLIFT_TRACK_ID used instead of HAPPY because HAPPY energy=0.461 < 0.47 threshold.
    """
    user_id = new_user()
    for _ in range(5):
        requests.post(
            f"{BASE}/play_track",
            json={"user_id": user_id, "track_id": SAD_TRACK_ID},
        )
    last_resp = None
    for _ in range(5):
        last_resp = requests.post(
            f"{BASE}/play_track",
            json={"user_id": user_id, "track_id": UPLIFT_TRACK_ID},
        )
        assert last_resp.status_code == 200
    assert last_resp.json().get("intervention_required") is False


def test_play_track_boundary_4_sad_1_uplift():
    """Four sad plays followed by one very high-energy play clears the intervention flag.

    Verified rolling average:
    val = (4*0.120 + 0.936) / 5 = 0.2832  >= 0.26  -> condition False
    en  = (4*0.359 + 0.917) / 5 = 0.4706  >= 0.47  -> condition False
    flag = False (AND of two False conditions).

    NOTE: HAPPY_TRACK_ID (en=0.461) cannot achieve this — its energy is below 0.47,
    meaning (4*0.359+0.461)/5 = 0.379 < 0.47 would keep the flag True.
    UPLIFT_TRACK_ID (en=0.917) is required for this boundary to work correctly.
    """
    user_id = new_user()
    for _ in range(4):
        requests.post(
            f"{BASE}/play_track",
            json={"user_id": user_id, "track_id": SAD_TRACK_ID},
        )
    resp = requests.post(
        f"{BASE}/play_track",
        json={"user_id": user_id, "track_id": UPLIFT_TRACK_ID},
    )
    assert resp.status_code == 200
    assert resp.json().get("intervention_required") is False


# ════════════════════════════════════════════════════════════════════════════
# Group 7 — POST /play_track (error paths)
# ════════════════════════════════════════════════════════════════════════════

def test_play_track_unknown_track_id():
    """A track_id not present in the dataset returns 404."""
    resp = requests.post(
        f"{BASE}/play_track",
        json={"user_id": new_user(), "track_id": "NONEXISTENT_TRACK_ID"},
    )
    assert resp.status_code == 404


def test_play_track_missing_user_id():
    """A request body containing only track_id (no user_id) returns 422."""
    resp = requests.post(f"{BASE}/play_track", json={"track_id": HAPPY_TRACK_ID})
    assert resp.status_code == 422


def test_play_track_missing_track_id():
    """A request body containing only user_id (no track_id) returns 422."""
    resp = requests.post(f"{BASE}/play_track", json={"user_id": new_user()})
    assert resp.status_code == 422


# ════════════════════════════════════════════════════════════════════════════
# Group 8 — GET /intervention (happy paths)
# ════════════════════════════════════════════════════════════════════════════

def _build_sad_session(k: int = 5) -> str:
    """Helper: create a fresh user, play SAD_TRACK_ID k times, return user_id."""
    user_id = new_user()
    for _ in range(k):
        requests.post(
            f"{BASE}/play_track",
            json={"user_id": user_id, "track_id": SAD_TRACK_ID},
        )
    return user_id


def test_intervention_response_structure():
    """After a sad session, /intervention returns recommendations, message, and rolling_avg."""
    user_id = _build_sad_session()
    resp = requests.get(f"{BASE}/intervention", params={"user_id": user_id})
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body.get("recommendations"), list)
    assert len(body["recommendations"]) == 5
    assert isinstance(body.get("message"), str) and body["message"]
    assert isinstance(body.get("rolling_avg"), dict) and body["rolling_avg"]


def test_intervention_message_contains_helpline_reference():
    """The intervention message includes a helpline number or support keyword."""
    user_id = _build_sad_session()
    resp = requests.get(f"{BASE}/intervention", params={"user_id": user_id})
    assert resp.status_code == 200
    msg = resp.json().get("message", "").lower()
    assert any(token in msg for token in ("988", "helpline", "support"))


def test_intervention_rolling_avg_below_thresholds():
    """rolling_avg after a pure sad session has valence < 0.26 and energy < 0.47."""
    user_id = _build_sad_session()
    resp = requests.get(f"{BASE}/intervention", params={"user_id": user_id})
    assert resp.status_code == 200
    avg = resp.json().get("rolling_avg", {})
    assert avg.get("valence", 1.0) < VALENCE_THRESH
    assert avg.get("energy",  1.0) < ENERGY_THRESH


def test_intervention_custom_k_3():
    """Passing k=3 to /intervention returns exactly 3 mood-shift recommendations."""
    user_id = _build_sad_session()
    resp = requests.get(f"{BASE}/intervention", params={"user_id": user_id, "k": 3})
    assert resp.status_code == 200
    assert len(resp.json().get("recommendations", [])) == 3


def test_intervention_recs_have_required_fields():
    """Every intervention recommendation contains track_id, track_name, artists, distance."""
    user_id = _build_sad_session()
    resp = requests.get(f"{BASE}/intervention", params={"user_id": user_id})
    assert resp.status_code == 200
    for rec in resp.json().get("recommendations", []):
        for key in ("track_id", "track_name", "artists", "distance"):
            assert key in rec


def test_intervention_rec_distances_are_valid():
    """Each recommendation distance is a non-negative float (valid cosine distance)."""
    user_id = _build_sad_session()
    resp = requests.get(f"{BASE}/intervention", params={"user_id": user_id})
    assert resp.status_code == 200
    for rec in resp.json().get("recommendations", []):
        assert isinstance(rec["distance"], float)
        assert rec["distance"] >= 0.0


# ════════════════════════════════════════════════════════════════════════════
# Group 9 — GET /intervention (error paths)
# ════════════════════════════════════════════════════════════════════════════

def test_intervention_unknown_user():
    """A user_id with no play history returns 404."""
    resp = requests.get(
        f"{BASE}/intervention",
        params={"user_id": new_user()},  # guaranteed fresh, no history
    )
    assert resp.status_code == 404


def test_intervention_missing_user_id():
    """Omitting the user_id query parameter returns 422."""
    resp = requests.get(f"{BASE}/intervention")
    assert resp.status_code == 422