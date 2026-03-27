# MoodTune: Affect-Aware Music Recommendation System

[cite_start]MoodTune is an end-to-end, audio feature-based music classification and recommendation engine. [cite_start]It analyzes raw acoustic signatures to predict genres and maintains a rolling-window session to detect and gently intervene during periods of low-mood listening.

**Author:** Sanath

## 🧠 System Architecture

[cite_start]The project is divided into three independently-evaluated ML components[cite: 4]:

1. **Genre Classification (Two-Resolution)**
   - [cite_start]**Fine-Grained (37 classes):** Reduced from 114 raw labels to remove acoustically-indistinguishable and culturally-defined tags[cite: 4, 7].
   - [cite_start]**Coarse-Grained (10 clusters):** Super-genres mapped by dominant acoustic coordinates[cite: 4, 7].
   - [cite_start]**Ensemble Engine:** A soft-voting classifier utilizing XGBoost, LightGBM, CatBoost, and an MLP[cite: 4]. [cite_start]Features 14 raw audio signals and 15 engineered interaction terms (29 total)[cite: 4].

2. **Content-Based Recommendation**
   - [cite_start]Utilizes a KNN-all cosine index. 
   - [cite_start]Implements a genre-aware post-filter to improve track cohesion without sacrificing diversity[cite: 4].

3. **Mood Intervention**
   - [cite_start]Maintains a rolling average of session valence and energy.
   - [cite_start]Triggers an intervention flag on a hard threshold (valence < 0.26 AND energy < 0.47).
   - [cite_start]Generates synthetic query vectors to recommend tracks with a gradual valence/energy uplift[cite: 4].

## 🚀 Getting Started

### Prerequisites
- [cite_start]Python 3.11+ 
- [cite_start]A valid `dataset.csv` file (~19MB) placed in the project root.

### Local Setup (Windows / macOS / Linux)

For ease of use, initialization scripts are included to handle virtual environments and dependencies automatically.

**On Windows:**
Double-click `run.bat` or run it via terminal:
```cmd
.\run.bat