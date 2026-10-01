# Spotify Audio Features: What Predicts Popularity?

An exploratory machine learning project using the Spotify Tracks Dataset (Kaggle, ~114,000 rows, 125 genres) to find patterns in popularity, genre, and audio features, and to group songs by "vibe".

**Tools:** Python, pandas, seaborn, matplotlib, scikit-learn

## Questions

1. Which audio features and genres relate to a track's popularity?
2. Can popularity be predicted from audio features and genre?
3. Can songs be grouped into meaningful "vibes"?

## Data cleaning

- Dropped the leftover index column and one row with missing values.
- Removed duplicate tracks (the same `track_id` appears once per genre it is listed under). This left roughly 89,700 unique tracks, and it prevents the same song from landing in both the training and test sets.

## Key findings

### 1. Audio features are weak predictors of popularity
Every audio feature has a correlation below 0.15 with popularity. The strongest is `instrumentalness` (-0.13): instrumental tracks tend to be less popular.

Features do relate to each other sensibly:
- Energy and loudness: +0.76
- Energy and acousticness: -0.73
- Danceability and valence (happiness): +0.49

### 2. Genre matters more than sound
K-pop, pop-film, metal, and chill have the highest average popularity (about 54 to 59). Genre dominates the model's feature importance.

Some of this reflects the dataset rather than listener taste. For example, 66% of `iranian` tracks and 61% of `romance` tracks have a popularity of 0, which makes those genres a strong "probably unpopular" signal for the model.

### 3. Simple beat complex
Three models on the same 80/20 split, plus shuffled 5-fold cross-validation (shuffling matters: the data is sorted by genre, so unshuffled folds test on genres the model never saw and give misleading negative R²).

| Model | MAE | RMSE | R² | 5-fold CV R² |
|---|---|---|---|---|
| Linear Regression | 12.02 | 16.86 | 0.32 | 0.328 ± 0.005 |
| Random Forest | 14.55 | 18.40 | 0.19 | 0.197 ± 0.006 |
| Decision Tree | 16.01 | 19.53 | 0.09 | not run |
| Baseline (average) | 17.12 | - | - | - |

Linear regression cut error by about 30% versus the baseline, and cross-validation matched the single split. A likely reason the simpler model wins (untested): most usable signal is a genre-level average, while deeper models overfit noisy audio features.

Excluding zero-popularity tracks raised the Random Forest's R² from 0.19 to 0.31 (MAE 15.24 to 11.79 against its own baseline), because those tracks add noise audio and genre can't explain. That model answers a narrower question: how popular a track is, given that it has some listeners. Without the zeros, `instrumentalness` becomes its most important feature.

**Where the model struggles:** in the Random Forest's actual-vs-predicted plot, predictions cluster at a few levels (about 10-20, 35, and 55-60), very popular songs (70+) are under-predicted, and many true zeros are predicted at 10-20.

### 4. Songs fall into four "vibes"
K-Means (k=4) on danceability, energy, valence, and acousticness:

| Cluster | Profile | Songs | Avg popularity |
|---|---|---|---|
| Feel-good party | High danceability, energy, valence | 30,202 | 33.1 |
| Mellow acoustic | Mid energy, high acousticness | 19,059 | 34.1 |
| Sad and quiet | Low energy and valence, very acoustic | 12,753 | 30.0 |
| Intense and dark | High energy, low valence, not acoustic | 27,726 | 34.1 |

Popularity is almost the same across vibes (30 to 34), so how a song feels says little about how popular it is.

## Limitations

- The dataset has no artist fame, playlist placement, or release date, which likely drive much of popularity (even the best model, R² of about 0.33, leaves most of it unexplained).
- Each track keeps only one genre label after removing duplicates.
- Genre averages can be skewed by small genres and by how the data was collected.
- The clusters are slices of a continuous distribution, not sharply separated groups.

## Possible next steps

- Add artist-level features (for example, an artist's average popularity).
- Try gradient boosting and compare it to linear regression.
- Tune the number of clusters (elbow method or silhouette score).
- Apply the clustering to your own Spotify listening history.
