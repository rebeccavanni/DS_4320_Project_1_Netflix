# DS 4320 Project 1 — Solution Pipeline
**New User Recommendation Model for Netflix**

**Author:** Rebecca Vanni  
**NetID:** ecn2wh

---

## Setup and Packages

```python
import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import logging
import os
from sklearn.metrics.pairwise import cosine_similarity

# Creating Pipeline
logging.basicConfig(
    filename='pipeline.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.info('Pipeline notebook started')
print('Setup complete.')
```

---

## Load Raw Data into DuckDB

Creating the four tables:

- **movies** — cleaned movie metadata with year extracted
- **ratings_agg** — per-movie average rating and count (min 50 ratings threshold)
- **genome_tags** — tag label lookup table
- **genome_scores** — tag relevance scores per movie (filtered to valid movies only)

```python
# Paths where the raw CSVs will be in Colab
RAW_DIR = '/content'
DB_PATH = '/content/movielens.db'
# Setting the threshold for movies with 50 ratings or above
MIN_RATINGS = 50

# Connecting to DuckDB
try:
    con = duckdb.connect(DB_PATH)
    logging.info('Connected to DuckDB at %s', DB_PATH)

    # Drop tables that already exist
    for t in ['genome_scores', 'genome_tags', 'ratings_agg', 'movies']:
        con.execute(f'DROP TABLE IF EXISTS {t}')

    # Table 1: movies
    # Extract release year from title using regex; drop rows with no genre
    con.execute(f"""
        CREATE TABLE movies AS
        SELECT
            CAST(movieId AS INTEGER) AS movieId,
            title,
            genres,
            TRY_CAST(regexp_extract(title, '\\((\\d{{4}})\\)$', 1) AS INTEGER) AS year
        FROM read_csv_auto('{RAW_DIR}/movies.csv')
        WHERE genres != '(no genres listed)'
    """)
    logging.info('movies created')

    # Table 2: ratings_agg
    # Aggregate 25M ratings into per-movie avg + count and apply min filter
    con.execute(f"""
        CREATE TABLE ratings_agg AS
        SELECT
            CAST(movieId AS INTEGER) AS movieId,
            ROUND(AVG(rating), 4)    AS avg_rating,
            COUNT(*)                 AS rating_count
        FROM read_csv_auto('{RAW_DIR}/ratings.csv', ignore_errors=true)
        GROUP BY movieId
        HAVING COUNT(*) >= {MIN_RATINGS}
    """)
    logging.info('ratings_agg created')

    # Apply threshold to only keep movies with enough ratings
    con.execute('DELETE FROM movies WHERE movieId NOT IN (SELECT movieId FROM ratings_agg)')

    # Table 3: genome_tags
    con.execute(f"""
        CREATE TABLE genome_tags AS
        SELECT
            CAST(tagId AS INTEGER) AS tagId,
            tag
        FROM read_csv_auto('{RAW_DIR}/genome-tags.csv')
    """)
    logging.info('genome_tags created')

    # Table 4: genome_scores
    # Filter to only movies that survived the threshold and cleaning
    con.execute(f"""
        CREATE TABLE genome_scores AS
        SELECT
            CAST(gs.movieId   AS INTEGER) AS movieId,
            CAST(gs.tagId     AS INTEGER) AS tagId,
            CAST(gs.relevance AS DOUBLE)  AS relevance
        FROM read_csv_auto('{RAW_DIR}/genome-scores.csv') gs
        WHERE gs.movieId IN (SELECT movieId FROM movies)
    """)
    logging.info('genome_scores created')

except Exception as e:
    logging.error('Data load failed: %s', str(e))
    raise
```

---

## Query

The three queries used to extract data for the recommender:

- **Genre mapping** — explodes the pipe-delimited genre string into one row per genre per movie so a user's survey responses can be matched.
- **Tag-genome pivot** — pulls the full relevance matrix for candidate movies. This is the content feature space for cosine similarity.
- **Popularity scores** — computes a normalized popularity score combining `avg_rating` and `log(rating_count)` to break ties between similarly relevant movies.

```python
# Genre mapping — explodes movieId into one row per genre
# Filters candidate movies by user's liked genres
try:
    genre_df = con.execute("""
        SELECT
            m.movieId,
            m.title,
            m.year,
            TRIM(genre_item) AS genre
        FROM movies m,
             UNNEST(string_split(m.genres, '|')) AS t(genre_item)
        WHERE TRIM(genre_item) != 'IMAX'
        ORDER BY m.movieId
    """).df()

    logging.info('genre_df: %d rows, %d unique movies', len(genre_df), genre_df['movieId'].nunique())
    print(f'genre_df: {len(genre_df):,} rows | {genre_df["movieId"].nunique():,} unique movies')
    print(f'Genres available: {sorted(genre_df["genre"].unique())}')
    genre_df.head()

except Exception as e:
    logging.error('Genre query failed: %s', str(e))
    raise

# Pull genome scores and join tags
# Movie × tag matrix for cosine similarity
try:
    genome_df = con.execute("""
        SELECT
            gs.movieId,
            gt.tag,
            gs.relevance
        FROM genome_scores gs
        JOIN genome_tags gt ON gs.tagId = gt.tagId
        ORDER BY gs.movieId, gt.tag
    """).df()

    logging.info('genome_df: %d rows', len(genome_df))
    print(f'genome_df: {len(genome_df):,} rows')
    genome_df.head()

except Exception as e:
    logging.error('Genome query failed: %s', str(e))
    raise

# Compute popularity score
# Combines avg_rating and log-scaled rating_count into a single 0–1 normalized score
# Normalized to include more niche movies alongside popular ones
try:
    popularity_df = con.execute("""
        WITH scored AS (
            SELECT
                r.movieId,
                m.title,
                r.avg_rating,
                r.rating_count,
                -- weighted score: 60% average rating (normalized 0-5) + 40% log popularity
                (0.6 * (r.avg_rating / 5.0)) +
                (0.4 * (LN(r.rating_count) / LN((SELECT MAX(rating_count) FROM ratings_agg))))
                    AS popularity_score
            FROM ratings_agg r
            JOIN movies m ON r.movieId = m.movieId
        )
        SELECT
            movieId,
            title,
            avg_rating,
            rating_count,
            ROUND(popularity_score, 4) AS popularity_score
        FROM scored
        ORDER BY popularity_score DESC
    """).df()

    logging.info('popularity_df: %d rows', len(popularity_df))
    print(f'popularity_df: {len(popularity_df):,} movies')
    popularity_df.head(10)

except Exception as e:
    logging.error('Popularity query failed: %s', str(e))
    raise
```

---

## Model — Cold-Start Problem

**Model Choice: Content-Based Filtering via Cosine Similarity on the Tag Genome**

Steps for the solution:

1. The user answers a one-question genre survey — goal of having 1–3 genres to filter between
2. Identify genre-relevant tags from the genome
3. Build a user profile vector (the mean tag-genome vector of the top-rated movies in those genres)
4. Compute cosine similarity between the user profile and every candidate movie's genome vector
5. Rerank by a weighted combination of cosine similarity (70%) and popularity score (30%)
6. The top 5 results are returned as the recommendation list

```python
# Pivot the genome into a movie × tag matrix
# Core feature matrix for cosine similarity
try:
    genome_matrix = genome_df.pivot(index='movieId', columns='tag', values='relevance').fillna(0)
    logging.info('genome_matrix shape: %s', genome_matrix.shape)
    print(f'Genome matrix shape: {genome_matrix.shape}  (movies × tags)')
    genome_matrix.iloc[:3, :5]

except Exception as e:
    logging.error('Pivot failed: %s', str(e))
    raise

# Cold-start recommender using cosine similarity on the tag genome.
def recommend_movies(user_genres, genre_df, genome_matrix, popularity_df,
                     top_n=5, sim_weight=0.7, pop_weight=0.3):

    try:
        # Find candidate movies that match at least one of the user's genres
        candidates = genre_df[genre_df['genre'].isin(user_genres)]['movieId'].unique()
        logging.info('Candidates for genres %s: %d movies', user_genres, len(candidates))

        if len(candidates) == 0:
            raise ValueError(f'No movies found for genres: {user_genres}')

        # Build the user profile vector
        # Popular movies as the profile seed ensures a meaningful vector
        pop_candidates = (
            popularity_df[popularity_df['movieId'].isin(candidates)]
            # Take the top-50 most popular genre-matching movies and average their genome vectors
            .nlargest(50, 'popularity_score')['movieId']
            .values
        )

        # Filter to only those that have genome data
        seed_ids = [mid for mid in pop_candidates if mid in genome_matrix.index]
        if len(seed_ids) == 0:
            raise ValueError('No seed movies with genome data found.')

        user_profile = genome_matrix.loc[seed_ids].mean(axis=0).values.reshape(1, -1)
        logging.info('User profile built from %d seed movies', len(seed_ids))

        # Filter genome_matrix to candidate movies only
        candidate_ids = [mid for mid in candidates if mid in genome_matrix.index]
        candidate_matrix = genome_matrix.loc[candidate_ids]

        # Compute cosine similarity between user profile and all candidates
        sim_scores = cosine_similarity(user_profile, candidate_matrix.values)[0]
        sim_series = pd.Series(sim_scores, index=candidate_ids, name='cosine_sim')

        # Normalize similarity scores to 0–1
        sim_min, sim_max = sim_series.min(), sim_series.max()
        sim_norm = (sim_series - sim_min) / (sim_max - sim_min + 1e-9)

        # Merge with popularity scores
        pop_lookup = popularity_df.set_index('movieId')['popularity_score']
        pop_norm = pop_lookup.reindex(candidate_ids).fillna(0)
        # Normalize popularity too
        pop_norm = (pop_norm - pop_norm.min()) / (pop_norm.max() - pop_norm.min() + 1e-9)

        # Final weighted score
        final_score = (sim_weight * sim_norm) + (pop_weight * pop_norm)

        # Assemble results DataFrame
        results = pd.DataFrame({
            'movieId'    : candidate_ids,
            'cosine_sim' : sim_series.values,
            'pop_score'  : pop_lookup.reindex(candidate_ids).values,
            'final_score': final_score.values
        }).set_index('movieId')

        # Merge in title and rating info
        results = results.join(popularity_df.set_index('movieId')[['title', 'avg_rating', 'rating_count']])
        results = results.sort_values('final_score', ascending=False)

        # Remove seed movies from recommendations (don't recommend profile-builders)
        results = results[~results.index.isin(seed_ids)]

        logging.info('Recommendation complete. Returning top %d.', top_n)
        return results.head(top_n).reset_index()

    except Exception as e:
        logging.error('Recommendation failed: %s', str(e))
        raise
# Returns a DataFrame with top_n recommended movies and their scores
```

---

## Top-5 Recommendations (Experiment)

Simulating with a fake user — implementing two genres they marked as interested.

```python
# Simulate a new user survey
# This would appear as a UI when the user signs into the platform
USER_GENRES = ['Action', 'Sci-Fi']

recommendations = recommend_movies(
    user_genres   = USER_GENRES,
    genre_df      = genre_df,
    genome_matrix = genome_matrix,
    popularity_df = popularity_df,
    top_n         = 5,
    sim_weight    = 0.7,
    pop_weight    = 0.3
)

# Create clean table to show results
display_cols = ['title', 'avg_rating', 'rating_count', 'cosine_sim', 'final_score']
print('=== Top-5 Recommended Movies ===')
recommendations[display_cols].style.format({
    'avg_rating'   : '{:.2f}',
    'rating_count' : '{:,.0f}',
    'cosine_sim'   : '{:.4f}',
    'final_score'  : '{:.4f}'
})
```

---

## Visualization

**Score Breakdown Bar Chart:** A horizontal stacked bar chart was chosen to make the model's reasoning transparent. This shows exactly how much each recommendation's final score comes from content similarity versus popularity so the reader can see the model is not simply recommending the most popular movies. Netflix brand colors were used to ground the visualization in the platform context and distinguish the two score components clearly.

**Tag Genome Heatmap:** A heatmap was chosen to display the 20 most variable tags across the top-5 recommended movies because it simultaneously reveals both which content features drove each recommendation and how the five movies differ from one another along those features. Variance-based tag selection ensures the heatmap shows the most discriminating features rather than tags that are uniformly high or low across all recommendations, making the content-based reasoning interpretable at a glance.

```python
# Score breakdown bar chart
# Shows how much each component (content similarity vs popularity) contributes
sns.set_theme(style='whitegrid', font_scale=1.1)
BRAND_RED  = '#E50914'   # Netflix red
BRAND_DARK = '#221F1F'   # Netflix dark
BRAND_GRAY = '#B3B3B3'

fig, ax = plt.subplots(figsize=(10, 5))
titles   = recommendations['title'].str.replace(r'\s*\(\d{4}\)', '', regex=True)
sim_vals = recommendations['cosine_sim']
pop_vals = recommendations['pop_score']
y_pos    = range(len(titles))

# Stacked horizontal bars (similarity component + popularity component)
bars1 = ax.barh(y_pos, sim_vals * 0.7, color=BRAND_RED,  label='Content Similarity (70%)', height=0.55)
bars2 = ax.barh(y_pos, pop_vals * 0.3, left=sim_vals * 0.7,
                color=BRAND_GRAY, label='Popularity Score (30%)', height=0.55)

# Rank labels on the left
ax.set_yticks(list(y_pos))
ax.set_yticklabels([f'#{i+1}  {t}' for i, t in enumerate(titles)], fontsize=11)
ax.invert_yaxis()
ax.set_xlabel('Weighted Score Contribution', fontsize=11)
ax.set_title(
    f'Top-5 Netflix Recommendations\nNew User Genres: {", ".join(USER_GENRES)}',
    fontsize=13, fontweight='bold', color=BRAND_DARK, pad=12
)
ax.legend(loc='lower right', framealpha=0.9)
ax.set_facecolor('#F9F9F9')
fig.patch.set_facecolor('white')

# Add avg rating annotation
for i, row in recommendations.iterrows():
    ax.text(
        row['final_score'] + 0.002, i,
        f'★ {row["avg_rating"]:.2f}  ({row["rating_count"]/1000:.0f}K ratings)',
        va='center', fontsize=9, color=BRAND_DARK
    )

plt.tight_layout()
plt.savefig('recommendation_scores.png', dpi=150, bbox_inches='tight')
plt.show()

# Tag Relevance Heatmap for Top-5 Movies
# Shows which content features (tags) are most strongly associated with each recommendation
# Makes the model's content-based reasoning interpretable

# Select the top 15 most discriminating tags for the recommended movies
top_movie_ids = recommendations['movieId'].tolist()
top_titles    = dict(zip(recommendations['movieId'],
                         recommendations['title'].str.replace(r'\s*\(\d{4}\)', '', regex=True)))

# Get genome vectors for just the top-5
heat_data = genome_matrix.loc[[mid for mid in top_movie_ids if mid in genome_matrix.index]]
heat_data.index = [top_titles[mid] for mid in heat_data.index]

# Pick the top 20 tags by variance across these 5 movies (most informative)
top_tags  = heat_data.var(axis=0).nlargest(20).index
heat_data = heat_data[top_tags]

fig, ax = plt.subplots(figsize=(14, 4))
sns.heatmap(
    heat_data,
    ax=ax,
    cmap='Reds',
    linewidths=0.4,
    linecolor='#EEEEEE',
    cbar_kws={'label': 'Tag Relevance Score', 'shrink': 0.8},
    annot=True,
    fmt='.2f',
    annot_kws={'size': 8}
)

ax.set_title(
    'Tag Genome Relevance — Top-5 Recommended Movies\n(Top 20 most variable tags shown)',
    fontsize=12, fontweight='bold', pad=12
)
ax.set_xlabel('Tag', fontsize=10)
ax.set_ylabel('')
ax.tick_params(axis='x', rotation=45, labelsize=8)
ax.tick_params(axis='y', rotation=0,  labelsize=9)

plt.tight_layout()
plt.savefig('tag_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## Analysis Rationale

The cold-start recommendation model uses content-based filtering via cosine similarity on the MovieLens tag genome, rather than collaborative filtering, because new users have no watch history to compute user-user or item-item similarity against. Final scores blend 70% content similarity with 30% log-normalized popularity to ensure recommendations are both personally relevant to the user's stated genre preferences and well-regarded by the broader platform community. A log-scaled rating count is used because it prevents blockbusters from dominating and rewards quality over volume.

## Limitations

- The tag genome covers only 13,000 of the 62,000 movies in the full dataset; movies without genome data are excluded from content-based ranking.
- The user profile is bootstrapped from popularity, not true user taste — this is appropriate for cold-start but should be updated quickly as watch history accumulates.
- Genre labels are coarse; a user who likes "Action" may prefer very different sub-genres (martial arts vs. superhero). The tag genome partially compensates for this.
