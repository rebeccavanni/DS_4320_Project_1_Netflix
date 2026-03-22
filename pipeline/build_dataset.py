import pandas as pd
import os

#folder to hold raw data, in colab
RAW_DIR    = "/content"      
#folder for clean data, in colab
OUTPUT_DIR = "/content/data"      
#setting threshold at 50 movies 
MIN_RATINGS = 50          

os.makedirs(OUTPUT_DIR, exist_ok=True)

#getting files
movies   = pd.read_csv(os.path.join(RAW_DIR, "movies.csv"))
ratings  = pd.read_csv(os.path.join(RAW_DIR, "ratings.csv"))
g_scores = pd.read_csv(os.path.join(RAW_DIR, "genome-scores.csv"))
g_tags   = pd.read_csv(os.path.join(RAW_DIR, "genome-tags.csv"))

#computing the aggregate ratings
ratings_agg = (
    ratings
    .groupby("movieId")
    .agg(
        avg_rating   = ("rating", "mean"),
        rating_count = ("rating", "count")
    )
    .reset_index()
)

#rounding to make easier to look at (4 decimal places
ratings_agg["avg_rating"] = ratings_agg["avg_rating"].round(4)


#filtering out based on threshold
ratings_agg = ratings_agg[ratings_agg["rating_count"] >= MIN_RATINGS].copy()



#cleaning the movies in data
#remove movies with no genre listed or only IMAX tag
movies = movies[~movies["genres"].isin(["(no genres listed)"])].copy()

#extract release year from title into its own column
movies["year"] = movies["title"].str.extract(r"\((\d{4})\)$")

#keep movies with enough ratings
movies_clean = movies.merge(ratings_agg[["movieId"]], on="movieId", how="inner")

#filter genome for valid scores
valid_movie_ids = set(movies_clean["movieId"])
genome_filtered = g_scores[g_scores["movieId"].isin(valid_movie_ids)].copy()

#merdging tag labels
genome_filtered = genome_filtered.merge(g_tags, on="tagId", how="inner")

#saving the outputs into cleaned data
movies_clean_path   = os.path.join(OUTPUT_DIR, "movies_clean.csv")
ratings_agg_path    = os.path.join(OUTPUT_DIR, "ratings_agg.csv")
genome_filtered_path = os.path.join(OUTPUT_DIR, "genome_filtered.csv")

movies_clean.to_csv(movies_clean_path,    index=False)
ratings_agg.to_csv(ratings_agg_path,      index=False)
genome_filtered.to_csv(genome_filtered_path, index=False)
