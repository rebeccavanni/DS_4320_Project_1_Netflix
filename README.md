# DS 4320 Project 1: New User Recommendation Model for Neflix

This project builds a secondary relational dataset from the MovieLens 25M dataset created by GroupLens Research from University of Minnesota. With the goal of helping recommend movies to new users of a streaming platform. Given a small survey about genre interest and no prior watch history, the model will recommend a rank list of the top 5 movies using tag genome relevance and aggregate rankings from 25 million people. This repository has a fully normalized relational schema, data pipeline, solution notebook, and press release. 

**Name:** Rebecca Vanni

**NetID:** ecn2wh

**DOI:** [10.5281/zenodo.XXXXXXX](https://doi.org/10.5281/zenodo.XXXXXXX)

**License:** [MIT License](./LICENSE)

## Problem Definition 

### General Problem:

Recommending content (e.g. Netflix)

### Specific Problem:

User of netflix needs to engage and find something to watch. With a new user with no prior history (except for onboarding genre survey) of their genre preferences, can we recommend a ranked list of the top 5 movies by using both the ratings and relevance score of movies from the MovieLens 25m Dataset. 

### Rationale for Refinement:

Recommending movies to users is relatively easy when you have past history on shows watched, ratings given, time spent in a specific genre. However it becomes more difficult when the user is new to the platform or a cold start. In this case the model needs to compile basic data, like a starting survey of genre interested and what all other people on the platform enjoy. The MovieLens 25m dataset comes buth user interaction data which can be used for collaborative filtering and a tag genome (precomputed matrix of tag relevance scores). The tag-genome will allows filtering based on the content of the movie. The added challenge of the cold start recommendation of 5 movies is imperative to Netflix. They must recommend something that pulls the new user in and begins the process. They can not just use collaborative filtering because there is no history. Additionally the model can not only recommend popular movies, must reason using expressed preferences.

### Motivation for Project:

Netflix and other streaming platforms are in constant competition for users. Once on the platform, it is the goal of Netflix to find the user a show or movie which capitvates their attention. If users spend a long time searching for content, they will become frustrated and leave. Netflix's recommendation engine is imperative to its income. When the user is new the search engine is particularly important. If the user is unsatifised then they will cancel their subscription. The MovieLens dataset which was collected by GroupLens Reasearch Group has over 25 years of movies and ratings.

### Headline of Press Release:

**New Users recieve personalized suggestions with answer to only one question - Netflix recommends 5 movies you will love**
→ [press_release.md](./press_release.md)

## Domain Exposition

### Terminology

| Term / KPI | Definition | Type |
|---|---|---|
| Collaborative Filtering| Recomendation method used that predicts user's preferences based on the ratings of other similiar users | ML Method |
| Content-Based Filtering | Recomendation method used that recommends thinsg what have similiar features to those a user has suggested interest in | ML Method |
| Tag Genome | A dense matrix of relevance scores between 0 and 1 which indicate how strongly each movie exhibits preset features of interest (tags) | Dataset Feature |
| Tag Relevance Score | A continous value between 0 and 1 which indicates how strongly a movie is associated with a particular tag | Feature |
| Rating | A user's explicit 5-star evaluation of a movie (MovieLens allows half scores) | Target Variable |
| Cosine Similarity | Metric used to find the similarity between two tag-genome profiles | Similarity Metric |
| Top-5 Recommendation | The ranked output of a recommender system (5 movies the user will enjoy) | Output |
| RMSE | Root Mean Squared Error — measures average prediction error for rating-based models | Evaluation Metric |
| Implicit Feedback | Behavioral signals (views, clicks) as opposed to explicit ratings; not present in MovieLens but relevant context | Domain Term |

### Project Domain

This project lives in the streaming platform domain. Within that is is focused into the recommendation area. Recommender systems are a subfield of information filtering that seek to predict the preference a user would give to an item they have not yet encountered, and to rank candidate items accordingly. Within the broader landscape of recommender systems research, movie recommendation has served as the canonical benchmark domain since the Netflix Prize competition. Netflix has been at the forefront of recommendation services and through the MovieLens dataset we attempt to solve the problem of new users. This project's solution sits squarely in between recommendation based on past and inference, using the tag genome as a rich content feature space and aggregate rating statistics as a collaborative popularity. 

### Background Reading:

Readings are linked [here]([https://myuva-my.sharepoint.com/:f:/g/personal/ecn2wh_virginia_edu/IgA9ZguJMmhsRI5pEXVOEl3LAZx3wsYaePkO5Oyb9hZjxnI?e=zhi7z6](https://myuva-my.sharepoint.com/:f:/g/personal/ecn2wh_virginia_edu/IgA9ZguJMmhsRI5pEXVOEl3LAQqIJn5mLbPQUKsEjolq94M?e=6jW6BZ))

| # | Title | Brief Description | Link |
|---|---|---|---|
| R1 | How Netflix’s Recommendations System Works | Broad background from Netflix on what they use to create their recommendations (users interactions, platform wide interaction, and information about item, like title or genre) and their current aproach to new user problem. | [link](https://myuva-my.sharepoint.com/:i:/g/personal/ecn2wh_virginia_edu/IQCjgCWgY9idRbA23qiyJTyOAWnfXyBIewRRC24XvMgLmaI?e=OvSfAy) |
| R2 | MovieLens 25M Dataset Summary | Readme from the Movielens dataset. Includes the copy right, description of variables, and explanation on the calculated variables.Good to understand origin of data | [link](https://myuva-my.sharepoint.com/:u:/g/personal/ecn2wh_virginia_edu/IQCFNRM3NdS4T4RHOGr7-hN_Ad967DVy6lahz673TLe9j04?e=ugdKek) |
| R3 | Movie Recommender Systems: Concepts, Methods, Challenges, and Future Directions | Literature review on the current recommendations used by streaming platforms. Highlighting the filtering methods, machine-learning algorithims, and basic preformanc metrics. Also provides insights into what the future of recommendation algorithims will look liked. | [link](https://myuva-my.sharepoint.com/:b:/g/personal/ecn2wh_virginia_edu/IQASZWr1EL88T6bkvqmUlyyzAesqD1z8tgrqORSwNp-XYdI?e=554HG6)
| R4 | The Cold Start Problem for Recommender Systems | Blog on the cold start problem, explains why it is so hard to recommend items to new users. Goes through the basic problem, some of its nuances (identifying new users, returning users, etc), and provides a solution via use survey and popularity data. | [link](https://myuva-my.sharepoint.com/:u:/g/personal/ecn2wh_virginia_edu/IQBKPXXgeKURTIv0u0Ab0UDHAeGslu7eka3ZE1GzrtCcC-E?e=vqyhac)
| R5 | An Introduction to Movie Genres | Blog explaining how a film is placed into a genre and a summary of the genre types (Action, Drama, Comedy, etc). Explains why a genre is relevant to movie. | [link](https://myuva-my.sharepoint.com/:u:/g/personal/ecn2wh_virginia_edu/IQDg_WPqt51uSZyFQAFAv2c4AepA0qgl1_2vRhY63Uns9PY?e=9S6y0j) |

## Data Creation 

### Provenance

The raw data for this Project comes from the MovieLens 25M dataset. This data was published by GroupLens Research at the University of Minnesota. It can be downloaded online, from the GroupLens website and it a compressed archive. Once downloaded the data will be contained in a folder. Within the folder there is six CVS files: movies.csv, ratings.cvs, tags.csv, links.csv, genome-scores.csv, genome-tags.csv. Within the files there are 25 millions ratings submited by 162,541 users on 62,423 movies between Jan.1995 adn Nov.2019. The genome file was created by MovieLens using a machine learning model trained on user-submitted tags, ratings, and review text. The model assigns a tag to a movie based on the relevance of its other tags, ratings, and movie title.

In order to create the secondary dataset, I did transformed the raw files. I needed to calculate the per-movie aggregate stats (from ratings file). Additonally created minimum threshold of 50 ratings per movie to ensure movies with not enough information wouldnt mess with the model. The genome-score file and genome-tags were joined on genre-relevant tags which producted a content-feature vector per movie. Then from there joined the movie metadata with teh rating statistics, and genome score. Which created the relational structure to be used for cold-starts.

### Code Table


| File | Description | Link |
|---|---|---|
| pipeline/build_dataset.py | Loads raw MovieLens CSVs computes per-movie avg rating and count, filters out by threshold (50), parses genres, and saves into clean files (movies_clean and raggtings_agg) | [link](https://github.com/rebeccavanni/DS_4320_Project_1_Netflix/blob/main/pipeline/build_dataset.py) |

### Bias identification

Areas for bias in the MovieLens dataset include: Self-selection bias where only users who actively rate movies are represented on platform. Which means passive viewers are absent from the sample, and only self-motivated users are included which doesn't represent the population. Popularity bias, where heavily watched movies accumulate many more ratings than niche films. Which makes aggregate stats higher for the block-busters than smaller films. There is also temporal and demographic bias. The dataset spans 25 years during this time rating behavior and trends have changed vastly. Additionally older movies have longer to accumulate ratings. Demography was not verified when the dataset was created. So the dataset will likely be skewed to younger, english speaking, and technically literate individuals.

### Bias mitigation

Popularity bias can be mitigated by using the Bayesian average or a credibility weighted mean. It will shrink the movie's estimated rating towards the global mean, so movies with few ratings won't out rank movies with many. Creating a threshold to remove movies with less than 50 rankings will also prevent bias from statistically unreliable movies. Temporal bias can be partially fixed by a recent weight to account for movies which haven't had enough time to accumulate as many ratings. Can't eliminate the demographic bias, but using the tag-genome is not user dependent as it is from content, so it will be used for the cold-start situation where the user is unknown.

### Rationale for key decisions

Threshold of 50 ratings: a threshold was needed to balance the reliability of low rating movies. If the movie is too low (5 ratings) in ratings then it creates noise in the dataset and if it is too high then it excludes a large fraction from the catalog. 50 ratings will remove the outliers while maintaining older and niche titles.


Genre to tag mapping: MovieLens had genre labels like Action and drama. These are very broad, but the tag genome contains 1128 fine-grained semantic tags. By mapping the tags to genre (using highest-relevance tags per genre and genre keywords) the user's survey response can provide broader aid. Can be seen based on what genres they like to then understand what in a movie they want to see.


The final ranking combines a content-based cosine similarity score with a Bayesian-weighted aggregate rating at 50/50. This is the largest single source of modeling uncertainty. Shifting weight toward collaborative ratings increases popularity bias while shifting toward genome similarity surfaces more niche films. This parameter is explicitly tunable and flagged for sensitivity analysis.

## Metadata

### ER diagram

![erd_hw8.drawio.png](./data/erd_hw8.drawio.png)

### Data table

| Table | Description | Link to CSV |
|---|---|---|
| movies_clean.csv | Movie metadata: movieId, title, genres (pipe-delimited); filtered to movies with ≥50 ratings | [movies.csv](https://myuva-my.sharepoint.com/:x:/g/personal/ecn2wh_virginia_edu/IQARLRHF1EdrSY64vkLu8doxASY409cDGU0Rp48BwU9ZdsE?e=mFO8Si) |
| ratings_agg.csv | Per-movie aggregate statistics: avg_rating and rating_count; one row per movieId | [ratings_agg.csv](https://myuva-my.sharepoint.com/:x:/g/personal/ecn2wh_virginia_edu/IQANrch5XAKxQLzSRZUjooB_AT6U0oCgPvDZ5TLK64V-Eak?e=BNL0r9) |
| genome-scores.csv | Tag relevance scores: one row per (movieId, tagId) pair with a continuous relevance score 0–1 | [genome-scores.csv](https://myuva-my.sharepoint.com/:x:/g/personal/ecn2wh_virginia_edu/IQDUbJgJWjqDTp73OukxyRF1AaxoOtRE5cxEnO9e84i3EBo?e=wNIcsP) |
| genome-tags.csv | Tag label lookup: one row per tagId mapping to a human-readable tag string | [genome-tags.csv](https://myuva-my.sharepoint.com/:x:/g/personal/ecn2wh_virginia_edu/IQDBn1IsJoWtQqyXVBTxPO28AYxlXRFKs19QlZIau_ybmqI?e=rvDu8t) |

### Data dictionary

Movies:

| Feature | Data Type | Description | Example |
|---|---|---|---|
| movieId | INTEGER | Unique movie identifier assigned by MovieLens | 1 |
| title | VARCHAR | Full movie title including release year in parentheses | Toy Story (1995) |
| genres | VARCHAR | Pipe-delimited genre labels assigned to the movie | Adventure\|Animation\|Children |

Ratings Aggregated

| Feature | Data Type | Description | Example |
|---|---|---|---|
| movieId | INTEGER | Foreign key referencing MOVIES.movieId | 1 |
| avg_rating | FLOAT | Mean of all user ratings for this movie (scale 0.5–5.0) | 3.92 |
| rating_count | INTEGER | Total number of user ratings submitted for this movie | 49695 |

Genome Scores

| Feature | Data Type | Description | Example |
|---|---|---|---|
| movieId | INTEGER | Foreign key referencing MOVIES.movieId | 1 |
| tagId | INTEGER | Foreign key referencing GENOME_TAGS.tagId | 1 |
| relevance | FLOAT | Continuous score 0–1 indicating how strongly the movie exhibits this tag | 0.029 |

Genome Tags

| Feature | Data Type | Description | Example |
|---|---|---|---|
| tagId | INTEGER | Unique tag identifier | 1 |
| tag | VARCHAR | Human-readable semantic tag label | 007 |

### Uncertainty quantification for numerical features

| Feature | Table | Range | Mean | Std Dev | Source of Uncertainty |
|---|---|---|---|---|---|
| avg_rating | RATINGS_AGG | 0.5 – 5.0 | ~3.53 | ~0.52 | Computed mean of user ratings; movies with low rating_count have high variance — a movie with 50 ratings has a much wider confidence interval around its true mean than one with 50,000 |
| rating_count | RATINGS_AGG | 50 – ~81,491 | ~1,200 | ~3,800 | Right-skewed; a few blockbusters dominate; the minimum threshold of 50 truncates the left tail but high variance remains |
| relevance | GENOME_SCORES | 0.0 – 1.0 | ~0.07 | ~0.13 | Generated by a ML model trained on user tags and ratings — carries model uncertainty and is not a ground-truth label; scores near 0.5 are less reliable than those near 0 or 1 |
