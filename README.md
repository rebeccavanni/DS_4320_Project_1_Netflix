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

Readings are linked [link]([https://](https://myuva-my.sharepoint.com/:f:/g/personal/ecn2wh_virginia_edu/IgA9ZguJMmhsRI5pEXVOEl3LAZx3wsYaePkO5Oyb9hZjxnI?e=zhi7z6))

| # | Title | Brief Description | Link |
|---|---|---|---|
| R1 | How Netflix’s Recommendations System Works | Broad background from Netflix on what they use to create their recommendations (users interactions, platform wide interaction, and information about item, like title or genre) and their current aproach to new user problem. | [link](https://myuva-my.sharepoint.com/:i:/g/personal/ecn2wh_virginia_edu/IQCjgCWgY9idRbA23qiyJTyOAWnfXyBIewRRC24XvMgLmaI?e=OvSfAy) |
| R2 | MovieLens 25M Dataset Summary | Readme from the Movielens dataset. Includes the copy right, description of variables, and explanation on the calculated variables.Good to understand origin of data | [link](https://myuva-my.sharepoint.com/:u:/g/personal/ecn2wh_virginia_edu/IQCFNRM3NdS4T4RHOGr7-hN_Ad967DVy6lahz673TLe9j04?e=ugdKek) |
| R3 | Movie Recommender Systems: Concepts, Methods, Challenges, and Future Directions | Literature review on the current recommendations used by streaming platforms. Highlighting the filtering methods, machine-learning algorithims, and basic preformanc metrics. Also provides insights into what the future of recommendation algorithims will look liked. | [link](https://myuva-my.sharepoint.com/:b:/g/personal/ecn2wh_virginia_edu/IQASZWr1EL88T6bkvqmUlyyzAesqD1z8tgrqORSwNp-XYdI?e=554HG6)
| R4 | The Cold Start Problem for Recommender Systems | Blog on the cold start problem, explains why it is so hard to recommend items to new users. Goes through the basic problem, some of its nuances (identifying new users, returning users, etc), and provides a solution via use survey and popularity data. | [link](https://myuva-my.sharepoint.com/:u:/g/personal/ecn2wh_virginia_edu/IQBKPXXgeKURTIv0u0Ab0UDHAeGslu7eka3ZE1GzrtCcC-E?e=vqyhac)
| R5 | An Introduction to Movie Genres | Blog explaining how a film is placed into a genre and a summary of the genre types (Action, Drama, Comedy, etc). Explains why a genre is relevant to movie. | [link](https://myuva-my.sharepoint.com/:u:/g/personal/ecn2wh_virginia_edu/IQDg_WPqt51uSZyFQAFAv2c4AepA0qgl1_2vRhY63Uns9PY?e=9S6y0j) |


