# New Users recieve personalized suggestions with answer to only one question - Netflix recommends 5 movies you will love

## The cold start problem is an issue which has plagued streaming services for years. Platforms, like Netflix, need to ensure engagment of new users who add to revenue. Netflix has cracked the code on recommendations using the MovieLens dataset and can now recommend new users 5 movies they are going to love. 

## New users have limited prior history on their shows, watch time, and typical viewing patterns. Netflix needs to recommend the new users something they will love from the start or risk losing the business. There are lots of other streaming platforms who will take the new user in and attempt to satisfy them. Additionally, the model created must not be a simple popularity recommendation as the user wants something personalized to them. There is no way to know for sure that the user will like what is trending and it might do the opposite and turn them off from the platform. 

## Using the MovieLens dataset we attempt to recommend 5 shows to a new user based off a limited genre interest survey and millions of ratings from other viewers. The combination of these two approaches enables us to ensure the new user is satisfied with the platform. Within the Movielens dataset the movies are classified by genre and each have a rating from another user. This provides insights, a way to claissfy the movie and then historical background on what people emjoyed about it. Each movie is has tag which further describe elements of the movies which could indicate a user would like it more. Providing enough insights to gather 5 movies a new user would be interested in and therefore create a personalized expierence for them. In the chart below you can see how many movies are the same genre (Drama and Comedy), indicating a need for further variables when determining if the user would be interested. Simply asking the user their favorite genres and choosing the top five, would yield to many results. The combination of genre, movie tags, and ratings the model can correctly identify movies which a new user will enjoy. 

### Chart: 
```python
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

movies = pd.read_csv('movies.csv')

genre_counts = Counter()
for genres in movies['genres']:
    for g in str(genres).split('|'):
        if g not in ['(no genres listed)', 'IMAX']:
            genre_counts[g] += 1

labels = [g for g, c in genre_counts.most_common()]
counts = [c for g, c in genre_counts.most_common()]

plt.figure(figsize=(10, 6))
plt.barh(labels[::-1], counts[::-1], color='steelblue')
plt.xlabel('Number of Movies')
plt.title('MovieLens 25M - Number of Movies by Genre')
plt.tight_layout()
plt.savefig('genre_rating_distribution.png', dpi=150)
plt.show()
```


