# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 18:07:45 2020

Based on:
https://blog.consdata.tech/2018/08/07/algorytmy-rekomendacyjne-przyklad-implementacji-w-pythonie.html

@authors:
    Jakub Włoch s16912
    Mateusz Woźniak s18182
"""
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import pairwise_distances

arrayOfRecommendedMovies = {}

pd.set_option("display.max_rows", 1000, "display.max_columns", 1000)
ratings_columns = ['user_name', 'movie_name', 'rating']
ratings = pd.read_csv('NAI-MovieRecommender.csv', names=ratings_columns, encoding='utf-8')

user_movies = ratings.pivot( index='user_name', columns='movie_name', values = "rating" )

user_movies.fillna( 0, inplace = True )
user_movies=pd.DataFrame(user_movies)
print (user_movies.shape)

users_similarity = 1 - pairwise_distances( user_movies.as_matrix(), metric="correlation" )
users_similarity_df = pd.DataFrame( users_similarity )


def find_neighborhood(user_id, n):

    model_knn = NearestNeighbors(metric = "correlation", algorithm = "brute")
    model_knn.fit(user_movies)
    distances, indices = model_knn.kneighbors(user_movies.iloc[user_id-1, :].values.reshape(1, -1), n_neighbors = n+1)
    similarities = 1-distances.flatten()
    #print ('{0} most similar users for user with id {1}:\n'.format(n, user_id))

    for i in range(0, len(indices.flatten())):
        # pomiń, jeśli ten sam użytkownik
        if indices.flatten()[i]+1 == user_id:
            continue;
    return similarities,indices


def predict_rate(user_id, item_id, n):
    type(item_id)
    similarities, indices=find_neighborhood(user_id, n)
    neighborhood_ratings =[]

    for i in range(0, len(indices.flatten())):
        if indices.flatten()[i]+1 == user_id:
            continue;
        else:
            neighborhood_ratings.append(user_movies.iloc[indices.flatten()[i],item_id-1])


    weights = np.delete(indices.flatten(), 0) #delete weight for input user
    prediction = round((neighborhood_ratings * weights).sum() / weights.sum())
    temp = {user_movies.columns[item_id]:prediction}
    arrayOfRecommendedMovies.update(temp)
    
    
profiledUsId = 9
i=0
while i < 245:
    predict_rate(profiledUsId,i,11)
    i+= 1

print()
print("Profilowane filmy/seriale dla użytkownika",user_movies.index[profiledUsId])
print()
print(dict(sorted(arrayOfRecommendedMovies.items(), key=lambda item: item[1])))

