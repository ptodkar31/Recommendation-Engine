# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 21:31:46 2024

@author: Priyanka
"""

import pandas as pd
game_data=pd.read_csv("C:\Data Set\game.csv")
#Let us check the diamensions of the dataframe
game_data.shape
#it will be 5000,3
#let us check the columns of the data
game_data.columns
#Let us check top entries in dataframe
game_data.head()
#game column is text,hence create TfidfVector matrix
from sklearn.feature_extraction.text import TfidfVectorizer
# craeting TfidfVectorizer to seperate all stop words
Tfidf=TfidfVectorizer(stop_words='english')
#checking for nan values
game_data['rating'].isna().sum()
tfidf_matrix=Tfidf.fit_transform(game_data.game)
tfidf_matrix.shape
#measure the similarity using cosine similarity
from sklearn.metrics.pairwise import linear_kernel
#creating cosine similarity matrix which will create matrix of similarity
cos_sim_matrix=linear_kernel(tfidf_matrix,tfidf_matrix)
# we will create series of game_data
game_data_index=pd.Series(game_data.index,index=game_data['userId']).drop_duplicates()
game_data_index.head()
#checking the same for random game picked up
game_data_id=game_data_index[269]
game_data_id
# Now let us create user defined function
def get_recommendations(UserId,topN):
    #getting game index and its user id
    game_data_id=game_data_index[UserId]
    #getting pairwise similarity score
    cosine_scores=list(enumerate(cos_sim_matrix[game_data_id]))
    cosine_scores = sorted(cosine_scores, key = lambda x:x[1], reverse=True)
    cosine_scores_N=cosine_scores[0:topN+1]
    #getting game index
    game_data_idx=[i[0] for i in cosine_scores_N]
    game_data_scores=[i[1] for i in cosine_scores_N]
    games_similar=pd.DataFrame(columns=["game","rating"])
    games_similar['game']=game_data.loc[game_data_idx,'game']
    games_similar['rating']=game_data_scores
    games_similar.reset_index(inplace=True)
    print(games_similar)
    
    #let us use this function which will give topN gamelist
get_recommendations(285,topN=10)
game_data_index[285]