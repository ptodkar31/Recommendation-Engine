# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 08:40:06 2023

@author: Priyanka
"""

import pandas as pd
anime=pd.read_csv("C:\Data Set\anime.csv",encoding='utf8')
anime.shape
#you will get 12294X7
anime.columns
anime.genre
#here we are considering only genre
from sklearn.feature_extraction.text import TfidfVectorizer
#this is term frequency inverse documnet
#each row is treated as documnet
tfidf=TfidfVectorizer(stop_words='english')
#it is going to create TfidfVectorizer to seperate all the stop words
#it is goging to seperate
#out all words from the row
#now let us check is there any null values
anime['genre'].isnull().sum()
#there are 62 null values
#suppose one movie has got genre Drama,Romance......
#there may be many empty spaces
#so let us impute these empty spaces,general is like simple imputer
anime['genre']=anime['genre'].fillna('general')
#now let us create tfidf_matrix
tfidf_matrix=tfidf.fit_transform(anime.genre)
tfidf_matrix.shape

#you will get 12294,47
# It has created sparse matrix,it means that we have 47 genre on this particular matrix,
#we want to do item based recommendation,if a user has watched gadar
#then you can recommand Shershah movie
from sklearn.matrix.pairwise import linear_kernel
#this is for measuring similarity
cosine_sim_matrix=linear_kernel(tfidf_matrix,tfidf_matrix)
#each element of tfidf_matrix is compared
#with each element of tfidf_matrix only
#output will be similarity matrix of size 12294X12294 size
#here in cosine_sim_matrix,
#there are no movie names only index are providing
#we will try to map movie name with movie index is given
#for that purpose custom function is written
anime_index=pd.Series(anime.index,index=anime['name']).drop_duplicates()
#we are converting anime_index into series format
#we want index and corresponding
anime_id=anime_index['Assassins (1995)']
anime_id
def get_recommendations(Name,topN):
    #topN=10
    #Name=Assassins (1995)
    anime_id=anime_index[Name]
    #we want to cature whole row of given movie
    #name,its score and column id
    #for that purpose we are applying cosine_sim_matrix to enumerate function
    #Enumerate function create a object
    #which we need to create in list form
    #we are using enumerate function
    #(2,10,15,18),if we apply to enumerate then it will create a list
    #(0,2,    1,10,   3,15,    4,18 )
    cosine_scores=list(enumerate(cosine_sim_matrix[anime_id]))
    #the cosine score capture we want to arrange in descending oreder
    #so that we can recommand top 10 based on highest similarity i.e score
    #if we will check the cosine score,it comprises of index:cosine score
    #x[0]=index and x[1] is cosine score
    #we want arrange tupple according to decreasing oredr
    #of the score not index
    #sorting the cosine_similarity scores based on score i.e x[1]
    consine_scores=sorted(cosine_scores, key=lambda x:x[1], reverse=True)
    #get the score of top N most similar movies
    #To capture top N movies,you need to give topN+1
    consine_scores_N =consine_scores[0:topN+1]
    #getting the movie index

    anime_idx=[i[0] for i in consine_scores_N]
    #getting consine_scores
    anime_scores=[i[1] for i in consine_scores_N]
    #we are going to use this information to create a dataframe
    #create a empty dataframe
    anime_similar_show=pd.DataFrame(columns=['name','score'])
    #assign animal_idx to name column
    anime_similar_show['name']=anime.loc[anime_idx,'name']
    #assign score to score column
    anime_similar_show['score']=anime_scores
    #while assiging values, it is by default capturing original index of the movie
    #we wnat to reset the index
    anime_similar_show.reset_index(inplace=True)
    print(anime_similar_show)
    #enter your anime and number of anime to be recommended
get_recommendations('Bad Boys (1995)',topN=0)
    
