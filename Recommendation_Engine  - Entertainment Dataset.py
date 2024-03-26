# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 21:31:20 2024

@author: Priyanka
"""

import pandas as pd
ent=pd.read_csv("C:\Data Set\Entertainment.csv",encoding='utf8')
ent.shape
#you will get 12294X7 matrix
ent.columns
ent.Category
#Here we are considering only Category
from sklearn.feature_extraction.text import TfidfVectorizer
#This is term frequency inverse document
#Each row is treated as document
tfidf=TfidfVectorizer(stop_words='english')
#It is going to create TfidfVectorizer to seperate all stop words.It is going to seperate
#out all words from the row
#Now let us check is there any null value 
ent['Category'].isnull().sum()
#There are 0 null values
#Suppose one movie has got genre Drama,Romance,..There may be many empty spaces
#so let us impute these empty spaces,general is like simple imputer

#Now let us create tfidf_matrix
tfidf_matrix=tfidf.fit_transform(ent.Category)
tfidf_matrix.shape
#You will get 51X34
#It has created sparse matrix,it means that we have 34 categories of DVD
#on this particular matrix ,we want to do item based recommendation,if a user has
#watched Gadar,then you can recommend Shershah movie
from sklearn.metrics.pairwise import linear_kernel
#This is for measuring similarity
cosine_sim_matrix=linear_kernel(tfidf_matrix,tfidf_matrix)
#each element of tfidf_matrix is compared with each element of tfidf_matrix only
#ouput will be similarity matrix of size 51X51 size
#Here in cosine_sim_matrix ,there are no movie names only index are provided
#We will try to map movie name with movie index given
#for that purpose custom function is written
ent_index=pd.Series(ent.index,index=ent['Titles']).drop_duplicates()
#We are converting anime_index into series format,we want index and corresponding name
ent_id=ent_index['Casino (1995)']
ent_id
def get_recommendations(Name,topN):
    #topN=10
    ent_id=ent_index[Name]
    
    #We want to cature whole row of given movie name,its score and column id
    #For that purpose we are applying cosine_sim_matrix to enumerate function
    #Enumerate function create a object ,which we need to create in list form
    #we are using enumerate function ,what enumerate does,suppose we have given
    #(2,10,15,18),if we apply to enumerate then it will create a list
    #(0,2,   1,10,  3,15,  4,18)
    cosine_scores=list(enumerate(cosine_sim_matrix[ent_id]))
    #The cosine scores captured,we want to arrange in descending order so that
    #we can recomment top 10 based on highest similarity i.e.score
    #if we will check the cosine score, it comprises of index:cosine score
    #x[0]=index and x[1] is cosine score
    #we want arrange tupples according to decreasing order of the score not index
    # Sorting the cosine_similarity scores based on scores i.e.x[1]
    cosine_scores = sorted(cosine_scores, key=lambda x:x[1], reverse = True)
    # Get the scores of top N most similar movies 
    #To capture TopN movies,you need to give topN+1
    cosine_scores_N = cosine_scores[0: topN+1]
    #getting the movie index
    ent_idx=[i[0] for i in cosine_scores_N]
    #getting cosine score
    ent_scores=[i[1]for i in cosine_scores_N]
    #We are going to use this information to create a dataframe
    #create a empty dataframe
    ent_similar_show=pd.DataFrame(columns=['Titles','score'])
    #assign anime_idx to name column
    ent_similar_show['Titles']=ent.loc[ent_idx,'Titles']
    #assign score to score column
    ent_similar_show['score']=ent_scores
    # while assigning values,it is by default capturing original index of the movie
    # we want to reset the index
    ent_similar_show.reset_index(inplace=True)
    print(ent_similar_show)
#Enter your anime and number of animes to be recommended
get_recommendations('Heat (1995)',topN=10)
    