# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
#import numpy as np
userCol = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
users = pd.read_csv('Data/ml-100k/u.user', sep='|', names=userCol, encoding='latin-1')
ratingsCol = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv('Data/ml-100k/u.data', sep='\t', names=ratingsCol, encoding='latin-1')



movieCol = ['movie_id', 'title','release', 'date_of_release', 'url', 'unknown', 'Action', 'Adventure','Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy','Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western' ]
movies = pd.read_csv('Data/ml-100k/u.item', sep='|',names=movieCol, encoding='latin-1')

TrainRatings = pd.read_csv('Data/ml-100k/ua.base', sep='\t', names=ratingsCol, encoding='latin-1')
TestRatings = pd.read_csv('Data/ml-100k/ua.test', sep='\t', names=ratingsCol, encoding='latin-1')

#TestRatings.shape
#TrainRatings.shape

import graphlab as gl
trainData = gl.SFrame(TrainRatings)
testData = gl.SFrame(TestRatings)

popModel = gl.popularity_recommender.create(trainData, user_id='user_id', item_id='movie_id', target='rating')
recommendedMovies = popModel.recommend(users=range(1,6),k=5)
recommendedMovies.print_rows(num_rows=25)

itemItemCF = gl.item_similarity_recommender.create(trainData, user_id='user_id', item_id='movie_id', target='rating', )
recommendedMovies2 = itemItemCF.recommend(users=range(1,6),k=5)
recommendedMovies2.print_rows(num_rows=25)

performanceComparasion = gl.compare(testData, [popModel, itemItemCF])
gl.show_comparison(performanceComparasion,[popModel, itemItemCF])