# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 14:48:41 2017

@author: hp
"""

import pandas as pd
import numpy as np
userCol = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
users = pd.read_csv('Data/ml-100k/u.user', sep='|', names=userCol, encoding='latin-1')
ratingsCol = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv('Data/ml-100k/u.data', sep='\t', names=ratingsCol, encoding='latin-1')



movieCol = ['movie_id', 'title','release', 'date_of_release', 'url', 'unknown', 'Action', 'Adventure','Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy','Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western' ]
movies = pd.read_csv('Data/ml-100k/u.item', sep='|',names=movieCol, encoding='latin-1')


userDF = pd.DataFrame(users,columns=userCol)
movieDF = pd.DataFrame(movies, columns=movieCol)
ratingDF = pd.DataFrame(ratings, columns=ratingsCol)

ratingsAll = ratingDF.pivot(index = 'user_id', columns ='movie_id', values = 'rating').fillna(0)
ratingsAll.head()
'''
userDF.head()
movieDF.head()
ratingDF.head()
'''

#Normalisation as a prestep for svd

r = ratingsAll.as_matrix()
ratingMean = np.mean(r, axis = 1)
rDemeaned = r - ratingMean.reshape(-1, 1)

from scipy.sparse.linalg import svds
U, sigma, Vt = svds(rDemeaned, k = 50)
sigma = np.diag(sigma)

all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + ratingMean.reshape(-1, 1)
preds_df = pd.DataFrame(all_user_predicted_ratings, columns = ratingsAll.columns)

def recommend_movies(predictions_df, userID, movies_df, original_ratings_df, num_recommendations=5):
    
    # Get and sort the user's predictions
    user_row_number = userID - 1 # UserID starts at 1, not 0
    sorted_user_predictions = predictions_df.iloc[user_row_number].sort_values(ascending=False)
    
    # Get the user's data and merge in the movie information.
    user_data = original_ratings_df[original_ratings_df.user_id == (userID)]
    user_full = (user_data.merge(movies_df, how = 'left', left_on = 'movie_id', right_on = 'movie_id').
                     sort_values(['rating'], ascending=False)
                 )

    print 'User {0} has already rated {1} movies.'.format(userID, user_full.shape[0])
    print 'Recommending the highest {0} predicted ratings movies not already rated.'.format(num_recommendations)
    
    # Recommend the highest predicted rating movies that the user hasn't seen yet.
    recommendations = (movies_df[~movies_df['movie_id'].isin(user_full['movie_id'])].
         merge(pd.DataFrame(sorted_user_predictions).reset_index(), how = 'left',
               left_on = 'movie_id',
               right_on = 'movie_id').
         rename(columns = {user_row_number: 'Predictions'}).
         sort_values('Predictions', ascending = False).
                       iloc[:num_recommendations, :-1]
                      )

    return user_full, recommendations

already_rated, predictions = recommend_movies(preds_df, 837, movieDF, ratingDF, 10)

already_rated.head()
predictions.head()


