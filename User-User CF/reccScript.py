# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 23:39:09 2017

@author: hp
"""

import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data/ml-100k/u.data', sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])
#X = dataset.iloc[:, [0,1]].values
#y = dataset.iloc[:, [2] ].values

n_users = dataset.user_id.unique().shape[0]
n_movies = dataset.movie_id.unique().shape[0]
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test = train_test_split(dataset, test_size = 0.2, random_state = 0)

X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)

training = np.zeros((n_users, n_movies))
for line in X_train.itertuples():
    training[line[1]-1, line[2]-1] = line[3]

testing = np.zeros((n_users, n_movies))
for line in X_test.itertuples():
    testing[line[1]-1, line[2]-1] = line[3]

from sklearn.metrics.pairwise import pairwise_distances
user_similarity = pairwise_distances(training, metric='cosine')
item_similarity = pairwise_distances(training, metric='cosine')

def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        #You use np.newaxis so that mean_user_rating has same format as ratings
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = np.dot(ratings.transpose(), similarity) / np.array([np.abs(similarity).sum(axis=1)])
        pred = pred.transpose()
    return pred
item_prediction = predict(training, item_similarity, type='item')
user_prediction = predict(training, user_similarity, type='user')

from sklearn.metrics import mean_squared_error
from math import sqrt
def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))
print('User-based CF RMSE: ' + str(rmse(user_prediction, testing)))
print('Item-based CF RMSE: ' + str(rmse(item_prediction, testing)))


sparsity=round(1.0-len(dataset)/float(n_users*n_movies),3)
print('The sparsity level of MovieLens100K is ' +  str(sparsity*100) + '%')

import scipy.sparse as sp
from scipy.sparse.linalg import svds

#get SVD components from train matrix. Choose k.
u, s, vt = svds(training, k = 20)
s_diag_matrix=np.diag(s)
X_pred = np.dot(np.dot(u, s_diag_matrix), vt)
print('User-based CF MSE: ' + str(rmse(X_pred, testing)))







