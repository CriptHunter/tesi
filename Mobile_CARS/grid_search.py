#!/usr/bin/env python
# coding: utf-8

# # Grid search MDF dataset

# In[ ]:


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.utils import plot_model
from keras.models import Sequential
from keras.layers import Dense, Input, Embedding, Flatten, Concatenate, Lambda
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.model_selection import GridSearchCV, KFold
import rs_models
from kerashypetune import KerasGridSearch

pd.options.display.max_columns = 1000
import warnings
warnings.filterwarnings("ignore")


# ## Open Dataset

# In[ ]:


dataset = 'frappe'

if dataset == 'mdf':
    df = pd.read_csv('MDF_final.csv')
    df_mf = pd.read_csv('MDF_matrix_factorization.csv')
    df = df.drop_duplicates()
    df.user = pd.factorize(df.user)[0] # make sure that user and item IDs start from zero
    df.item = pd.factorize(df.item)[0]
    # df = df[df.item != 2]
    # df = df.drop(['place_type_food_and_drink', 'place_type_health', 'place_type_home', 'place_type_lodging','place_type_outdoors', 'place_type_point_of_interest_establishment','place_type_public_transport_station', 'place_type_school','place_type_service', 'place_type_store', 'place_type_workplace'], axis = 1)
    df.reset_index(drop=True, inplace=True)
    item_labels = [i for i in list(df.columns) if i.find("category") == 0] # labels that describe an item
    user_labels = []
    context_labels = list(set(df.iloc[:, 3:]) - set(item_labels)) # takes all the columns after user, item rating and remove item labels

elif dataset == 'frappe':
    df = pd.read_csv('frappe_final.csv')
    #df_mf = pd.read_csv('frappe_matrix_factorization.csv')
    item_labels = list(df.columns[27:54])
    user_labels = list(df.columns[54:])
    context_labels = list(df.columns[3:27])


# In[ ]:


n_users = df.user.nunique()
n_items = df.item.nunique()
n_contexts = len(context_labels)

print(f"rating with value 1: {df[df.rating == 1]['rating'].count() * 100 / len(df)} %")
print(f"users: {n_users} \t items: {n_items} \t rating: {len(df)} \t user_features: {len(user_labels)} \t items_features: {len(item_labels)} \t contexts_features: {n_contexts} \t ")


# ## Grid search

# In[ ]:


param_grid = {
    'n_users': n_users,
    'n_items': n_items,
    'n_contexts': n_contexts,
    'epochs': [5, 10, 15, 20], 
    'batch_size': [64, 128, 256],
    'learn_rate': [0.0001, 0.001, 0.01]
}
open('grid_search_result.txt', 'w').close()



print("grid search on NeuMF...")
x_train, x_test, y_train, y_test = train_test_split(df[['user', 'item']], df['rating'], test_size=0.20, random_state=42)
neumf = rs_models.NeuMF
kgs = KerasGridSearch(neumf, param_grid, monitor='val_auc', greater_is_better=True, tuner_verbose=0)
kgs.search([x_train.user, x_train.item], y_train, validation_data=([x_test.user, x_test.item], y_test))
print(f'NeuMF best AUC: {kgs.best_score} using {kgs.best_params}', file=open("grid_search_result.txt", "a"))
print("Done!")


# ### ECAM NeuMF

# In[ ]:


print("grid search on ECAM NeuMF...")
x_train, x_test, y_train, y_test = train_test_split(df[['user', 'item'] + context_labels], df['rating'], test_size=0.20, random_state=42)
ecam_neumf = rs_models.ECAM_NeuMF
kgs = KerasGridSearch(ecam_neumf, param_grid, monitor='val_auc', greater_is_better=True, tuner_verbose=0)
kgs.search([x_train.user, x_train.item, x_train[context_labels]], y_train, validation_data=([x_test.user, x_test.item, x_test[context_labels]], y_test))
print(f'ECAM NeuMF best AUC: {kgs.best_score} using {kgs.best_params}', file=open("grid_search_result.txt", "a"))
print("Done!")


# ### Classifier

# In[ ]:


print("grid search on feed-forward network...")

x = df[item_labels + context_labels + user_labels]
y = df['rating']

ff_net = KerasClassifier(build_fn=rs_models.mobile_model, verbose=False)

param_grid = {
    'learn_rate': [0.0001, 0.001, 0.005, 0.01],
    'batch_size': [64, 128, 256],
    'epochs': [5, 10, 15, 20, 30], 
    'layers': [3],
    'neurons': [200]
}

# create and fit gridsearch
grid = GridSearchCV(estimator=ff_net, scoring=['accuracy', 'roc_auc'], refit='roc_auc', param_grid=param_grid, 
                    cv=KFold(shuffle=True, n_splits=2, random_state=42), verbose=True)
grid_results = grid.fit(x, y)

mean_accuracy = grid_results.cv_results_['mean_test_accuracy']
mean_auc = grid_results.cv_results_['mean_test_roc_auc']
params = grid_results.cv_results_['params']

print(f'FFnet best AUC: {grid_results.best_score_} using {grid_results.best_params_}', file=open("grid_search_result.txt", "a"))
print("Done!")

