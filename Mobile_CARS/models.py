#!/usr/bin/env python
# coding: utf-8

# # Running app recommendations

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


df = pd.read_csv('MDF_final.csv')
df = df.drop_duplicates()
df.reset_index(drop=True, inplace=True)

item_labels = [i for i in list(df.columns) if i.find("category") == 0] # labels that describe an item
context_labels = list(set(df.iloc[:, 3:]) - set(item_labels)) # takes all the columns after user, item rating and remove item labels

n_users = df.user.nunique()
n_items = df.item.nunique()
n_contexts = len(context_labels)
    
print(f"rating with value 1: {df[df.rating == 1]['rating'].count() * 100 / len(df)} %")
print(f"users: {n_users} \t items: {n_items} \t rating: {len(df)} \t items_features: {len(item_labels)} \t contexts_features: {n_contexts} \t ")


# In[ ]:


param_grid = {
    'n_users': n_users,
    'n_items': n_items,
    'n_contexts': n_contexts,
    'epochs': [5, 10, 15, 20], 
    'batch_size': [64, 128, 256],
    'learn_rate': [0.0001, 0.001, 0.01]
}


# ## Matrix factorization

# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(df[['user', 'item']], df['rating'], test_size=0.20, random_state=42)
mf = rs_models.MF
kgs = KerasGridSearch(mf, param_grid, monitor='val_auc', greater_is_better=True, tuner_verbose=0)
kgs.search([x_train.user, x_train.item], y_train, validation_data=([x_test.user, x_test.item], y_test))
print(f'MF best AUC: {kgs.best_score} using {kgs.best_params}')


# ## NeuMF

# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(df[['user', 'item']], df['rating'], test_size=0.20, random_state=42)
neumf = rs_models.NeuMF
kgs = KerasGridSearch(neumf, param_grid, monitor='val_auc', greater_is_better=True, tuner_verbose=0)
kgs.search([x_train.user, x_train.item], y_train, validation_data=([x_test.user, x_test.item], y_test))
print(f'NeuMF best AUC: {kgs.best_score} using {kgs.best_params}')


# ## ECAM NeuMF

# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(df[['user', 'item'] + context_labels], df['rating'], test_size=0.20, random_state=42)
ecam_neumf = rs_models.ECAM_NeuMF
kgs = KerasGridSearch(ecam_neumf, param_grid, monitor='val_auc', greater_is_better=True, tuner_verbose=0)
kgs.search([x_train.user, x_train.item, x_train[context_labels]], y_train, validation_data=([x_test.user, x_test.item, x_test[context_labels]], y_test))
print(f'NeuMF best AUC: {kgs.best_score} using {kgs.best_params}')


# ## Classifier

# In[ ]:


x = df[item_labels + context_labels]
y = df['rating']

ff_net = KerasClassifier(build_fn=rs_models.mobile_model, verbose=False)

# define hyperparameters to tune
learn_rate = [0.0005, 0.001, 0.005]
batch_size = [64, 128, 256]
epochs = [5, 10, 15, 20, 30]
neurons = [100, 200, 400]
layers = [3, 4, 5]

# Make a dictionary of the grid search parameters
param_grid = dict(learn_rate=learn_rate, batch_size=batch_size, epochs=epochs, neurons=neurons, layers=layers)

# create and fit gridsearch
grid = GridSearchCV(estimator=ff_net, scoring=['accuracy', 'roc_auc'], refit='roc_auc', param_grid=param_grid, 
                    cv=KFold(shuffle=True, n_splits=2, random_state=42), verbose=True)
grid_results = grid.fit(x, y)

mean_accuracy = grid_results.cv_results_['mean_test_accuracy']
mean_auc = grid_results.cv_results_['mean_test_roc_auc']
params = grid_results.cv_results_['params']

print(f'FFnet best AUC: {grid_results.best_score_} using {grid_results.best_params_}')



