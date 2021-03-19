import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Input, Embedding, Flatten, Concatenate, Lambda
from keras.optimizers import Adam
from scipy.sparse import coo_matrix, csr_matrix
from implicit.evaluation import AUC_at_k, precision_at_k, train_test_split
from implicit.als import AlternatingLeastSquares
import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm


def MF(param):
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name = 'user_input')
    item_input = Input(shape=(1,), dtype='int32', name = 'item_input')

    MF_Embedding_User = Embedding(input_dim = param['n_users']+1, output_dim = 50, name = 'user_embedding', input_length=1)
    MF_Embedding_Item = Embedding(input_dim = param['n_items']+1, output_dim = 50, name = 'item_embedding', input_length=1)   
    
    # flatten an embedding vector
    user_latent = Flatten()(MF_Embedding_User(user_input))
    item_latent = Flatten()(MF_Embedding_Item(item_input))
    
    # Element-wise product of user and item embeddings 
    predict_vector = tf.keras.layers.Dot(axes=1)([user_latent, item_latent])
    
    # Final prediction layer
    prediction = Dense(1, activation='sigmoid', name = 'prediction')(predict_vector)
    
    model = keras.Model([user_input, item_input], prediction)
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=param['learn_rate']), metrics=['accuracy', 'AUC'])

    return model


def NeuMF(param):
    user_input = Input(shape=(1,), dtype='int32', name = 'user_input')
    item_input = Input(shape=(1,), dtype='int32', name = 'item_input')
    
    # Embedding layer
    MF_Embedding_User = Embedding(input_dim = param['n_users']+1, output_dim = 50, name = 'mf_embedding_user', input_length=1)
    MF_Embedding_Item = Embedding(input_dim = param['n_items']+1, output_dim = 50, name = 'mf_embedding_item', input_length=1)   

    MLP_Embedding_User = Embedding(input_dim = param['n_users']+1, output_dim = 50, name = "mlp_embedding_user", input_length=1)
    MLP_Embedding_Item = Embedding(input_dim = param['n_items']+1, output_dim = 50, name = 'mlp_embedding_item', input_length=1)   
    
    # MF part
    mf_user_latent = Flatten()(MF_Embedding_User(user_input))
    mf_item_latent = Flatten()(MF_Embedding_Item(item_input))
    mf_vector = tf.keras.layers.Multiply()([mf_user_latent, mf_item_latent])

    # MLP part 
    mlp_user_latent = Flatten()(MLP_Embedding_User(user_input))
    mlp_item_latent = Flatten()(MLP_Embedding_Item(item_input))
    mlp_vector = Concatenate()([mlp_user_latent, mlp_item_latent])
    
    # dense layers
    dense = Dense(200, name='fully_connected_1')(mlp_vector)
    dense_2 = Dense(100, name='fully_connected_2')(dense)
    dense_3 = Dense(50, name='fully_connected_3')(dense_2)
    dense_4 = Dense(25, name='fully_connected_3')(dense_3)

    # Concatenate MF and MLP parts
    predict_vector = Concatenate()([mf_vector, dense_3])
    
    # Final prediction layer
    prediction = Dense(1, activation='sigmoid', name = "prediction")(predict_vector)
    
    model = keras.Model([user_input, item_input], prediction)
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=param['learn_rate']), metrics=['accuracy', 'AUC', 'Precision', 'Recall'])
    
    return model


def ECAM_NeuMF(param):
    user_input = Input(shape=(1,), dtype='int32', name = 'user_input')
    item_input = Input(shape=(1,), dtype='int32', name = 'item_input')
    context_input = Input(shape=(param['n_contexts'], ), name='context_input')
    
    # Embedding layer
    MF_Embedding_User = Embedding(input_dim = param['n_users']+1, output_dim = 50, name = 'mf_embedding_user', input_length=1)
    MF_Embedding_Item = Embedding(input_dim = param['n_items']+1, output_dim = 50, name = 'mf_embedding_item', input_length=1)   

    MLP_Embedding_User = Embedding(input_dim = param['n_users']+1, output_dim = 50, name = "mlp_embedding_user", input_length=1)
    MLP_Embedding_Item = Embedding(input_dim = param['n_items']+1, output_dim = 50, name = 'mlp_embedding_item', input_length=1)   
    
    # MF part
    mf_user_latent = Flatten()(MF_Embedding_User(user_input))
    mf_item_latent = Flatten()(MF_Embedding_Item(item_input))
    mf_vector = tf.keras.layers.Multiply()([mf_user_latent, mf_item_latent])

    # MLP part 
    mlp_user_latent = Flatten()(MLP_Embedding_User(user_input))
    mlp_item_latent = Flatten()(MLP_Embedding_Item(item_input))
    mlp_vector = Concatenate()([mlp_user_latent, mlp_item_latent, context_input])
    
    # dense layers
    dense = Dense(200, name='fully_connected_1')(mlp_vector)
    dense_2 = Dense(100, name='fully_connected_2')(dense)
    dense_3 = Dense(50, name='fully_connected_3')(dense_2)

    # Concatenate MF and MLP parts
    predict_vector = Concatenate()([mf_vector, dense_3])
    
    # Final prediction layer
    prediction = Dense(1, activation='sigmoid', name = "prediction")(predict_vector)
    
    model = keras.Model([user_input, item_input, context_input], prediction)
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=param['learn_rate']), metrics=['accuracy', 'AUC', 'Precision', 'Recall'])
    
    return model


def mobile_model(neurons, layers, learn_rate):
    model = Sequential()
    for x in range(layers):
        model.add(Dense(neurons, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=learn_rate))
    return model


def kfold_split(df, x, y, n_splits=5):
    """
    Generator that split a dataframe in train and test. Every yield returns a new split until n_splits iterations are done
    
    Parameters
    ----------
    df : pandas dataframe
        Dataframe to split
    x : list of strings
        x labels
    y: list of strings
        y labels
    n_splits : int
        how many splits the generator can yield
        
    Returns
    -------
    (x_train, y_train, x_test, y_test) : panda dataframes
        four pandas dataframe for train and test
    """
    kf = KFold(n_splits=n_splits, random_state=42, shuffle=True)

    for train_index, test_index in kf.split(df[x], df[y]):
        x_train, x_test = df[x].loc[train_index, :], df[x].loc[test_index, :]
        y_train, y_test = df[y].loc[train_index], df[y].loc[test_index]
        yield x_train, y_train, x_test, y_test


def kfold_train(model, param, df, context_labels=[], n_splits=2):
    """
    Train a model with two or more input using Kfold
    
    Parameters
    ----------
    model : function
        A function that return a compiled keras model
    param : dictionary
        A dictionary that contains model parameters like learning rate, epochs, batch size
    df : pandas dataframe
        Data on which the model will be trained
        
    Returns
    -------
    results : numpy array
        An array containing metrics averages obtained by evaluating the model on n_splits
    """
    x_labels = ['user', 'item'] 
    y_labels = 'rating'
    df = df.sample(frac=1) # shuffle dataset
    kfold = kfold_split(df, x_labels+context_labels, y_labels, n_splits) # generator that returns training and test index
    idx = 0

    for x_train, y_train, x_test, y_test in kfold:
        net = model(param)

        input_list = [x_train[e] for e in x_labels] # split user, item input
        input_list = [input_list + [x_train[context_labels]] if context_labels else input_list] # add context if it's available
        net.fit(input_list, y_train, epochs=param['epochs'], batch_size=param['batch_size'], verbose=False)

        input_list = [x_test[e] for e in x_labels] # same split for test values
        input_list = [input_list + [x_test[context_labels]] if context_labels else input_list]
        if idx == 0: # if it is the first fold, create results array
            results = np.array(net.evaluate(input_list, y_test, batch_size=512, verbose=False))
        else: # else add new results to array
            results = np.add(results, net.evaluate(input_list, y_test, batch_size=512, verbose=False))
        idx = idx + 1
    return results/idx


def train_mf(df, factors=128, regularization=5, iterations=50, n_splits=10, k=10):
    """
    Train ALS matrix factorization model from implicit library on n splits
    
    Parameters
    ----------
    df : pandas dataframe
        user, item, rating data
    factors: int
        The number of latent factors to compute
    regularization : float
        The regularization factor to use
    iterations : int
        The number of ALS iterations to use when fitting data
    n_splits: int
        the number of train/test
        
    Returns
    -------
    results : numpy array
        An array containing metrics averages obtained by evaluating the model on n_splits
    """
    ratings = coo_matrix((df['rating'].astype(np.float32),
                         (df['item'],
                          df['user']))).tocsr()
    results = np.zeros(2)
    for x in range(n_splits):
        train, test = train_test_split(ratings, train_percentage=0.8)
        model = AlternatingLeastSquares(factors=factors, regularization=regularization, iterations=iterations, calculate_training_loss=True)
        model.fit(train, show_progress=False)

        auc = AUC_at_k(model, train.T.tocsr(), test.T.tocsr(), K=k, show_progress=False, num_threads=4)
        precision = precision_at_k(model, train.T.tocsr(), test.T.tocsr(), K=k, show_progress=False, num_threads=4)
        results = np.add(results, [auc, precision])
    return results/(x+1)



def mf_grid_search(df, factors, regularization, iterations, n_splits, monitor, k):
    """
    Grid search for ALS matrix factorization from implicit library
    
    Parameters
    ----------
    df : pandas dataframe
        user, item, rating data
    factors: list of int
        The number of latent factors to compute
    regularization : list of float
        The regularization factor to use
    iterations : list of int
        The number of ALS iterations to use when fitting data
    n_splits: int
        the number of train/test
    monitor: string
        Metric that is used to select the best parameters combination
        
    Returns
    -------
    best_parameters : numpy array
        An array containing the best parameters combination (factors, regularization, iterations)
    """
    combinations = []  # possible combination of grid search parameters   
    for x in factors:  # fill combinations list         
        for y in regularization: 
            for z in iterations:
                combinations.append((x, y, z)) 
             
     # select which metric is used to select the best parameters combination
    if monitor == 'auc':
        idx = 0
    elif monitor == 'precision':
        idx = 1
    else:
        raise Exception("Unknown metric, possible metrics are: auc, precision")


    best_metric = 0
    best_parameters = []
    for factors, regularization, iterations in tqdm(combinations):
        results = train_mf(df, factors=factors, regularization=regularization, 
                           iterations=iterations, n_splits=n_splits, k=k) # train model on n splits
        if results[idx] > best_metric:
            best_parameters = [factors, regularization, iterations]
            best_metric = results[idx]
    return best_parameters