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

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score


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
    kfold = kfold_split(df, x_labels+context_labels, y_labels, n_splits) # generator that returns train and test dataframes
    idx = 0

    for x_train, y_train, x_test, y_test in kfold:
        net = model(param)

        input_list = [x_train[e] for e in x_labels] # split (user, item) input into two separate inputs
        input_list = [input_list + [x_train[context_labels]] if context_labels else input_list] # add context if it's available
        net.fit(input_list, y_train, epochs=param['epochs'], batch_size=param['batch_size'], verbose=False) # train network

        input_list = [x_test[e] for e in x_labels] # same split for test values
        input_list = [input_list + [x_test[context_labels]] if context_labels else input_list]
        if idx == 0: # if it is the first fold, create results array
            results = np.array(net.evaluate(input_list, y_test, batch_size=512, verbose=False))
        else: # else add new results to array
            results = np.add(results, net.evaluate(input_list, y_test, batch_size=512, verbose=False))
        idx = idx + 1
    return results/idx # return results average


def mf_AUC(model, train, test):
    '''
    Parameters
    ----------
    model : implicit
        Implicit library MF model
    train : numpy array
        (item, user) matrix used to train the model
    test : numpy array
        (item, user) matrix with new interactions
        
    Returns
    -------
    results : float
        mean AUC between users AUC
        
        
    - selezionare solo utenti che hanno almeno un rating alterato (uno o più rating con valore 1 sono stati messi a 0 nel train set)
    - per ogni utente alterato:
        - `idx` = indici nel train set per cui rating == 0, in questo modo evito di calcolare l'AUC su rating con valore 1 nel train set
        - `pred` = valori dalla matrice predicted con indici in `idx`
        - `actual` = valori dal test set con indici `idx`
        - calcolare AUC per l'utente tra `actual` e `pred`
    - fare media AUC
    '''
    altered_users = [] # list of user that has at least one rating with value 1 hidden from the train set
    train_dense = np.array(train.todense())
    test_dense = np.array(test.todense())
    pred_dense = np.dot(model.item_factors, model.user_factors.T)
    
    # fill altered_user list
    for user in range(np.shape(test_dense)[1]):
        test_user = test_dense[:, user]
        if 1 in test_user: # check if user has items altered
            altered_users.append(user)

    auc = 0

    for user in altered_users:
        train_user = train_dense[:, user] # get ratings in train set for one user
        zero_idx = np.where(train_user == 0) # find where ratings is zero

        # get predicted values
        user_vec = pred_dense[:, user]
        pred_user = user_vec[zero_idx]

        # get actual value in test set
        user_vec2 = test_dense[:, user]
        actual_user = user_vec2[zero_idx]

        auc_user = roc_auc_score(actual_user, pred_user)
        auc = auc + auc_user


    return auc/len(altered_users)