import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score


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
    
    # train and test matrices do not contains ratings as 0 and 1, but as how many time a user
    # consumed an item, so to calculate AUC that is a binary metric we binarize the matrices
    train_dense = (train_dense > 0).astype(np.int_)
    test_dense = (test_dense > 0).astype(np.int_)
    
    # scale predicted value between [0 1] because usually they are in a range like [-0.1, 1.1]
    min_max = MinMaxScaler()
    pred_dense = min_max.fit_transform(pred_dense)
    
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