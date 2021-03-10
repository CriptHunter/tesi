import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Input, Embedding, Flatten, Concatenate, Lambda
from keras.optimizers import Adam

def MF(n_users, n_items, latent_dim=50):
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name = 'user_input')
    item_input = Input(shape=(1,), dtype='int32', name = 'item_input')

    MF_Embedding_User = Embedding(input_dim = n_users+1, output_dim = latent_dim, name = 'user_embedding', input_length=1)
    MF_Embedding_Item = Embedding(input_dim = n_items+1, output_dim = latent_dim, name = 'item_embedding', input_length=1)   
    
    # flatten an embedding vector
    user_latent = Flatten()(MF_Embedding_User(user_input))
    item_latent = Flatten()(MF_Embedding_Item(item_input))
    
    # Element-wise product of user and item embeddings 
    predict_vector = tf.keras.layers.Dot(axes=1)([user_latent, item_latent])
    
    # Final prediction layer
    prediction = Dense(1, activation='sigmoid', name = 'prediction')(predict_vector)
    
    model = keras.Model([user_input, item_input], prediction)
    model.compile(loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC()])

    return model


def NCF(num_users, num_items, latent_dim=50):
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name = 'user_input')
    item_input = Input(shape=(1,), dtype='int32', name = 'item_input')

    MLP_Embedding_User = Embedding(input_dim = num_users+1, output_dim = latent_dim, name = 'user_embedding', input_length=1)
    MLP_Embedding_Item = Embedding(input_dim = num_items+1, output_dim = latent_dim, name = 'item_embedding', input_length=1)   
    
    # flatten an embedding vector
    user_latent = Flatten()(MLP_Embedding_User(user_input))
    item_latent = Flatten()(MLP_Embedding_Item(item_input))
    
    # The 0-th layer is the concatenation of embedding layers
    concat = Concatenate(name='user_item')([user_latent, item_latent])
    
    # dense layers
    dense = Dense(200, name='fully_connected_1')(concat)
    dense_2 = Dense(100, name='fully_connected_2')(dense)
    dense_3 = Dense(50, name='fully_connected_3')(dense_2)
        
    # Final prediction layer
    prediction = Dense(1, activation='sigmoid', name = 'prediction')(dense_3)
    
    model = keras.Model([user_input, item_input], prediction)
    model.compile(loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC()])
    
    return model


def NeuMF(num_users, num_items, mf_dim=50, latent_dim=50):
    user_input = Input(shape=(1,), dtype='int32', name = 'user_input')
    item_input = Input(shape=(1,), dtype='int32', name = 'item_input')
    
    # Embedding layer
    MF_Embedding_User = Embedding(input_dim = num_users+1, output_dim = mf_dim, name = 'mf_embedding_user', input_length=1)
    MF_Embedding_Item = Embedding(input_dim = num_items+1, output_dim = mf_dim, name = 'mf_embedding_item', input_length=1)   

    MLP_Embedding_User = Embedding(input_dim = num_users+1, output_dim = latent_dim, name = "mlp_embedding_user", input_length=1)
    MLP_Embedding_Item = Embedding(input_dim = num_items+1, output_dim = latent_dim, name = 'mlp_embedding_item', input_length=1)   
    
    # MF part
    mf_user_latent = Flatten()(MF_Embedding_User(user_input))
    mf_item_latent = Flatten()(MF_Embedding_Item(item_input))
    mf_vector = tf.keras.layers.Dot(axes=1)([mf_user_latent, mf_item_latent])

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
    model.compile(loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC()])
    
    return model


def ECAM_NCF(num_users, num_items, num_contexts, latent_dim=50):
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name = 'user_input')
    item_input = Input(shape=(1,), dtype='int32', name = 'item_input')
    context_input = Input(shape=(num_contexts, ), name='context_input')

    MLP_Embedding_User = Embedding(input_dim = num_users+1, output_dim = latent_dim, name = 'user_embedding', input_length=1)
    MLP_Embedding_Item = Embedding(input_dim = num_items+1, output_dim = latent_dim, name = 'item_embedding', input_length=1)   
    
    # flatten an embedding vector
    user_latent = Flatten()(MLP_Embedding_User(user_input))
    item_latent = Flatten()(MLP_Embedding_Item(item_input))
    
    # The 0-th layer is the concatenation of embedding layers
    concat = Concatenate(name='user_item')([user_latent, item_latent, context_input])
    
    # dense layers
    dense = Dense(200, name='fully_connected_1')(concat)
    dense_2 = Dense(100, name='fully_connected_2')(dense)
    dense_3 = Dense(50, name='fully_connected_3')(dense_2)
        
    # Final prediction layer
    prediction = Dense(1, activation='sigmoid', name = 'prediction')(dense_3)
    
    model = keras.Model([user_input, item_input, context_input], prediction)
    model.compile(loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC()])
    
    return model


def ECAM_NeuMF(num_users, num_items, num_contexts, mf_dim=50, latent_dim=50):
    user_input = Input(shape=(1,), dtype='int32', name = 'user_input')
    item_input = Input(shape=(1,), dtype='int32', name = 'item_input')
    context_input = Input(shape=(num_contexts, ), name='context_input')
    
    # Embedding layer
    MF_Embedding_User = Embedding(input_dim = num_users+1, output_dim = mf_dim, name = 'mf_embedding_user', input_length=1)
    MF_Embedding_Item = Embedding(input_dim = num_items+1, output_dim = mf_dim, name = 'mf_embedding_item', input_length=1)   

    MLP_Embedding_User = Embedding(input_dim = num_users+1, output_dim = latent_dim, name = "mlp_embedding_user", input_length=1)
    MLP_Embedding_Item = Embedding(input_dim = num_items+1, output_dim = latent_dim, name = 'mlp_embedding_item', input_length=1)   
    
    # MF part
    mf_user_latent = Flatten()(MF_Embedding_User(user_input))
    mf_item_latent = Flatten()(MF_Embedding_Item(item_input))
    mf_vector = tf.keras.layers.Dot(axes=1)([mf_user_latent, mf_item_latent])

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
    model.compile(loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC()])
    
    return model


def mobile_model(neurons=200, learn_rate=0.001):
    model = Sequential()
    model.add(Dense(neurons, activation='relu'))
    model.add(Dense(neurons//2, activation='relu'))
    model.add(Dense(neurons//4, activation='relu'))
    model.add(Dense(neurons//8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=learn_rate))
    return model