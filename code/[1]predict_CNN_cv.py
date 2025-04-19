import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import tensorflow.keras.backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, load_model, Sequential
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import multilabel_confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
import os
from sklearn.decomposition import PCA
from tensorflow.keras.callbacks import EarlyStopping
from keras.losses import MeanSquaredError
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import pickle

path = '../'
path_BD = 'BD_1/'
path_models = 'models/'
path_img = 'im/'

file = ['X_EfficientNetB7_PCA.pkl']

sizes = ((11,16))

i =0
file =file[i]
sizes = sizes[i]

import time
time_start = time.time()

def load(filepath, keys):
  with open(filepath, 'rb') as filepath:
      arquivos = pickle.load(filepath)
      
  retorno = list()
  for i in range(len(keys)):
      retorno.append(arquivos[keys[i]])

  return retorno

def load_database(file):

  filepath = path + path_BD + file
  key = ['X'] # ["X_EfficientNetB7_encoded"]
  X = load(filepath, key)[0]


  filepath = path + path_BD + "y_pack_3apr.pkl"
  key = ["vol", "porosity", "visc_ratio"]
  y, y_porosity, y_visc_ratio  = load(filepath, key)

  return X, y, y_porosity, y_visc_ratio

X, y, y_porosity, y_visc_ratio = load_database(file)

X = np.concatenate((X, X, X), axis = 0)
aux = np.concatenate((y_porosity[:, np.newaxis], y_visc_ratio[:, np.newaxis]), axis = 1)

def define_model():

  inputs = tf.keras.Input(shape=(sizes[0], sizes[1], X.shape[-1]))
  input_aux = tf.keras.Input(shape=(2))

  act = 'relu'
  act_2 = 'tanh'

  cnn_nneurons = 50

  x = inputs

  x  = Conv2D(cnn_nneurons, (2, 2), padding = 'same', activation = act)(x)
  x1 = Conv2D(cnn_nneurons, (3, 3), padding = 'same', activation = act)(x)
  x2 = Conv2D(cnn_nneurons, (5, 5), padding = 'same', activation = act)(x)
 
  for _ in range(1):
    x  = Conv2D(cnn_nneurons, (2, 2), padding = 'same', activation = act)(x)
    x1 = Conv2D(cnn_nneurons, (3, 3), padding = 'same', activation = act)(x1)
    x2 = Conv2D(cnn_nneurons, (5, 5), padding = 'same', activation = act)(x2)

  x = Conv2D(cnn_nneurons, (1, 1), padding = 'same', activation = act)(tf.concat([x,x1,x2], axis=3))

  x = MaxPooling2D(pool_size=(2, 2), strides=None, padding="valid")(x)
  
  x  = Conv2D(cnn_nneurons, (2, 2), padding = 'same', activation = act)(x)
  x1 = Conv2D(cnn_nneurons, (3, 3), padding = 'same', activation = act)(x)
 
  for _ in range(2):
    x  = Conv2D(cnn_nneurons, (2, 2), padding = 'same', activation = act)(x)
    x1 = Conv2D(cnn_nneurons, (3, 3), padding = 'same', activation = act)(x1)

  x = Conv2D(100, (1, 1), padding = 'same', activation = act)(tf.concat([x,x1], axis=3))

  x = Conv2D(50, (x.shape[1], x.shape[2]), padding = 'valid', activation = 'sigmoid')(x)

  naux = 20
  x_aux = Dense(naux, activation='sigmoid')(input_aux)
  for _ in range(1):
    x_aux = Dense(naux, activation='sigmoid')(x_aux)

  x = Conv2D(50, (1, 1), padding = 'valid', activation = None)(tf.concat([x, tf.reshape(x_aux, (-1, 1, 1, naux))], axis=-1))
    
  x = tf.squeeze(x, axis=[1, 2])
  
  output = x

  model = Model(inputs = [inputs, input_aux], outputs = [output])
  model.summary()

  # Compile the model
  lr = tf.keras.optimizers.schedules.ExponentialDecay(
      initial_learning_rate=.3e-3,
      decay_steps=250,
      decay_rate=0.98,
      staircase=False)

  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                 loss='mse',  
                  metrics=['mse']
                  )

  return model

def train_and_get_metrics(data_train, data_test, data_val, epochs, filename = '../models/cnn.h5', batch_size = 128):
    
    X_train, y_train, aux_train = data_train
    X_test, y_test, aux_test = data_test
    X_val, y_val, aux_val = data_val

    model = define_model()

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath= filename,
        save_weights_only=False,
        monitor= 'val_loss',
        save_best_only=True)

    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=250,
        restore_best_weights=True
    )

    loss = model.fit((X_train, aux_train), y_train, epochs = epochs, batch_size = batch_size, callbacks = [early_stopping_callback],
                validation_data = ((X_val, aux_val), y_val))

    tf.keras.saving.save_model(model, filename, overwrite=True)

    y_predicted = model((X_test, aux_test))
    y_predicted = y_predicted.numpy()
    
    def get_metrics(y_predicted, y_test):
        R2 = r2_score(y_test, y_predicted)
        MSE = mean_squared_error(y_predicted, y_test)
        mean_error_oil_prediction = np.mean(((np.absolute(y_predicted[:,-1]-y_test[:,-1]))/(np.absolute(y_test[:,-1]))))
        return R2, MSE, mean_error_oil_prediction
    
    R2, MSE, mean_error_oil_prediction = get_metrics(y_predicted, y_test)
    return loss, R2, MSE, mean_error_oil_prediction
    
def cross_validation(X, y, aux, n_cv, epochs=5000):#, parametros=(2, 10, 20)):
    n_samples = X.shape[0]
    n_samples_group = int(np.ceil(n_samples/n_cv))

    id_sample = np.arange(0,n_samples,1)
    np.random.shuffle(id_sample)

    X = X[id_sample]
    y = y[id_sample]
    aux = aux[id_sample]
    
    filepath = './don_data.pkl'
    with open(filepath, 'rb') as filepath:
        arquivos = pickle.load(filepath)
        
    X = arquivos['X']
    y = arquivos['y']
    aux = arquivos['aux']
    del arquivos

    X_groups =[]
    y_groups =[]
    aux_groups =[]
    metrics =[]

    for i in range(n_cv):
        X_groups.append(X[i*n_samples_group:(i+1)*n_samples_group])
        y_groups.append(y[i*n_samples_group:(i+1)*n_samples_group])
        aux_groups.append(aux[i*n_samples_group:(i+1)*n_samples_group])

    for i in range(n_cv):
        print(f'==fold {i+1}/{n_cv}==')
        X_test = X_groups[i]
        y_test = y_groups[i]
        aux_test = aux_groups[i]

        groups = np.arange(n_cv)
        groups = np.delete(groups, i)
        groups = groups.astype(int)
        X_train = []
        y_train = []
        aux_train = []
        for group in groups: 
            X_train.append(X_groups[group])
            y_train.append(y_groups[group])
            aux_train.append(aux_groups[group])
        X_train = np.vstack(X_train)
        y_train = np.vstack(y_train)
        aux_train = np.vstack(aux_train)

        X_train, X_val, y_train, y_val, aux_train, aux_val = train_test_split(X_train, y_train, aux_train, test_size=0.2, random_state=1)

        data_train= (X_train, y_train, aux_train)
        data_test = (X_test, y_test, aux_test)
        data_val  = (X_val, y_val, aux_val)

                                  #train_and_get_metrics(data_train, data_test, data_val,units, nneurons, window, epochs, model_name, batch_size):
        loss, MSE, R2, error_oil = train_and_get_metrics(data_train, data_test, data_val, epochs, '../models/cnn_' + str(i) + '.h5', 128)
        tf.keras.backend.clear_session()

        metrics.append([loss, MSE, R2, error_oil])

    return metrics

metricas = cross_validation(X, y, aux, n_cv=10, epochs=5000)

filepath = './metricas_cnn.pkl'
with open(filepath,"wb") as filepath:
	pickle.dump({"metricas": metricas }, filepath)
 
time_stop = time.time()
print(time_stop-time_start)
