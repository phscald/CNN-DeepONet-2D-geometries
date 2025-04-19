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

import time
time_start = time.time()

path = '../'
path_BD = 'BD_1/'
path_models = 'models/'

file = ['X_EfficientNetB7_PCA.pkl']

sizes = ((11,16))

i = 0 
file =file[i]
sizes = sizes[i]

def load(filepath, keys):
  with open(filepath, 'rb') as filepath:
      arquivos = pickle.load(filepath)
      
  retorno = list()
  for i in range(len(keys)):
      retorno.append(arquivos[keys[i]])

  return retorno

def load_database(file):

  filepath = path + path_BD + file
  key = ["X"]
  X = load(filepath, key)[0]


  filepath = path + path_BD + "y_pack_3apr.pkl"
  key = ["vol", "porosity", "visc_ratio"]#, "d_a"]
  y, y_porosity, y_visc_ratio  = load(filepath, key)

  return X, y, y_porosity, y_visc_ratio#, d_a

X, y, y_porosity, y_visc_ratio = load_database(file)

x0tam = X.shape[0]

y_porosity = y_porosity/.2
y_visc_ratio = y_visc_ratio/21
X = np.reshape(X, (x0tam*sizes[0]*sizes[1], 32))
x_norm = MinMaxScaler(feature_range=(-1, 1))
X = x_norm.fit_transform(X)
X = np.reshape(X, (x0tam,sizes[0],sizes[1], 32))


X = np.concatenate((X, X, X), axis = 0)

aux = np.concatenate((y_porosity[:, np.newaxis], y_visc_ratio[:, np.newaxis]), axis = 1)

def construct_data(full_size, window, X, y, aux):
  X_   = np.zeros(((full_size-window)*y.shape[0], X.shape[1],  X.shape[2], X.shape[3]))
  Xy  = np.zeros(((full_size-window)*y.shape[0], window))
  y_  = np.zeros(((full_size-window)*y.shape[0], 1))
  aux_time = np.zeros(((full_size-window)*y.shape[0], 1))
  aux_img  = np.zeros(((full_size-window)*y.shape[0], 2))

  for j in range(y.shape[0]):
    for i in range(window, full_size-1):
      X_[j*(full_size-window)+i-window,:,:] = X[j,:,:]
      Xy[j*(full_size-window)+i-window,:] = y[j, i-window:i]
      y_[j*(full_size-window)+i-window,:] = y[j , i]
      aux_img[j*(full_size-window)+i-window,:] = aux[j]
      aux_time[j*(full_size-window)+i-window,:] = np.array([(i-window)])

  return X_, Xy, y_, aux_img, aux_time

def prediction_FR(model, window, y_predicted, X_test, img_aux):
  for j in range(y_predicted.shape[1]-window):

    y_input = y_predicted[:,j:j+window]
    time_aux = j*np.ones((y_predicted.shape[0],1))

    if j == 0:
      time_aux = np.zeros((y_predicted.shape[0],1))

    y_predicted[:,j+window] = np.squeeze(model((X_test, y_input, img_aux, time_aux)).numpy())

  return y_predicted

def prediction_model(units = 15, nneurons = 100, window = 5):


  input_img = tf.keras.Input(shape=(sizes[0], sizes[1], 32))
  input_aux = tf.keras.Input(shape=(2))
  input_previous = Input(shape=(window, 1))
  input_time = Input(shape=(1))

  act = 'relu'
  act_2 = 'tanh'

  fcn_nneurons = 50
  cnn_nneurons = 50

  x = input_img

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
  x_img = x

  ##################

  x = input_previous
  x = Conv1D(filters=units, kernel_size=3, padding='same', activation='sigmoid')(x)
  x = tf.concat([Flatten()(x), Flatten()(input_previous[:, -1])], 1)
  x = (tf.concat([x, input_time], 1))
  
  for i in range(2):
    x = Dense(fcn_nneurons, activation='tanh')(x)
  x_time = x

  output = Dot(axes = 1)([x_img, x_time])

  model = Model(inputs = [input_img, input_previous, input_aux, input_time], outputs = [output])

  model.compile(loss = 'mse',
                  optimizer= Adam(learning_rate = .5e-4 , beta_1=0.9), metrics=['mse','mae'])

  model.summary()

  return model

def train_and_get_metrics(data_train, data_test, data_val, units, nneurons, window, epochs, model_name, batch_size):

    X_train, Xy_train, y_train, aux_img_train, aux_time_train = data_train
    X_test, y_test, aux_test = data_test
    X_val, Xy_val, y_val, aux_img_val, aux_time_val = data_val

    model = prediction_model(units = units, nneurons = nneurons, window = window)

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
      filepath= path + path_models + model_name + '.h5',
      save_weights_only=False,
      monitor= 'val_mse',
      # mode='max',
      save_best_only=True)

    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=250,
        restore_best_weights=True)
    
    loss = model.fit((X_train, Xy_train, aux_img_train, aux_time_train),
              y_train, epochs=epochs, batch_size=batch_size, callbacks=[model_checkpoint_callback, early_stopping_callback],
              validation_data=((X_val, Xy_val, aux_img_val, aux_time_val), y_val))

  
    filename = path + path_models + model_name + '.h5'
    model = tf.keras.models.load_model(filename)
    y_predicted = np.zeros(y_test.shape)
    y_predicted[:,:window] = y_test[:,:window]
    y_predicted_FR = prediction_FR(model, window, y_predicted, X_test, aux_test)
    MSE_FR = mean_squared_error(y_predicted_FR, y_test)
    R2_FR = r2_score(y_test, y_predicted_FR)
    mean_error_oil_prediction = np.mean(((np.absolute(y_predicted[:,-1]-y_test[:,-1]))/(np.absolute(y_test[:,-1]))))

    return loss, MSE_FR, R2_FR, mean_error_oil_prediction

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
    
    filepath = './metricas_don.pkl'
    with open(filepath, 'rb') as filepath:
        arquivos = pickle.load(filepath) 
        
    metrics = arquivos['metricas']
    del arquivos

    for i in range(n_cv):
        X_groups.append(X[i*n_samples_group:(i+1)*n_samples_group])
        y_groups.append(y[i*n_samples_group:(i+1)*n_samples_group])
        aux_groups.append(aux[i*n_samples_group:(i+1)*n_samples_group])

    for i in range(6,n_cv):
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

        window = 5
        full_size = y_train.shape[1]

        X_train, X_val, y_train, y_val, aux_train, aux_val = train_test_split(X_train, y_train, aux_train, test_size=0.2, random_state=1)

        data_train= construct_data(full_size, window, X_train, y_train, aux_train)
        data_test = (X_test, y_test, aux_test)
        data_val  = construct_data(full_size, window, X_val, y_val, aux_val)

                                  #train_and_get_metrics(data_train, data_test, data_val,units, nneurons, window, epochs, model_name, batch_size):
        loss, MSE, R2, error_oil = train_and_get_metrics(data_train, data_test, data_val, 15, 100, 5, epochs, 'prediction_model_don_'+str(i), 1024)
        metrics[i] = [loss, MSE, R2, error_oil]
        # metrics.append([loss, MSE, R2, error_oil])
        tf.keras.backend.clear_session()

        filepath = './metricas_don.pkl'
        with open(filepath,"wb") as filepath:
          pickle.dump({"metricas": metrics }, filepath)

    return metrics

metricas = cross_validation(X, y, aux, n_cv=10, epochs=5000)

 
time_stop = time.time()
print(time_stop-time_start)