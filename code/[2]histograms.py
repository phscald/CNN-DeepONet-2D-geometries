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

y_final = y[:, -1]

plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12

import seaborn as sns
sns.histplot(y_porosity, bins=9, kde= False, binrange=(0.14, 0.185), color='#5A9BD6')
plt.xlabel('porosity')
plt.ylabel('Frequency')
plt.show()
plt.savefig('hist-porosity.eps', format='eps')
plt.close()

sns.histplot(y_final, bins=6, kde= False, binrange=(0.4, 1), color='#5A9BD6')
plt.xlabel('oil recovered [pore volumes]')
plt.ylabel('Frequency')
plt.show()
plt.savefig('hist-final-oil.eps', format='eps')