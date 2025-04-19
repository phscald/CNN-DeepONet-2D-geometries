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
import seaborn as sns

plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12

filepath = 'metricas_cnn.pkl'
with open(filepath, 'rb') as filepath:
	arquivos = pickle.load(filepath)
metricas = arquivos['metricas']
del arquivos

path = '../'
path_BD = 'BD_1/'
path_models = 'models/'
path_img = 'im/'

for i in range(2):
    plt.figure()
    sns.lineplot(metricas[i][0].history['val_mse'], label='validation loss', color='#D55E00')
    sns.lineplot(metricas[i][0].history['mse'], label='loss', color='#5A9BD6')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.yscale('log')
    plt.legend()
    plt.savefig(path+path_img+'loss_CNN'+str(i)+'.eps', format='eps')
    
filepath = 'metricas_don.pkl'
with open(filepath, 'rb') as filepath:
	arquivos = pickle.load(filepath)
metricas = arquivos['metricas']
del arquivos

for i in range(2):
    plt.figure()
    sns.lineplot(metricas[i][0].history['val_mse'], label='validation loss', color='#D55E00')
    sns.lineplot(metricas[i][0].history['mse'], label='loss', color='#5A9BD6')
    plt.yscale('log')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(path+path_img+'loss_DON'+str(i)+'.eps', format='eps')