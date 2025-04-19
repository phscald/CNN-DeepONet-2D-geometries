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
import time

plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12

path = '../'
path_BD = 'BD_1/'
path_models = 'models/'
path_img = 'im/'

def prediction_FR(model, window, y_predicted, X_test, img_aux):
  for j in range(y_predicted.shape[1]-window):

    y_input = y_predicted[:,j:j+window]
    time_aux = j*np.ones((y_predicted.shape[0],1))

    if j == 0:
      time_aux = np.zeros((y_predicted.shape[0],1))

    y_predicted[:,j+window] = np.squeeze(model((X_test, y_input, img_aux, time_aux)).numpy())

  return y_predicted

filepath = './don_data.pkl'
with open(filepath, 'rb') as filepath:
    arquivos = pickle.load(filepath)
    
X = arquivos['X']
y = arquivos['y']
aux = arquivos['aux']
del arquivos

filepath = 'metricas_don.pkl'
with open(filepath, 'rb') as filepath:
	arquivos = pickle.load(filepath)
metricas = arquivos['metricas']
del arquivos

def write_metrics_porosity(lower_porosity, higher_porosity, flag_plot=False):

    X_groups =[]
    y_groups =[]
    aux_groups =[]
    # metrics =[]

    MSE = []
    R2 = []
    oil_err = []

    losshist_train = []
    losshist_test = []

    n_cv =10
    window = 5

    n_samples = X.shape[0]
    n_samples_group = int(np.ceil(n_samples/n_cv))

    y_test_list = []
    y_pred_list = []
    err_list = []
    
    time_don =[]

    for i in range(n_cv):#[0,1,2,6,7,8,9]:#[1,2,4,5,6,7,8,9]: #range(n_cv):
        X_test = (X[i*n_samples_group:(i+1)*n_samples_group])
        y_test = (y[i*n_samples_group:(i+1)*n_samples_group])
        aux_test = (aux[i*n_samples_group:(i+1)*n_samples_group])

        ind0 = np.where(aux_test[:,0]*.2 >  lower_porosity)[0]
        ind1 = np.where(aux_test[:,0]*.2 <= higher_porosity)[0]

        ind0 = np.intersect1d(ind0, ind1)
        X_test = X_test[ind0]
        y_test = y_test[ind0]
        aux_test = aux_test[ind0] 

        print(f'shape: {ind0.shape}')

        filename = path + path_models + 'prediction_model_don_' +str(i)+ '.h5'
        model = tf.keras.models.load_model(filename)

        time_start_don = time.time()
        y_predicted = np.zeros(y_test.shape)
        y_predicted[:,:window] = y_test[:,:window]
        y_predicted_FR = prediction_FR(model, window, y_predicted, X_test, aux_test)
        time_stop_don = time.time()
        time_don.append(time_stop_don-time_start_don)
        
        ind_remove = np.where(y_predicted_FR[:,-1]<=.37)[0]
        y_predicted_FR = np.delete(y_predicted_FR, ind_remove, axis=0)
        y_test = np.delete(y_test, ind_remove, axis=0)

        MSE.append(mean_squared_error(y_predicted_FR, y_test))
        R2.append(r2_score(y_test, y_predicted_FR))
        oil_err.append(mean_absolute_percentage_error(y_test[:,-1], y_predicted_FR[:,-1]))
        # oil_err.append(np.mean(((np.absolute(y_predicted[:,-1]-y_test[:,-1]))/(np.absolute(y_test[:,-1])))))
 
        if flag_plot:
            y_test_list.append(y_test[:,-1])
            y_pred_list.append(y_predicted_FR[:,-1])

            err_list.append((y_predicted_FR[:,-1]-y_test[:,-1])/(np.absolute(y_predicted_FR[:,-1])))
        # ind = np.where(y_predicted_FR[:,-1]<0)[0][0]
        # sns.lineplot(y_test[ind])
        # sns.lineplot(y_predicted_FR[ind])

    
    if flag_plot:
        y_test_list = np.concatenate(y_test_list, axis=0)
        y_pred_list = np.concatenate(y_pred_list, axis=0)
        err_list = np.concatenate(err_list, axis=0)

        plt.figure()
        sns.scatterplot(x=y_test_list, y=y_pred_list, label='oil recovered', color='#5A9BD6')
        sns.lineplot(x=[0.35, 1], y=[0.35, 1], color='#D55E00', linewidth=2, label='perfect prediction line')
        sns.lineplot(x=[0.35, 1], y=[0.35*1.1, 1*1.1], color='#D55E00', linewidth=1.5, linestyle='--',label='10% error line')
        sns.lineplot(x=[0.35, 1], y=[0.35*.9, 1*.9], color='#D55E00', linewidth=1.5, linestyle='--')
        plt.xlabel('oil recovered [pore volumes]')
        plt.ylabel('predicted oil recovered [pore volumes]')
        plt.legend()
        plt.savefig(path+path_img+'pred_dist_DON.eps', format='eps', transparent=True)
        
       
        plt.figure()
        
        sns.histplot(y_pred_list, color='#5A9BD6', alpha=0.0, label='predicted', kde=0, element="step")
        sns.histplot(y_test_list, color='#D55E00', alpha=0.0, label='ideal distribution', kde=0, element="step")
        plt.xlabel('oil recovered [pore volumes]')
        plt.ylabel('Frequency')
        plt.legend()
        plt.savefig(path+path_img+'hist_DON.svg', format='svg', transparent=True)

        ind = np.where(np.absolute(err_list)<=.1)[0]
        perc_bom = ind.shape[0]/err_list.shape[0]
        ind = np.where(np.absolute(err_list)>.1)[0]
        perc_ruim = ind.shape[0]/err_list.shape[0]
        print(f'{perc_bom}, {perc_ruim}')

        plt.figure()
        sns.histplot(err_list, color='#5A9BD6', alpha=1, label='percentual error')
        # sns.lineplot(x=[-.1, -.1], y=[0, 400], color='#D55E00', linewidth=5, linestyle='-', label='10% error line')
        # sns.lineplot(x=[.1, .1], y=[0, 400], color='#D55E00', linewidth=5, linestyle='-')
        plt.xlabel('percentual error in the oil recovered prediction')
        plt.ylabel('Frequency')
        plt.legend()
        plt.savefig(path+path_img+'hist_err_DON.eps', format='eps')

    meanR2 = np.mean(R2)
    stdR2 = np.std(R2, ddof=1)
    meanMSE = np.mean(MSE)
    stdMSE = np.std(MSE, ddof=1)
    meanoil = np.mean(oil_err)
    stdoil = np.std(oil_err, ddof=1)

    print(f'{lower_porosity} - {higher_porosity}')
    print(f'MSE & ${np.round(meanMSE, 4):.4f}\pm{np.round(stdMSE, 4):.4f}$ \\\\')
    print(f'R2 & ${np.round(meanR2, 2):.2f}\pm{np.round(stdR2, 2):.2f}$ \\\\')
    print(f'\%oil\_error & ${np.round(meanoil, 3):.3f}\pm{np.round(stdoil, 3):.3f}$ \\\\ \\hline')
    print('-----')
    print(np.mean(time_don))

write_metrics_porosity(.14, .19, True)