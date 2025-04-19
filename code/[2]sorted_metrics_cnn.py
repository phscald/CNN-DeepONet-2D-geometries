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

filepath = './don_data.pkl'
with open(filepath, 'rb') as filepath:
    arquivos = pickle.load(filepath)
    
X = arquivos['X']
y = arquivos['y']
aux = arquivos['aux']
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

    n_samples = X.shape[0]
    n_samples_group = int(np.ceil(n_samples/n_cv))

    y_test_list = []
    y_pred_list = []
    err_list=[]

    time_cnn = []
    

    for i in range(n_cv): #[0,1,2,6,7,8,9]:#[1,2,4,5,6,7,8,9]: #range(n_cv):
        X_test = (X[i*n_samples_group:(i+1)*n_samples_group])
        y_test = (y[i*n_samples_group:(i+1)*n_samples_group])
        aux_test = (aux[i*n_samples_group:(i+1)*n_samples_group])

        ind0 = np.where(aux_test[:,0]*.2 >  lower_porosity)[0]
        ind1 = np.where(aux_test[:,0]*.2 <= higher_porosity)[0]

        ind0 = np.intersect1d(ind0, ind1)
        X_test = X_test[ind0]
        y_test = y_test[ind0]
        aux_test = aux_test[ind0] 

        filename = path + path_models + 'cnn_' +str(i)+ '.h5'
        model = tf.keras.models.load_model(filename)

        time_start_cnn = time.time()
        y_predicted = model((X_test, aux_test))
        y_predicted = y_predicted.numpy()
        time_stop_cnn = time.time()
        time_cnn.append(time_stop_cnn-time_start_cnn)


        MSE.append(mean_squared_error(y_predicted, y_test))
        R2.append(r2_score(y_test, y_predicted))
        # oil_err.append(np.mean(((np.absolute(y_predicted[:,-1]-y_test[:,-1]))/(np.absolute(y_test[:,-1])))))
        oil_err.append(mean_absolute_percentage_error(y_test[:,-1], y_predicted[:,-1]))

        if flag_plot:
            y_test_list.append(y_test[:,-1])
            y_pred_list.append(y_predicted[:,-1])
            err_list.append((y_predicted[:,-1]-y_test[:,-1])/(np.absolute(y_predicted[:,-1])))
        
    if flag_plot:
        y_test_list = np.concatenate(y_test_list, axis=0)
        y_pred_list = np.concatenate(y_pred_list, axis=0)
        err_list = np.concatenate(err_list, axis=0)
                
        print(y_test_list.shape)
        print(y_test_list.shape)
        
        plt.figure()
        sns.scatterplot(x=y_test_list, y=y_pred_list, label='oil recovered', color='#5A9BD6')
        sns.lineplot(x=[0.35, 1], y=[0.35, 1], color='#D55E00', linewidth=2, label='perfect prediction line')
        sns.lineplot(x=[0.35, 1], y=[0.35*1.1, 1*1.1], color='#D55E00', linewidth=1.5, linestyle='--',label='10% error line')
        sns.lineplot(x=[0.35, 1], y=[0.35*.9, 1*.9], color='#D55E00', linewidth=1.5, linestyle='--')
        plt.xlabel('oil recovered [pore volumes]')
        plt.ylabel('predicted oil recovered [pore volumes]')
        plt.legend()
        plt.savefig(path+path_img+'pred_dist_CNN.eps', format='eps')
       
        plt.figure()
        sns.histplot(y_pred_list, color='#5A9BD6', alpha=0.0, label='predicted', kde=0, element="step")
        sns.histplot(y_test_list, color='#D55E00', alpha=0.0, label='ideal distribution', kde=0, element="step")
        plt.xlabel('oil recovered [pore volumes]')
        plt.ylabel('Frequency')
        plt.legend()
        plt.savefig(path+path_img+'hist_CNN.svg', format='svg', transparent=True)

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
        plt.savefig(path+path_img+'hist_err_CNN.eps', format='eps')
         

    meanR2 = np.mean(R2)
    stdR2 = np.std(R2, ddof=1)
    meanMSE = np.mean(MSE)
    stdMSE = np.std(MSE, ddof=1)
    meanoil = np.mean(oil_err)
    stdoil = np.std(oil_err, ddof=1)

    print(f'{lower_porosity} - {higher_porosity}')
    # print(f'MSE & ${np.round(meanMSE, 4):.4f}\pm{np.round(stdMSE, 4):.4f}$ \\\\')
    # print(f'R2 & ${np.round(meanR2, 2):.2f}\pm{np.round(stdR2, 2):.2f}$ \\\\')
    # print(f'oil_error & ${np.round(meanoil, 3):.3f}\pm{np.round(stdoil, 3):.3f}$ \\\\')
    print(f'& ${np.round(meanMSE, 4):.4f}\pm{np.round(stdMSE, 4):.4f}$ ')
    print(f'& ${np.round(meanR2, 2):.2f}\pm{np.round(stdR2, 2):.2f}$ ')
    print(f'& ${np.round(meanoil, 3):.3f}\pm{np.round(stdoil, 3):.3f}$ ')
    print('----')
    print(f'time cnn: {np.mean(time_cnn)}')

write_metrics_porosity(.14, .19, True)