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

filepath = path + path_BD + 'y_vimax_3apr.pkl'
with open(filepath, 'rb') as filepath:
	arquivos = pickle.load(filepath)
vinj_5 = arquivos['vinj_max5']
vinj_10 = arquivos['vinj_max10']
vinj_20 = arquivos['vinj_max20']
del arquivos 

filepath = path + path_BD + 'y_selected_7e15.pkl'
with open(filepath, 'rb') as filepath:
	arquivos = pickle.load(filepath)

visc_ratio_7e15 = arquivos['visc_ratio']
porosity_7e15 = arquivos['porosity']
vol_7e15 = arquivos['vol']
vinj_max_7e15 = arquivos['vinj_max']
del arquivos

filepath = path + path_BD + 'X_EfficientNetB7_PCA.pkl'
with open(filepath, 'rb') as filepath:
	arquivos = pickle.load(filepath)
X = arquivos['X']
del arquivos

sizes=(11,16)
x0tam = X.shape[0]
X = np.reshape(X, (x0tam*sizes[0]*sizes[1], 32))
x_norm = MinMaxScaler(feature_range=(-1, 1))
X = x_norm.fit_transform(X)
X = np.reshape(X, (x0tam,sizes[0],sizes[1], 32))
print(X.shape)

def load(filepath, keys):
  with open(filepath, 'rb') as filepath:
      arquivos = pickle.load(filepath)
      
  retorno = list()
  for i in range(len(keys)):
      retorno.append(arquivos[keys[i]])

  return retorno

def load_database():


  filepath = path + path_BD + "y_pack_3apr.pkl"
  key = ["vol", "porosity", "visc_ratio", "d_a_"]
  y, y_porosity, y_visc_ratio, da  = load(filepath, key)

  return da

da = load_database()

filepath = './don_data.pkl'
with open(filepath, 'rb') as filepath:
    arquivos = pickle.load(filepath)
X_ = arquivos['X']
y_ = arquivos['y']
aux_ = arquivos['aux']
del arquivos

n_cv = 10
n_samples = X_.shape[0]
n_samples_group = int(np.ceil(n_samples/n_cv))

X_groups =[]
y_groups =[]
aux_groups =[]

for i in range(n_cv):
    X_groups.append(X_[i*n_samples_group:(i+1)*n_samples_group])
    y_groups.append(y_[i*n_samples_group:(i+1)*n_samples_group])
    aux_groups.append(aux_[i*n_samples_group:(i+1)*n_samples_group])
    
def prediction_FR(model, window, y_predicted, X_test, img_aux):
  for j in range(y_predicted.shape[1]-window):

    y_input = y_predicted[:,j:j+window]
    time_aux = j*np.ones((y_predicted.shape[0],1))

    if j == 0:
      time_aux = np.zeros((y_predicted.shape[0],1))

    y_predicted[:,j+window] = np.squeeze(model((X_test[np.newaxis], y_input, img_aux[np.newaxis], time_aux)).numpy())

  return y_predicted

def oil_error(y_true, y_pred):
  return np.mean(((np.absolute(y_pred[-1]-y_true[-1]))/(np.absolute(y_true[-1]))))

# for j in range(X.shape[0]):
# j = [250, 200, 100, 15, 13 , 88]
# j=j[3]
# print(f'da: {da[j]}')
time_don = []
time_cnn = []
# indexes = [18,200,99]
indexes = [15,200]
for j in indexes:
    plt.figure()
    inds = []
    groups = []
    for i in range(n_cv):
        mask = np.all(X_groups[i] == X[j], axis=(1,2,3))
        ind = np.where(mask)
        # print(ind)
        ind = ind[0].tolist()
        if len(ind)>0:
            groups.extend(len(ind)*[i])
            inds.extend(ind)
        # if len(inds)>0:
        #     break

    colors = ['#5A9BD6', '#D55E00', '#009E73']
    print(groups)
    for i in range(len(inds)):
        # groups[i] = 0
        filename = 'prediction_model_don_' + str(groups[i]) + '.h5'
        # model_num = 6
        # filename = 'prediction_model_don_' + str(model_num) + '.h5'
        model_DON = tf.keras.models.load_model(path + path_models + filename)
        # filename = 'cnn_' + str(model_num) + '.h5'
        filename = 'cnn_' + str(groups[i]) + '.h5'
        model_CNN = tf.keras.models.load_model(path + path_models + filename)

        if aux_groups[groups[i]][inds[i],1] == 5/21: aux_groups[groups[i]][inds[i],1] = 7.5/21
        elif aux_groups[groups[i]][inds[i],1] == 10/21: aux_groups[groups[i]][inds[i],1] = 15/21
        # elif aux_groups[groups[i]][inds[i],1] == 20/21: aux_groups[groups[i]][inds[i],1] = 19/21

        time_start_don = time.time()
        window = 5
        y_pred_DON = np.zeros((1, y_groups[groups[i]].shape[1]))
        y_pred_DON[:,:window] = y_groups[groups[i]][inds[i],:window]
        y_pred_DON = prediction_FR(model_DON, window, y_pred_DON, X_groups[groups[i]][inds[i]], aux_groups[groups[i]][inds[i]])
        y_pred_DON = np.squeeze(y_pred_DON)
        time_stop_don = time.time()

        time_start_cnn = time.time()
        y_pred_CNN = model_CNN((X_groups[groups[i]][inds[i]][np.newaxis], aux_groups[groups[i]][inds[i]][np.newaxis]))
        y_pred_CNN = y_pred_CNN.numpy()
        y_pred_CNN = np.squeeze(y_pred_CNN)
        time_stop_cnn = time.time()

        time_don.append(time_stop_don-time_start_don)
        time_cnn.append(time_stop_cnn-time_start_cnn)

        

    # sns.lineplot(x=[-.1, -.1], y=[0, 400], color='red', linewidth=1.5, linestyle='--')


        if aux_groups[groups[i]][inds[i],1] == 7.5/21:
            if aux_groups[groups[i]][inds[i],0] > .75: idx_7e15=0
            else: idx_7e15=1
            vinj = np.arange(stop=vinj_max_7e15[idx_7e15]+(vinj_max_7e15[idx_7e15]/49)*.1,step = (vinj_max_7e15[idx_7e15]/49))
            sns.lineplot(x= vinj, y= y_pred_DON, color=colors[0], linestyle='--')
            sns.lineplot(x= vinj, y= y_pred_CNN, color=colors[0], linestyle=':')
            sns.lineplot(x= vinj, y= vol_7e15[idx_7e15], color=colors[0], linestyle='-')
            print(aux_groups[groups[i]][inds[i]])
            print(f' MSE CNN: {np.round(mean_squared_error(vol_7e15[idx_7e15], y_pred_CNN),4):.4f},\n MSE DON: {np.round(mean_squared_error(vol_7e15[idx_7e15], y_pred_DON),4):.4f},\n $%oil\_error$ CNN: {np.round(oil_error(vol_7e15[idx_7e15], y_pred_CNN),3):.3f},\n $%oil\_error$ DON: {np.round(oil_error(vol_7e15[idx_7e15], y_pred_DON),3):.3f}')

        elif aux_groups[groups[i]][inds[i],1] == 15/21:
            idx_7e15 =2
            if aux_groups[groups[i]][inds[i],0] < .75: idx_7e15+=1
            vinj = np.arange(stop=vinj_max_7e15[idx_7e15]+(vinj_max_7e15[idx_7e15]/49)*.1,step = (vinj_max_7e15[idx_7e15]/49))
            sns.lineplot(x= vinj, y= y_pred_DON, color=colors[1], linestyle='--')
            sns.lineplot(x= vinj, y= y_pred_CNN, color=colors[1], linestyle=':')
            sns.lineplot(x= vinj, y= vol_7e15[idx_7e15], color=colors[1], linestyle='-')
            print(aux_groups[groups[i]][inds[i]])
            print(f' MSE CNN: {np.round(mean_squared_error(vol_7e15[idx_7e15], y_pred_CNN),4):.4f},\n MSE DON: {np.round(mean_squared_error(vol_7e15[idx_7e15], y_pred_DON),4):.4f},\n $%oil\_error$ CNN: {np.round(oil_error(vol_7e15[idx_7e15], y_pred_CNN),3):.3f},\n $%oil\_error$ DON: {np.round(oil_error(vol_7e15[idx_7e15], y_pred_DON),3):.3f}')

        tf.keras.backend.clear_session()

    sns.lineplot(x= [0], y= [0], color=colors[0], linestyle='-', label=r'$\mu_o/\mu_w=7.5$')
    sns.lineplot(x= [0], y= [0], color=colors[1], linestyle='-', label=r'$\mu_o/\mu_w=15$')
    sns.lineplot(x= [0], y= [0], color='black', linestyle='-', label='ground truth')
    sns.lineplot(x= [0], y= [0], color='black', linestyle=':', label='CNN prediction')
    sns.lineplot(x= [0], y= [0], color='black', linestyle='--', label='DeepONet prediction')

    plt.grid()
    plt.legend()

    plt.xlabel(r'Water injected [pore volumes]')
    plt.ylabel(r'Oil recovered [pore volumes]')


    print(f'time cnn: {np.mean(time_cnn)}')
    print(f'time don: {np.mean(time_don)}')
    plt.savefig(path+path_img+'extrapolation'+str(j)+'.eps', format='eps')