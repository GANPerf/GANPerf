# -*- coding: utf-8 -*-
from __future__ import print_function, division
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.datasets import ImageFolder

from PIL import Image
from scipy import stats
from sklearn.metrics import mean_squared_error

from sklearn import preprocessing



def get_rank(x):
   
    arg = np.argsort(x)
    rank = np.zeros_like(x)

    for i in range( len(x) ):
        rank[ arg[i] ] = i

    return rank


def ToTensor(x):
    return torch.Tensor(x)




class MyDataset(Dataset):

    def __init__(self, input_dir, dataset_str):

        self.input_dir = input_dir
        self.dataset_str = dataset_str
        self.df = pd.read_csv( '%s/label/%s_label.csv' % ( input_dir, dataset_str ) )
        self.dx = pd.read_csv( '%s/features/%s_features.csv' % ( input_dir, dataset_str ) )
        attrs=['label']
        global xx
        xx=np.max(self.df[attrs])
        global yy
        yy=np.min(self.df[attrs])

        self.df[attrs]=(self.df[attrs]-np.min(self.df[attrs]))/(np.max(self.df[attrs])-np.min(self.df[attrs]))


    def __getitem__(self, index):

    
        X = self.dx.iloc[index]

        y = self.df.iloc[index, 0:].values.astype(float)

        return ToTensor(X), y




    def __len__(self):
        return len( self.df )










def model_eval( model, data_loader, use_gpu ):

    y_true = np.zeros( 0, dtype=float )
    y_pred = np.zeros( 0, dtype=float )

    for i, data in enumerate( data_loader, 0 ):

        batch_x, batch_y = data
        batch_x, batch_y = Variable(batch_x), batch_y[:, 0].squeeze().numpy()  

        if use_gpu:
            batch_x = batch_x.cuda()

        output = model( batch_x )
        output = output[:, 0].squeeze()  
        cur_pred = output.cpu().data.numpy() if use_gpu else output.data.numpy()

        y_true = np.concatenate( [y_true, batch_y] )
        y_pred = np.concatenate( [y_pred, cur_pred] )
        
   
    y_true_real=(float(xx-yy))*y_true+float(yy)
    y_pred_real=(float(xx-yy))*y_pred+float(yy)
 
    resul=abs(y_true_real-y_pred_real)/(y_true_real)
   
    mre = resul.sum()/len(resul)
    mse = mean_squared_error( y_true, y_pred )
    rank_spearman = stats.spearmanr( get_rank(y_true), get_rank(y_pred) )[0]
    
	
    
   

    return mre, mse, rank_spearman
    
def test_eval( model, test_loader, use_gpu ):

    y_true = np.zeros( 0, dtype=float )
    y_pred = np.zeros( 0, dtype=float )

    for i, data in enumerate( test_loader, 0 ):

        batch_x, batch_y = data
        batch_x, batch_y = Variable(batch_x), batch_y[:, 0].squeeze().numpy()  

        if use_gpu:
            batch_x = batch_x.cuda()

        output = model( batch_x )
        output = output[:, 0].squeeze() 
        cur_pred = output.cpu().data.numpy() if use_gpu else output.data.numpy()

        y_true = np.concatenate( [y_true, batch_y] )
        y_pred = np.concatenate( [y_pred, cur_pred] )
        
    y_true_real=(float(xx-yy))*y_true+float(yy)
    y_pred_real=(float(xx-yy))*y_pred+float(yy)
    resul=abs(y_true_real-y_pred_real)/(C3*y_true_real)
    mre = resul.sum()/len(resul)
    mse = mean_squared_error( y_true, y_pred )
    rank_spearman = stats.spearmanr( get_rank(y_true), get_rank(y_pred) )[0]
	
    
   

    return mre, mse, rank_spearman









def sample_real_labels( batch_size, filename ):
    
    df = pd.read_csv( filename ).sample( batch_size )
    y = df.iloc[:, 0:].values
    y = torch.from_numpy(y).float()

    return y

