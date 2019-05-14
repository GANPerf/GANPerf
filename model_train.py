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

from model_def import *
from util import *

seed = 0

torch.manual_seed( seed )

use_gpu = True


torch.set_default_tensor_type('torch.FloatTensor')






max_epoch = 50
BATCH_SIZE = 64
C1, C2, weight_decay = 1, 0, 1e-5
#C1, C2, weight_decay = 1, 0, 0
lr_model, lr_D = 1e-3, 5e-5
lambda1 = 0.5





train_set = MyDataset( '../Database_input', 'train' )
train_loader = DataLoader( train_set, batch_size=BATCH_SIZE, shuffle=True )

valid_set = MyDataset( '../Database_input', 'valid' )
valid_loader = DataLoader( valid_set, batch_size=BATCH_SIZE, shuffle=True )

test_set = MyDataset( '../Database_input', 'test' )
test_loader = DataLoader( test_set, batch_size=BATCH_SIZE, shuffle=True )








model = MyModel()
criterion_MSE = nn.MSELoss() 

if use_gpu:
    model.cuda()
    criterion_MSE.cuda()


optimizer = optim.Adam( filter( lambda p: p.requires_grad, model.parameters() ), lr=lr_model, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau( optimizer, 'max', patience=5 )  # 学习率衰减










model_D = Model_D()
criterion_BCE = nn.BCELoss()

if use_gpu:
    model_D.cuda()
    criterion_BCE.cuda()

optimizer_D = optim.Adam( model_D.parameters(), lr=lr_D, weight_decay=weight_decay )




def l1_loss(var):
    return torch.abs(var).sum()



print( '\n-------------- Training Phase --------------' )


record = pd.DataFrame( [], columns=['epoch', 'train_mse', 'train_rank_spearman',
    'valid_mse', 'valid_rank_spearman', 'test_mse', 'test_rank_spearman'] )




for epoch in range( max_epoch ):

    print( '\nEpoch %d' % epoch )
    t0 = time.time()

    values = [epoch]




    ################ Training on Train Set ################

    y_train = np.zeros( 0, dtype=float )
    pred_train = np.zeros( 0, dtype=float )

    for i, data in enumerate( train_loader, 0 ):
  
        # ------------ Prapare Variables ------------
        batch_x, batch_y = data

        batch_x, batch_y = Variable(batch_x), Variable(batch_y)
        if use_gpu:
            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
       
        
        real_labels = sample_real_labels( batch_x.size(0), '../Database_input/label/train_label.csv' )
        #real_labels = batch_y
        real_labels = Variable(real_labels)
        

       
        label_one = Variable( torch.ones( batch_x.size(0) ) )      # 1 for real
        label_zero = Variable( torch.zeros( batch_x.size(0) ) )    # 0 for fake
      
        if use_gpu:
            real_labels = real_labels.cuda()
            label_one = label_one.cuda()
            label_zero = label_zero.cuda()




        # ------------ Training model_D ------------
        optimizer_D.zero_grad()
        model.eval()  # lock model

     
        output_real = model_D( real_labels )
     
        loss_real = criterion_BCE( output_real.squeeze(), label_one )
        acc_real = ( output_real >= 0.5 ).data.float().mean()

   
        fake_labels = model( batch_x.squeeze() ).detach()
        #print(fake_labels)
        output_fake = model_D( fake_labels )
        loss_fake = criterion_BCE( output_fake.squeeze(), label_zero )        
        acc_fake = ( output_fake < 0.5 ).data.float().mean()
        
   
        loss_D = loss_real + loss_fake
        acc_D = ( acc_real + acc_fake ) / 2

        loss_D.backward()
        optimizer_D.step()






        # ------------ Training model ------------
        model.train()  # unlock model
        optimizer.zero_grad()


        fake_labels = model( batch_x )
        output = model_D( fake_labels )

        fake_labels=fake_labels.float()
        batch_y=batch_y.float()
        term1 = C1 * criterion_MSE( fake_labels, batch_y )
        term2 = C2 * criterion_BCE( output.squeeze(), label_one )
        l1_regular = float(torch.tensor(0))
        l2_regular = float(torch.tensor(0))
        for param in model.parameters():
            l1_regular += torch.norm(param, 1)
            l2_regular += torch.norm(param, 2)
            
			
   #     l1_regular = lambda1 * l1_loss(fake_labels)
        
        loss = term1 + term2 
        #loss = term1 + term2 + l1_regular
        loss.backward()
        optimizer.step()







        # ------------ Preparation for Evaluation on Train Set ------------
 

        fake_labels = fake_labels.cpu().data.numpy() if use_gpu else fake_labels.data.numpy()
        batch_y = batch_y.cpu().data.numpy() if use_gpu else batch_y.data.numpy()

     #   pred_train = np.concatenate( [pred_train, fake_labels] )
      #  y_train = np.concatenate( [y_train, batch_y] )









    ################ Evaluation on Train Set ################

    mse = mean_squared_error( batch_y, fake_labels )
    rank_spearman = stats.spearmanr( get_rank(batch_y), get_rank(fake_labels) )[0]
    values.append( mse )
    values.append( rank_spearman )
    print( 'Train Set\tmse=%f, rank_spearman=%f' % ( mse, rank_spearman ) )
   

    





    ################ Evaluation on Valid/Test Set ################

    model.eval()  # evaluation mode

    mre, mse, rank_spearman = model_eval( model, valid_loader, use_gpu )
    values.append( mre )
    values.append( mse )
    values.append( rank_spearman )
    print( 'Valid Set\tmre=%f, mse=%f, rank_spearman=%f' % ( mre, mse, rank_spearman ) )


    scheduler.step( rank_spearman )  
    
    if epoch == 49:
        mre, mse, rank_spearman = test_eval( model, test_loader, use_gpu )
    else:
        mre, mse, rank_spearman = model_eval( model, test_loader, use_gpu )
    values.append( mre )
    values.append( mse )
    values.append( rank_spearman )
    print( 'Test Set\tmre=%f, mse=%f, rank_spearman=%f' % ( mre, mse, rank_spearman ) )


    print( 'Done in %.2fs' % ( time.time() - t0 ) )

    
    






    ################ Writing Record ################

    temp = pd.DataFrame( [values], columns=['epoch', 'train_mse', 'train_rank_spearman',
        'valid_mre', 'valid_mse', 'valid_rank_spearman', 'test_mre', 'test_mse', 'test_rank_spearman'] )
    record = record.append( temp )






record.to_csv( 'record.csv', index=False )
torch.save( model.state_dict(), 'model_param.pkl' )





