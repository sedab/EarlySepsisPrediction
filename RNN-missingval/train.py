import pickle
import pandas as pd
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn.init as init
from torch.autograd import Variable
import torch.utils.data as utils
import itertools
from pandas.core.common import flatten

#####################################################
#get the data
filename = '/scratch/sb3923/time_series/data/data_proc.pickle'
with open(filename, 'rb') as f:
    all_df = pickle.load(f)
    

all_df=all_df.drop(['EtCO2'],axis=1)
all_df['pat_id']=all_df['pat_id'].astype(int)
all_df['hour']=all_df['hour']+1

#need to normalize the data (except the binary)
#drop the binary
all_df=all_df.drop(['Gender'],axis=1)
all_df=all_df.drop(['Unit1'],axis=1)
all_df=all_df.drop(['Unit2'],axis=1)

#normalize the data

for c in all_df.columns:
    all_df[c]=all_df[c]/(all_df[c].max())
        
    
all_df = all_df.fillna(0)
#to do: drop zero row and below go from 0 to length of the given patient
all_df = all_df.loc[~(all_df.drop(['hour','pat_id'],axis=1)==0).all(axis=1)]

#####################################################



def PrepareDataset(data, \
                   BATCH_SIZE = 64, \
                   seq_len = 12, \
                   pred_len = 1, \
                   train_propotion = 0.7, \
                   valid_propotion = 0.2, \
                   masking = False, \
                   mask_ones_proportion = 0.8):
    """ Prepare training and testing datasets and dataloaders.
    
    Convert speed/volume/occupancy matrix to training and testing dataset. 
    The vertical axis of speed_matrix is the time axis and the horizontal axis 
    is the spatial axis.
    
    Args:
        speed_matrix: a Matrix containing spatial-temporal speed data for a network
        seq_len: length of input sequence
        pred_len: length of predicted sequence
    Returns:
        Training dataloader
        Testing dataloader
    """
    time_len = data.shape[0]
    #speed_matrix = speed_matrix.clip(0, 100) #limit the values to 0-100
    
    max_data = data.max().max()
    #speed_matrix =  speed_matrix / max_speed
    
    data_sequences, data_labels = [], []
    for p in data['pat_id'].unique():
        pat_len = len(data[data['pat_id']==p])
        if (pat_len>(seq_len+pred_len)):
        #for i in range(time_len - seq_len - pred_len):
            for i in range(pat_len - seq_len - pred_len):
                data_sequences.append(data.drop(['SepsisLabel'], axis=1)[data['pat_id']==p].iloc[i:i+seq_len].values)
                data_labels.append(data['SepsisLabel'][data['pat_id']==p].iloc[i+seq_len:i+seq_len+pred_len].values)
                #data_labels.append(data.drop(['SepsisLabel'], axis=1)[data['pat_id']==p].iloc[i+seq_len:i+seq_len+pred_len].values)
                #data_sequences.append(data[data['pat_id']==p].iloc[i:i+seq_len].values)
                #data_labels.append(data[data['pat_id']==p].iloc[i+seq_len:i+seq_len+pred_len].values)

    #print(i)
    data_sequences, data_labels = np.asarray(data_sequences), np.asarray(data_labels)
    #print(data_sequences.shape)
    #(951, 48, 42)
    if masking:
        print('Split Speed finished. Start to generate Mask, Delta, Last_observed_X ...')
        np.random.seed(1024)
        #Mask = np.random.choice([0,1], size=(data_sequences.shape), p = [1 - mask_ones_proportion, mask_ones_proportion])
        #speed_sequences = np.multiply(speed_sequences, Mask)
        Mask = data_sequences
        Mask[Mask!=0]=1
        
        
        # temporal information
        interval = 1 # 5 minutes
        S = np.zeros_like(data_sequences) # time stamps
        for i in range(S.shape[1]):
            S[:,i,:] = interval * i
            
        #print(S)
        Delta = np.zeros_like(data_sequences) # time intervals
        for i in range(1, S.shape[1]):
            Delta[:,i,:] = S[:,i,:] - S[:,i-1,:]

        missing_index = np.where(Mask == 0)

        X_last_obsv = np.copy(data_sequences)
        for idx in range(missing_index[0].shape[0]):
            i = missing_index[0][idx] 
            j = missing_index[1][idx]
            k = missing_index[2][idx]
            if j != 0 and j != (seq_len-1):
                Delta[i,j+1,k] = Delta[i,j+1,k] + Delta[i,j,k]
            if j != 0:
                X_last_obsv[i,j,k] = X_last_obsv[i,j-1,k] # last observation
        
        #this should be column wise
        Delta = Delta / Delta.max() # normalize
    
    # shuffle and split the dataset to training and testing datasets
    print('Generate Mask, Delta, Last_observed_X finished. Start to shuffle and split dataset ...')
    sample_size = data_sequences.shape[0]/48
    index_ = np.arange(sample_size, dtype = int)
    np.random.seed(1024)
    np.random.shuffle(index_)
    
    index=[]
    for i in index_:
        index.append(list(range(i,i+48)))
     
    index=list(flatten(index))
    
       
    data_sequences = data_sequences[index]
    data_labels = data_labels[index]
    

    if masking:
        X_last_obsv = X_last_obsv[index]
        Mask = Mask[index]
        Delta = Delta[index]
        data_sequences = np.expand_dims(data_sequences, axis=1)
        X_last_obsv = np.expand_dims(X_last_obsv, axis=1)
        Mask = np.expand_dims(Mask, axis=1)
        Delta = np.expand_dims(Delta, axis=1)
        dataset_agger = np.concatenate((data_sequences, X_last_obsv, Mask, Delta), axis = 1)
        
    train_index = int(np.floor(sample_size * train_propotion))
    valid_index = int(np.floor(sample_size * ( train_propotion + valid_propotion)))
    
    if masking:
        train_data, train_label = dataset_agger[:train_index], data_labels[:train_index]
        valid_data, valid_label = dataset_agger[train_index:valid_index], data_labels[train_index:valid_index]
        test_data, test_label = dataset_agger[valid_index:], data_labels[valid_index:]
    else:
        train_data, train_label = data_sequences[:train_index], data_labels[:train_index]
        valid_data, valid_label = data_sequences[train_index:valid_index], data_labels[train_index:valid_index]
        test_data, test_label = data_sequences[valid_index:], data_labels[valid_index:]
    
    train_data, train_label = torch.Tensor(train_data), torch.Tensor(train_label)
    valid_data, valid_label = torch.Tensor(valid_data), torch.Tensor(valid_label)
    test_data, test_label = torch.Tensor(test_data), torch.Tensor(test_label)
    
    train_dataset = utils.TensorDataset(train_data, train_label)
    valid_dataset = utils.TensorDataset(valid_data, valid_label)
    test_dataset = utils.TensorDataset(test_data, test_label)
    
    train_dataloader = utils.DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=True, drop_last = True)
    valid_dataloader = utils.DataLoader(valid_dataset, batch_size = BATCH_SIZE, shuffle=True, drop_last = True)
    test_dataloader = utils.DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle=True, drop_last = True)
    
    X_mean = np.mean(data_sequences, axis = 0)
    
    print('Finished')
    
    return train_dataloader, valid_dataloader, test_dataloader, max_data, X_mean









def Train_Model(model, train_dataloader, valid_dataloader, num_epochs = 300, patience = 10, min_delta = 0.00001):
    
    print('Model Structure: ', model)
    print('Start Training ... ')
    
    model.cuda()
    
    if (type(model) == nn.modules.container.Sequential):
        output_last = model[-1].output_last
        print('Output type dermined by the last layer')
    else:
        output_last = model.output_last
        print('Output type dermined by the model')
        
    loss_MSE = torch.nn.MSELoss()
    loss_L1 = torch.nn.L1Loss()
    loss_CE=torch.nn.CrossEntropyLoss()
    loss_BCE=torch.nn.BCELoss()
    fc = torch.nn.Linear(38, 1)
    
    learning_rate = 0.0001
    optimizer = torch.optim.RMSprop(model.parameters(), lr = learning_rate, alpha=0.99)
    use_gpu = torch.cuda.is_available()
    
    interval = 100
    losses_train = []
    losses_valid = []
    losses_epochs_train = []
    losses_epochs_valid = []
    
    cur_time = time.time()
    pre_time = time.time()
    
    # Variables for Early Stopping
    is_best_model = 0
    patient_epoch = 0
    for epoch in range(num_epochs):
        
        trained_number = 0
        
        valid_dataloader_iter = iter(valid_dataloader)
        
        losses_epoch_train = []
        losses_epoch_valid = []
        
        for data in train_dataloader:
            inputs, labels = data

            if inputs.shape[0] != batch_size:
                continue

            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else: 
                inputs, labels = Variable(inputs), Variable(labels)
            
            model.zero_grad()

            #here pass the first columns of the model and labels become the last element
            
            #inputs doesnt include the sepsis label
            outputs = model(inputs)
         
            
            if output_last:
                loss_train = loss_BCE(torch.squeeze(outputs),torch.squeeze(labels) )
            else:
                loss_train = loss_BCE(torch.squeeze(outputs),torch.squeeze(labels) )

        
            losses_train.append(loss_train.data)
            losses_epoch_train.append(loss_train.data)
            
            optimizer.zero_grad()
            
            loss_train.backward()
            
            optimizer.step()
            
             # validation 
            try: 
                inputs_val, labels_val = next(valid_dataloader_iter)
            except StopIteration:
                valid_dataloader_iter = iter(valid_dataloader)
                inputs_val, labels_val = next(valid_dataloader_iter)
            
            if use_gpu:
                inputs_val, labels_val = Variable(inputs_val.cuda()), Variable(labels_val.cuda())
            else: 
                inputs_val, labels_val = Variable(inputs_val), Variable(labels_val)
                
            model.zero_grad()
            
            outputs_val = model(inputs_val)
            
            if output_last:
                loss_valid = loss_BCE(torch.squeeze(outputs_val),torch.squeeze(labels_val) )
            else:
                loss_valid = loss_BCE(torch.squeeze(outputs_val),torch.squeeze(labels_val) )


            losses_valid.append(loss_valid.data)
            losses_epoch_valid.append(loss_valid.data)
            
            # output
            trained_number += 1
            
        avg_losses_epoch_train = sum(losses_epoch_train).cpu().numpy() / float(len(losses_epoch_train))
        avg_losses_epoch_valid = sum(losses_epoch_valid).cpu().numpy() / float(len(losses_epoch_valid))
        losses_epochs_train.append(avg_losses_epoch_train)
        losses_epochs_valid.append(avg_losses_epoch_valid)
        
        # Early Stopping
        if epoch == 0:
            is_best_model = 1
            best_model = model
            min_loss_epoch_valid = 10000.0
            if avg_losses_epoch_valid < min_loss_epoch_valid:
                min_loss_epoch_valid = avg_losses_epoch_valid
        else:
            if min_loss_epoch_valid - avg_losses_epoch_valid > min_delta:
                is_best_model = 1
                best_model = model
                min_loss_epoch_valid = avg_losses_epoch_valid 
                patient_epoch = 0
            else:
                is_best_model = 0
                patient_epoch += 1
                if patient_epoch >= patience:
                    print('Early Stopped at Epoch:', epoch)
                    break
        
        # Print training parameters
        cur_time = time.time()
        print('Epoch: {}, train_loss: {}, valid_loss: {}, time: {}, best model: {}'.format( \
                    epoch, \
                    np.around(avg_losses_epoch_train, decimals=8),\
                    np.around(avg_losses_epoch_valid, decimals=8),\
                    np.around([cur_time - pre_time] , decimals=2),\
                    is_best_model) )
        pre_time = cur_time
                
    return best_model, [losses_train, losses_valid, losses_epochs_train, losses_epochs_valid]






