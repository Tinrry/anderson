from tqdm import tqdm, trange
import h5py
import numpy as np

import torch
import torch.nn as nn


class MultiLayerP():
    def __init__(self, 
                 network, 
                 loss_function,
                 chebyshev_model_range, 
                 scheduler=None, 
                 save_hdf5=None) -> None:
        self.network = network
        self.loss_function = loss_function
        self.chebyshev_model_range = chebyshev_model_range
        self.scheduler = scheduler
        self.hdf5_filename = save_hdf5
   
    def predict(self, x):
        return self.network(x)

    def train(self, optimizer, epochs, train_loader, val_loader=None):
        for chebyshev_i in self.chebyshev_model_range:
            # creating log
            if val_loader:
                log_dict = {
                'training_loss_per_batch': [],
                'training_loss_per_epoch': [],
                'validation_loss_per_batch': [],
                'validation_loss_per_epoch': []
            }
            else:
                log_dict = {
                    'training_loss_per_batch': [],
                    'training_loss_per_epoch': []
                }
            self._train(optimizer, epochs, train_loader, val_loader, chebyshev_i, log_dict)
            
            if self.hdf5_filename:
                self._save_hdf5(chebyshev_i, 'log_dict', log_dict)

    def _train(self, optimizer, epochs, train_loader, val_loader, chebyshev_i, log_dict):
        # defining weight initialization function
        def init_weights(Module):
            if isinstance(Module, nn.Conv2d):
                torch.nn.init.xavier_uniform(Module.weight)
                Module.bias.data.fill_(0.01)
            elif isinstance(Module, nn.Linear):
                torch.nn.init.xavier_normal_(Module.weight)
                Module.bias.data.fill_(0.01)

        self.network.apply(init_weights)
        # setting network to training mode
        self.network.train()

        for epoch in trange(epochs, desc='Training'):

            once_batch = True
            for x_batch, y_batch, _, _ in tqdm(train_loader, desc=f'epoch {epoch+1} in training', leave=False):
                y_pred = self.network(x_batch).squeeze()
                y_batch = y_batch[:, chebyshev_i, :, :].squeeze()
                    
                loss = self.loss_function(y_pred, y_batch)
                # backward
                optimizer.zero_grad()
                loss.backward()
                # update parameters
                optimizer.step()
                # record loss and accuracy
                log_dict['training_loss_per_batch'].append(loss.detach().cpu().item())
                if once_batch:
                    # we print 前10个样本的当前的预测值n-th order chebyshev alpha作为分析
                    print(f" y_pred : {y_pred[:10].flatten()}")
                    print(f"y_batch: {y_batch[:10].flatten()}")
                    once_batch = False
            
            if self.scheduler is not None:
                self.scheduler.step()
                print(f' Epoch-{epoch+1}/{epochs} lr: ' + f"{optimizer.param_groups[0]['lr']:.5e}")
            
            train_loss_mean = np.array(log_dict['training_loss_per_batch']).mean()
            log_dict['training_loss_per_epoch'].append(train_loss_mean)
            print()
            print(f"Epoch : {epoch+1}/{epochs}  train loss: {train_loss_mean:.10f}")
            
            # validation
            if val_loader:
                val_list = []
                self._inference(val_loader, chebyshev_i, val_list)
                log_dict['validation_loss_per_batch'] = val_list
                # print('-' * 10, 'validate', '-' * 10)
                validation_loss_mean = np.array(log_dict['validation_loss_per_batch']).mean()
                log_dict['validation_loss_per_epoch'].append(validation_loss_mean)
                print(f"Epoch : {epoch+1}/{epochs}  validation loss: {validation_loss_mean:.10f}") 
            
        return log_dict

        
    def test(self, data_loader):
        for chebyshev_i in self.chebyshev_model_range:
            test_log_dict={
                'test_loss_per_batch': [],
                'test_loss_per_epoch': []
            }
            log_list = []
            self._inference(data_loader, chebyshev_i, log_list)
            test_log_dict['test_loss_per_batch'] = log_list
            print('-' * 10, f'test {chebyshev_i:03}', '-' * 10)
            test_loss_mean = np.array(test_log_dict['test_loss_per_batch']).mean()
            test_log_dict['test_loss_per_epoch'].append(test_loss_mean)
            print(f"Test loss: {test_loss_mean:.10f}") 

            if self.hdf5_filename:
                self._save_hdf5(chebyshev_i, 'test_log_dict', test_log_dict)

    def _inference(self, data_loader, chebyshev_i, log_list):
        # test
        with torch.no_grad():
            for x_batch, y_batch, _, _ in tqdm(data_loader, leave=False):

                y_pred = self.network(x_batch).squeeze()
                y_batch = y_batch[:, chebyshev_i, :, :].squeeze()
                loss = self.loss_function(y_pred, y_batch)

                log_list.append(loss.detach().cpu().item())
            return log_list

    def _save_hdf5(self, chebyshev_i, name, data_dict):
        h5_handle = h5py.File(self.hdf5_filename, 'a')
        model_index = f'model_{chebyshev_i:03}'
        if model_index  in h5_handle.keys():
            grp_i = h5_handle.require_group(model_index)
        else:
            grp_i = h5_handle.create_group(model_index)
        if name in grp_i.keys():
            grp_name = grp_i.require_group(name)
        else:
            grp_name = grp_i.create_group(name)
        # save dict in hdf5
        for k, v in data_dict.items():
            grp_name.create_dataset(name=k, data=v)
        h5_handle.close()

    def read_hdf5(self, chebyshev_i=None):
        if chebyshev_i is None:
            chebyshev_i = self.chebyshev_model_range[0]
        h5_handle = h5py.File(self.hdf5_filename, 'r')
        grp_i = h5_handle.require_group(name=f'model_{chebyshev_i:03}')
        training_loss_per_batch = []
        training_loss_per_epoch = []
        test_loss_per_batch = []
        test_loss_per_epoch = []
        validation_loss_per_batch = []
        validation_loss_per_epoch = []
        if 'test_log_dict' in grp_i.keys():
            test_loss_per_batch = grp_i['test_log_dict']['test_loss_per_batch'][:]
            test_loss_per_epoch = grp_i['test_log_dict']['test_loss_per_epoch'][:]
        if 'log_dict' in grp_i.keys():
            training_loss_per_batch = grp_i['log_dict']['training_loss_per_batch'][:]
            training_loss_per_epoch = grp_i['log_dict']['training_loss_per_epoch'][:]
            if 'validation_loss_per_batch' in grp_i['log_dict'].keys():
                validation_loss_per_batch = grp_i['log_dict']['validation_loss_per_batch'][:]
                validation_loss_per_epoch = grp_i['log_dict']['validation_loss_per_epoch'][:]
        
        h5_handle.close()
        return (training_loss_per_batch, training_loss_per_epoch), (validation_loss_per_batch, validation_loss_per_epoch), (test_loss_per_batch, test_loss_per_epoch)