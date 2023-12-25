import os
from tqdm import tqdm, trange
import h5py
import numpy as np
import copy

import torch
import torch.nn as nn


class MultiLayerP():
    def __init__(self, 
                 network, 
                 loss_function,
                 chebyshev_i,
                 scheduler=None, 
                 save_log=None,
                 save_model=None,
                 logger=None
                 ) -> None:
        self.network = network
        self.loss_function = loss_function
        self.chebyshev_i = chebyshev_i
        self.scheduler = scheduler
        self.hdf5_filename = save_log
        self.model_dir = save_model
        self.logger = logger
   
    def train(self, optimizer, epochs, train_loader, val_loader=None):
            # creating log
            if val_loader:
                log_dict = {
                'train_loss_per_batch': [],
                'train_loss_per_epoch': [],
                'validate_loss_per_batch': [],
                'validate_loss_per_epoch': []
            }
            else:
                log_dict = {
                    'train_loss_per_batch': [],
                    'train_loss_per_epoch': []
                }
            self._train(optimizer, epochs, train_loader, val_loader, log_dict)
            
            if self.hdf5_filename is not None:
                self._save_log(log_dict)
            if self.model_dir is not None:
                self._save_model()
            

    def _train(self, optimizer, epochs, train_loader, val_loader, log_dict):
        # defining weight initialization function
        def init_weights(Module):
            if isinstance(Module, nn.Conv2d):
                torch.nn.init.xavier_uniform(Module.weight)
                Module.bias.data.fill_(0.01)
            elif isinstance(Module, nn.Linear):
                torch.nn.init.xavier_normal_(Module.weight)
                Module.bias.data.fill_(0.01)

        self.network.apply(init_weights)
        # setting network to mode
        self.network.train()

        for epoch in trange(epochs, desc='Training'):

            this_epoch = []
            once_batch = True
            for x_batch, y_batch, _, _ in tqdm(train_loader, desc=f'epoch {epoch+1} in training', leave=False):
                y_pred = self.network(x_batch).squeeze()
                y_batch = y_batch[:, self.chebyshev_i, :, :].squeeze()
                    
                loss = self.loss_function(y_pred, y_batch)
                # backward
                optimizer.zero_grad()
                loss.backward()
                # update parameters
                optimizer.step()
                # record loss and accuracy
                this_epoch.append(loss.detach().cpu().item())
                if once_batch:
                    # we print 前10个样本的当前的预测值n-th order chebyshev alpha作为分析
                    self.logger.debug(f" y_pred : {y_pred[:10].flatten()}")
                    self.logger.debug(f"y_batch: {y_batch[:10].flatten()}")
                    self.logger.debug(f'x_batch: {x_batch[:10,0,0,0].flatten()}')
                    once_batch = False
            
            if self.scheduler is not None:
                self.scheduler.step()
                self.logger.debug(f' Epoch-{epoch+1}/{epochs} lr: ' + f"{optimizer.param_groups[0]['lr']:.5e}")
            log_dict['train_loss_per_batch'].extend(this_epoch)
            train_loss_mean = np.array(this_epoch).mean()
            log_dict['train_loss_per_epoch'].append(train_loss_mean)
            self.logger.info(f"Epoch : {epoch+1}/{epochs}  train loss: {train_loss_mean:.10f}")
            
            # validation
            if val_loader:
                val_list = []
                self._inference(val_loader, self.network, val_list)
                log_dict['validate_loss_per_batch'].extend(val_list)
                validate_loss_mean = np.array(val_list).mean()
                log_dict['validate_loss_per_epoch'].append(validate_loss_mean)
                self.logger.info(f" Epoch : {epoch+1}/{epochs}  validation loss: {validate_loss_mean:.10f}") 
            
        return log_dict

    def test(self, data_loader):
        test_log_dict={
            'test_loss_per_batch': [],
            'test_loss_per_epoch': []
        }
        log_list = []
        if self.model_dir:
            # this will load state dict to self.network
            self._load_model()
        
        model = self.network
        self._inference(data_loader, model, log_list)
        test_log_dict['test_loss_per_batch'] = log_list
        self.logger.info(f'test {self.chebyshev_i:03}')
        test_loss_mean = np.array(test_log_dict['test_loss_per_batch']).mean()
        test_log_dict['test_loss_per_epoch'].append(test_loss_mean)
        self.logger.info(f"Test loss: {test_loss_mean:.10f}") 

        if self.hdf5_filename:
            self._save_log(test_log_dict)

    @torch.no_grad
    def _inference(self, data_loader, model, log_list):
        
        model.eval()
        for x_batch, y_batch, _, _ in tqdm(data_loader, leave=False):

            y_pred = model(x_batch).squeeze()
            y_batch = y_batch[:, self.chebyshev_i, :, :].squeeze()
            loss = self.loss_function(y_pred, y_batch)

            log_list.append(loss.detach().cpu().item())
        return log_list

    def _save_log(self, data_dict):
        h5_handle = h5py.File(self.hdf5_filename, 'a')
        model_index = f'model_{self.chebyshev_i:03}'
        grp_i = h5_handle.require_group(model_index)
        # save dict in hdf5
        for k, v in data_dict.items():
            grp_i.create_dataset(name=k, data=v)
        h5_handle.close()

    def read_log(self, chebyshev_i=None):
        if chebyshev_i is None:
            chebyshev_i = self.chebyshev_i
        h5_handle = h5py.File(self.hdf5_filename, 'r')
        grp_i = h5_handle.require_group(name=f'model_{chebyshev_i:03}')
        train_loss_per_batch = []
        train_loss_per_epoch = []
        test_loss_per_batch = []
        test_loss_per_epoch = []
        validate_loss_per_batch = []
        validate_loss_per_epoch = []
        if 'test_loss_per_batch' in grp_i.keys():
            test_loss_per_batch = grp_i['test_loss_per_batch'][:]
            test_loss_per_epoch = grp_i['test_loss_per_epoch'][:]
        train_loss_per_batch = grp_i['train_loss_per_batch'][:]
        train_loss_per_epoch = grp_i['train_loss_per_epoch'][:]
        if 'validate_loss_per_batch' in grp_i.keys():
            validate_loss_per_batch = grp_i['validate_loss_per_batch'][:]
            validate_loss_per_epoch = grp_i['validate_loss_per_epoch'][:]
        
        h5_handle.close()
        return (train_loss_per_batch, train_loss_per_epoch), (validate_loss_per_batch, validate_loss_per_epoch), (test_loss_per_batch, test_loss_per_epoch)

# todo 查看第二个模型分析如何下降loss
    def _save_model(self):
 
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        save_file = os.path.join(self.model_dir, f"{self.chebyshev_i:03}.pth")
        torch.save(self.network.state_dict(), save_file)
    
    def _load_model(self):
        model_name = os.path.join(self.model_dir, f"{self.chebyshev_i:03}.pth")
        checkpoint = torch.load(model_name)
        self.network.load_state_dict(checkpoint)
        try:
            self.network.eval()
        except AttributeError as error:
            print(error)
        return 