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
                 device, 
                 scheduler=None, 
                 save_hdf5=None) -> None:
        self.network = network.to(device)
        self.loss_function = loss_function
        self.chebyshev_model_range = chebyshev_model_range
        self.scheduler = scheduler
        self.save_hdf5 = save_hdf5
   
    def predict(self, x):
        return self.network(x)

    def train(self, optimizer, epochs, train_loader, val_loader=None):
        for chebyshev_i in self.chebyshev_model_range:
            # creating log
            train_log_dict = {
                'training_loss_per_batch': [],
                'validation_loss_per_batch': [],
                'test_loss_per_batch': []
            }
            self._train(optimizer, epochs, train_loader, val_loader, chebyshev_i, train_log_dict)
            
            if self.save_hdf5:
                self._save_hdf5(chebyshev_i, 'train_log_dict', train_log_dict)

    def _train(self, optimizer, epochs, train_loader, val_loader, chebyshev_i, log_dict):
        # defining weight initialization function
        def init_weights(Module):
            if isinstance(Module, nn.Conv2d):
                torch.nn.init.xavier_uniform(Module.weight)
                Module.bias.data.fill_(0.01)
            elif isinstance(Module, nn.Linear):
                torch.nn.init.xavier_uniform(Module.weight)
                Module.bias.data.fill_(0.01)

        self.network.apply(init_weights)
        # setting network to training mode
        self.network.train()

        for epoch in trange(epochs, desc='Training'):

            once_batch = True
            for x_batch, y_batch, _, _ in tqdm(train_loader, desc=f'epoch {epoch+1} in training', leave=False):
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

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
                total_sample += len(x_batch)
                if once_batch:
                    # we print 前10个样本的当前的预测值n-th order chebyshev alpha作为分析
                    print(f" y_pred : {y_pred[:10].flatten()}")
                    print(f"y_batch: {y_batch[:10].flatten()}")
                    once_batch = False
            
            if self.scheduler is not None:
                self.scheduler.step()
                print(f'Epoch-{epoch+1} lr: ' + f"{optimizer.param_groups[0]['lr']}")

            # validation
            if val_loader:
                self.test(val_loader, chebyshev_i, log_dict)
                print('-' * 10, 'validate', '-' * 10)

            train_loss_mean = np.array(log_dict['training_loss_per_batch']).mean()
            print(f" epoch : {epoch+1}/{epochs}  train loss: {train_loss_mean:.10f}")
            return log_dict

    def test(self, data_loader):
        for chebyshev_i in range(self.chebyshev_model_range):
            test_log_dict={
                'test_loss_per_batch': []
            }
            self._test(data_loader, chebyshev_i, test_log_dict)
            print('-' * 10, f'test {chebyshev_i:03}', '-' * 10)
            if self.save_hdf5:
                self._save_hdf5(chebyshev_i, 'test_log_dict', test_log_dict)

    def _test(self, data_loader, chebyshev_i, log_dict):
        # test
        with torch.no_grad():
            for x_batch, y_batch, _, _ in tqdm(data_loader, leave=False):
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                y_pred = self.network(x_batch).squeeze()
                y_batch = y_batch[:, chebyshev_i, :, :].squeeze()
                loss = self.loss_function(y_pred, y_batch)

                log_dict['test_loss_per_batch'].append(loss.detach().cpu().item())
            loss_mean = np.array(log_dict['test_loss_per_batch']).mean()
            print(f'loss : {loss_mean}')
            return log_dict

    def _save_hdf5(self, chebyshev_i, name, data):
        h5_handle = h5py.File(self.save_hdf5, 'a')
        grp_i = h5_handle.create_group(name=f'model_{chebyshev_i:03}')
        grp_i.create_dataset(name=name, data=data)
        h5_handle.close()
