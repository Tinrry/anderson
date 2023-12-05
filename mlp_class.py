import os
from tqdm import tqdm, trange
import h5py

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

class MultiLayerP():
    def __init__(self, network, lr, device) -> None:
        self.network = network.to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)

    def train(model, train_loader, n_epoch, criterion, lr, device, model_range, config):
        # creating log
        log_dict = {
            'training_loss': [],
            'validation_loss': [],
            'training_accuracy': [],
            'validation_accuracy': []
        }
        
        loss_f = config['loss_file']
        loss_h5 = h5py.File(loss_f, 'w')

        for chebyshev_i in model_range:
            grp = loss_h5.create_group(f'model_{chebyshev_i:03}')
            
            # init model
            model.to(device=device).double()
            # 第 i 个 model 预测 第i个chebyshev 的系数
            step_size = config['step_size']
            gamma = config['gamma']
            opt = torch.optim.Adam(model.parameters(), lr=lr)
            scheduler = StepLR(opt, step_size=step_size, gamma=gamma)
            plot_train_loss = []
            for epoch in trange(n_epoch, desc='Training'):
                train_loss = 0.0
                total_sample = 0

                once_batch = True
                # FIXME 
                for x_batch, y_batch, _, _ in tqdm(train_loader, desc=f'epoch {epoch+1} in training', leave=False):
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)

                    y_pred = model(x_batch).squeeze()
                    y_batch = y_batch[:, chebyshev_i, :, :].squeeze()
                        
                    # turn MSE to RMSE
                    loss = torch.sqrt(criterion(y_pred, y_batch))
                    # backward
                    opt.zero_grad()
                    loss.backward()
                    # update parameters
                    opt.step()
                    # record loss and accuracy
                    train_loss += loss.detach().cpu().item()
                    total_sample += len(x_batch)
                    if once_batch:
                        # we print 前10个样本的当前的预测值n-th order chebyshev alpha作为分析
                        print(f" y_pred : {y_pred[:10].flatten()}")
                        print(f"y_batch: {y_batch[:10].flatten()}")
                        once_batch = False
                scheduler.step()
                print(f'Epoch-{epoch+1} lr: ' + f"{opt.param_groups[0]['lr']}")

                train_loss = train_loss / len(train_loader)
                plot_train_loss.append(train_loss)
                if validate:
                    plot_validate_loss.append(validate(model, validate_loader=, chebyshev_i=))
                print(f" epoch : {epoch+1}/{n_epoch}  train RMSE loss: {train_loss:.10f}, train sample: {total_sample}")

            grp.create_dataset('train', data=plot_train_loss)

        loss_h5.close() 

def validate(model, validate_loader, chebyshev_i, criterion, device, config):
    # validate loop
    loss_f = config['loss_file']
    loss_h5 = h5py.File(loss_f, 'a')

    validate_loss = 0.0
    once_batch = True
    for x_batch, y_batch, _, _ in tqdm(validate_loader, desc='validate', leave=False):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        yv_pred = model(x_batch).squeeze()
        y_batch = y_batch[:, chebyshev_i, :, :].squeeze()
        loss_v = torch.sqrt(criterion(yv_pred, y_batch))

        validate_loss += loss_v.detach().cpu().item()
        if once_batch:
            # we print 前10个样本的当前的预测值n-th order chebyshev alpha作为分析
            print(f" y_pred : {yv_pred[:10].flatten()}")
            print(f"y_v: {y_batch[:10].flatten()}")
            once_batch = False

    validate_loss = validate_loss / len(validate_loader)
    return validate_loss





def retrain(model, train_loader, validate_loader, n_epoch, criterion, relr, device, model_range, config):
    re_epoch = config['re_epoch']
    loss_f = config['config_loss']
    loss_h5 = h5py.File(loss_f, 'w')
    for chebyshev_i in model_range:
        grp = loss_h5.require_group(f'model_{chebyshev_i:03}')
        
        # init model
        model_path = config['model_path']
        retrain_model = config['retrain_model']
        project_path = config['project_path']

        model.load_state_dict(torch.load(os.path.join(project_path, model_path, retrain_model)))
        model.to(device).double()
        # 第 i 个 model 预测 第i个chebyshev 的系数
        step_size = config['step_size']
        gamma = config['gamma']
        opt = torch.optim.Adam(model.parameters(), lr=relr)
        scheduler = StepLR(opt, step_size=step_size, gamma=gamma)

        plot_train_loss = []
        plot_validate_loss = []
        for epoch in trange(re_epoch, desc='Training'):
            train_loss = 0.0
            total_sample = 0

            once_batch = True
            for x_batch, y_batch, _ in tqdm(train_loader, desc=f'epoch {epoch+1} in training', leave=False):
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                y_pred = model(x_batch)
                y_batch = torch.squeeze(y_batch, dim=1)[:, chebyshev_i].flatten()
                # turn MSE to RMSE
                loss = torch.sqrt(criterion(y_pred, y_batch))
                # backward
                opt.zero_grad()
                loss.backward()
                # update parameters
                opt.step()
                # record loss and accuracy
                train_loss += loss.detach().cpu().item()
                total_sample += len(x_batch)
                if once_batch:
                    # we print 前10个样本的当前的预测值n-th order chebyshev alpha作为分析
                    print(f" y_pred : {y_pred[:10].flatten()}")
                    print(f"y_batch: {y_batch[:10].flatten()}")
                    once_batch = False
            train_loss = train_loss / len(train_loader)
            validate_loss = validate(validate_loader, model, chebyshev_i, criterion, device)
            plot_train_loss.append(train_loss)
            plot_validate_loss.append(validate_loss)
            # save every 10 epoch
            if (epoch + 1) % 10 == 0:
                # save checkpoint
                save_pt = os.path.join(project_path, model_path, f'e{epoch+1+n_epoch}_{chebyshev_i}th.pt')
                torch.save(model.state_dict(), save_pt)

            print(f" epoch : {epoch+1 + n_epoch}/{n_epoch + n_epoch} \
                  train RMSE loss: {train_loss:.10f}, train sample: {total_sample}")
            print(f" epoch : {epoch+1 + n_epoch}/{n_epoch + n_epoch} \
                  validate RMSE loss: {validate_loss:.10f}")
            
            scheduler.step()
            print(f' Epoch-{epoch+1+ n_epoch} lr: ' + f"{opt.param_groups[0]['lr']}")
        grp.create_dataset('retrain', data=plot_train_loss)
        grp.create_dataset('revalidate', data=plot_validate_loss)
        # save model
        save_pt = os.path.join(project_path, model_path, f'chebyshev_{chebyshev_i}_{n_epoch + re_epoch}.pt')
        torch.save(model.state_dict(),save_pt)
    loss_h5.close() 

# 改成class
def test(model, plot_loader,  model_range, criterion, device,model_checkpoint, config):
    loss_f = config['config_loss']
    loss_f = config['config_loss']
    model_path = config['model_path']
    project_path = config['project_path']

    loss_h5 = h5py.File(loss_f, 'a')
    finetune = bool(config['finetune'])
    pre_name=config["pre_name"]
    for chebyshev_i in model_range:
        grp = loss_h5.require_group(f'model_{chebyshev_i:03}')

        model_name = os.path.join(project_path, model_path, f'{pre_name}_{chebyshev_i}_{model_checkpoint}.pt')
        model.load_state_dict(torch.load(model_name))
        model.to(device).double()

        test_loss = 0.0
        test_sample = 0
        with torch.no_grad():
            for x_batch, y_batch, _ , _ in tqdm(plot_loader, desc=f'testing', leave=False):
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                y_pred = model(x_batch)           # (batch_n, 1)
                y_batch = torch.squeeze(y_batch, dim=1)[:, chebyshev_i]
                loss = torch.sqrt(criterion(y_pred, y_batch))
                test_loss += loss.detach().cpu().item()
                test_sample += len(x_batch)
                print(f'y_pred: {y_pred[:10].flatten()}')
                print(f'y_batch: {y_batch[:10].flatten()}')
                break
            # print(f"for {chebyshev_i}th order, test loss : {test_loss / test_sample:.10f}, test sample: {test_sample}")
            test_loss = test_loss / len(plot_loader)
            print(f"for {chebyshev_i}th order, test RMSE loss : {test_loss:.10f}, test sample: {test_sample}")
        if 'test' not in grp:
            grp.create_dataset('test', data=[test_loss])
    loss_h5.close()
