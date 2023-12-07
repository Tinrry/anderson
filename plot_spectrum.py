import os
import numpy as np
import h5py
from tqdm import tqdm

import torch

from json import load
import numpy as np
import h5py
import matplotlib.pyplot as plt


def get_alpha(model, plot_loader,  num_models, device=None, alpha_filename=None, config=None):
    alphas = np.array([])
    if alpha_filename:
        hf = h5py.File(alpha_filename, 'a')
        if 'mlp_alphas' not in hf.keys():
            alphas = _predict_alpha(model, plot_loader, num_models, device, config)
            hf.create_dataset(name='mlp_alphas', data=alphas, dtype='float64')
        hf.close()

        # read data
        hf = h5py.File(alpha_filename, 'r')
        alphas = hf['mlp_alphas'][:]
        hf.close()
    else:
        alphas = _predict_alpha(model, plot_loader, num_models, device, config)

    return alphas


def _predict_alpha(model, plot_loader,  num_models, device, config):
    alphas = np.array([])
    pre_name=config["pre_name"]
    model_path = config['model_path']
    project_path = config['project_path']
    plot_size = config['plot_size']

    # compute mlp alphas
    for chebyshev_i in range(num_models):
        model_name = os.path.join(project_path, model_path, f'{pre_name}_{chebyshev_i}.pt')
        model.load_state_dict(torch.load(model_name))
        model.to(device).double()

        n_alphas = np.array([])
        with torch.no_grad():
            # FIXME
            for anderson, _, _, _  in tqdm(plot_loader, desc=f'predict paras nn alphas', leave=False):
                anderson = anderson.to(device)
                # TODO  we should modify paras as dataloader to user predict data.
                y_pred = model(anderson)           # (batch_n, 1)
                y_np = y_pred.cpu().numpy()
                n_alphas = np.row_stack(
                    (n_alphas, y_np)) if n_alphas.size else y_np
                if len(n_alphas) > plot_size:
                    break

        alphas = np.column_stack(
            (alphas, n_alphas)) if alphas.size else n_alphas

    return alphas



def plot_loss_scale(filename, chebyshev_i=0):
    h5 = h5py.File(filename, 'r')
    fig = plt.figure()
    idx = 0
    loss_scale = np.array([0, 10, 20, 30])         # 前面的loss太高了
    
    for begin in loss_scale:
        if begin >= h5[f'model_{chebyshev_i:03}']['train'].shape[0]:
            break
        ax_i = fig.add_subplot(1, len(loss_scale), idx+1)
        train = h5[f'model_{chebyshev_i:03}']['train'][begin:]
        validate = h5[f'model_{chebyshev_i:03}']['validate'][begin:]
        # test = h5[f'model_{chebyshev_i:03}']['test'][:]
        ax_i.plot(np.array(range(len(train)))+1+begin, train, '-o', label='train loss')
        ax_i.plot(np.array(range(len(validate)))+1+begin, validate, '-o', label='validate loss')
        # ax_i.plot(len(train)+begin, test[0], '1', label='test loss')
        idx += 1
    fig.legend()
    savename = filename.split('.')[0]
    fig.savefig(savename + '.png')
    plt.show()
    h5.close()


def plot_retrain_loss_scale(filename, chebyshev_i=0):
    h5 = h5py.File(filename, 'r')
    fig = plt.figure()
    idx = 0
    loss_scale = np.array([0, 5, 10, 15, 20])         # 前面的loss太高了
    
    for begin in loss_scale:
        if begin >= h5[f'model_{chebyshev_i:03}']['retrain'].shape[0]:
            break
        ax_i = fig.add_subplot(1, len(loss_scale), idx+1)
        train = h5[f'model_{chebyshev_i:03}']['retrain'][begin:]
        validate = h5[f'model_{chebyshev_i:03}']['revalidate'][begin:]
        ax_i.plot(np.array(range(len(train)))+1+begin, train, '-o', label='train loss')
        ax_i.plot(np.array(range(len(validate)))+1+begin, validate, '-o', label='validate loss')
        idx += 1
    fig.legend()
    savename = filename.split('.')[0]
    fig.savefig(savename + '.png')
    plt.show()
    h5.close()
