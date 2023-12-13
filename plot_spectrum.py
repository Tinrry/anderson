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
    fig.text()
    savename = filename.split('.')[0]
    fig.savefig(savename + '.png')
    plt.show()
    h5.close()

# 公共函数
def plot_spectrum(spectrum_filename, nn_alpha, n_pic=1):
    # plot Greens
    # plot chebyshev, TF, by alphas--labels
    # plot chebyshev, TF, by alphas--nn-predict
    hf = h5py.File(spectrum_filename, 'r')
    # meta_len = config["N"] + 1
    omegas = hf['omegas'][:]
    T_pred = hf['T_pred'][:]
    x_grid = hf['x_grid'][:]
    Tfs = hf['Tfs'][:]
    Greens = hf['Greens'][:]

    # compute spectrum by alpha
    nn_Tfs = np.array([])
    for idx in range(len(Tfs)):
        nn_a = nn_alpha[idx]
        # compute chebyshev function
        nn_Tf = T_pred @ nn_a
        nn_Tfs = np.row_stack((nn_Tfs, nn_Tf)) if nn_Tfs.size else nn_Tf
        if len(nn_Tfs.shape) == 1:
            nn_Tfs = np.expand_dims(nn_Tfs, axis=0)

    fig = plt.figure()
    omegas = np.squeeze(omegas, axis=0)
    x_grid = np.squeeze(x_grid, axis=0)
    for i in range(n_pic):
        axs = fig.add_subplot(1, n_pic, i+1)
        axs.plot(omegas, Greens[i], color='r')
        axs.plot(x_grid, Tfs[i], color='g')
        axs.plot(x_grid, nn_Tfs[i], color='b')
    fig.suptitle('Greens, cheby_Tfs, nn_Tfs, spectrum plot')
    plt.show()




def plot_loss(train_loss, valitate_loss, test_loss: int):
    steps = np.array(range(len(train_loss)))
    plt.plot(steps+1, train_loss, '-o', label='train loss')
    plt.plot(steps+1, valitate_loss, '-o', label='validate loss')
    plt.plot(steps[-1]+1, test_loss, '1', label='test loss')
    plt.legend()
    plt.show()

def plot_loss_from_h5(filename,ncols=5, nrows=4):
    h5 = h5py.File(filename, 'r')
    fig = plt.figure()
    idx = 0
    for i in range(1, nrows+1):
        for j in range(1, ncols+1):
                ax_i = fig.add_subplot(nrows, ncols, idx+1)
                train = h5[f'model_{idx:03}']['train'][:]
                validate = h5[f'model_{idx:03}']['validate'][:]
                test = h5[f'model_{idx:03}']['test'][:]
                ax_i.plot(np.array(range(len(train)))+1, train, '-o', label='train loss')
                ax_i.plot(np.array(range(len(validate)))+1, validate, '-o', label='validate loss')
                ax_i.plot(len(train), test[0], '1', label='test loss')
                idx += 1
                ax_i.legend()
    plt.show()
    h5.close()
