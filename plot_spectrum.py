import os
import numpy as np
import h5py
from tqdm import tqdm

import torch


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

    # compute mlp alphas
    for chebyshev_i in range(num_models):
        model_name = os.path.join(project_path, model_path, f'{pre_name}_{chebyshev_i}.pt')
        model.load_state_dict(torch.load(model_name))
        model.to(device).double()

        n_alphas = np.array([])
        with torch.no_grad():
            for para  in tqdm(plot_loader, desc=f'predict paras nn alphas', leave=False):
                para = para.to(device)
                # TODO  we should modify paras as dataloader to user predict data.
                y_pred = model(para)           # (batch_n, 1)
                y_np = y_pred.cpu().numpy()
                n_alphas = np.row_stack(
                    (n_alphas, y_np)) if n_alphas.size else y_np

        alphas = np.column_stack(
            (alphas, n_alphas)) if alphas.size else n_alphas

    return alphas
