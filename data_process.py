import os
import h5py
import numpy as np


def loss_mean(in_f, out_f, mean_f='datasets/c_norm_meta.h5'):
    in_h5 = h5py.File(in_f, 'r')
    mean_h5 = h5py.File(mean_f, 'r')
    c_mean = mean_h5['mean'][:]
    for model in in_h5.keys():
        idx = int(model.split('_')[1])
        print('model ',idx, f'c_mean  {c_mean[idx]}')
        train_grp = in_h5.require_group(model)
        train_loss = train_grp['train_loss_per_epoch'][:] / c_mean[idx][:]
        validate_loss = train_grp['validate_loss_per_epoch'][:] / c_mean[idx][:]
        test_loss = train_grp['test_loss_per_epoch'][:] / c_mean[idx]
        with h5py.File(out_f, 'w') as out_h5:
            g = out_h5.create_group(model)
            g.create_dataset(name='train_loss_per_epoch', data=train_loss)
            g.create_dataset(name='validate_loss_per_epoch', data=validate_loss)
            g.create_dataset(name='test_loss_per_epoch', data=test_loss)

        in_h5.close()
        mean_h5.close()