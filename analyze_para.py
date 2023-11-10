import pandas as pd
from json import load
from tqdm import tqdm
import numpy as np
import h5py

from utils import load_config
from gen_train_data import Chebyshev, Anderson
from gen_train_data import scale_up, scale_down

test = False
config = load_config('config_L6_1.json')
L = config['L']
SIZE = config['SIZE']
N = config['N']
X_MIN = int(config['X_MIN'])
X_MAX = int(config['X_MAX'])
filename = config['spectrum_paras']

model = Anderson(l=L, size=SIZE)     # band=3 , parameters_size = 3*2*2+2=14
# calculate the Chebyshev coefficients
chebyshev = Chebyshev(n=N, x_min=X_MIN, x_max=X_MAX)
# compute setup
x_k = scale_up(chebyshev.r_k, X_MIN, X_MAX)
omegas = x_k.copy()
x_grid = chebyshev.x_grid()
z_grid = scale_down(x_grid, X_MIN, X_MAX)
T_pred = chebyshev.T_pred(z_grid)
T = chebyshev.T(chebyshev.r_k)

# check read data is precise
parameters = pd.read_csv(filepath_or_buffer='./datasets/paras.csv',
                         delimiter=',',
                         header=None,
                         index_col=0).values

Greens = np.array([])
Tfs = np.array([])
alphas = np.array([])
for para in tqdm(parameters, desc='analysis plot data', leave=False):
    Green = model.spectral_function_fermion(omegas, para, eta=0.55)
    Greens = np.row_stack((Greens, Green.T)) if Greens.size else Green
    if len(Greens.shape) == 1:
        Greens = np.expand_dims(Greens, axis=0)

    # chebyshev & anderson translate
    # 拟合alpha
    y_k = Green
    alpha = np.linalg.inv(T.T @ T) @ T.T @ y_k
    alphas = np.row_stack((alphas, alpha)) if alphas.size else alpha
    if len(alphas.shape) == 1:
        alphas = np.expand_dims(alphas, axis=0)

    # 计算Tf
    Tf = T_pred @ alpha
    Tfs = np.row_stack((Tfs, Tf.T)) if Tfs.size else Tf
    if len(Tfs.shape) == 1:
        Tfs = np.expand_dims(Tfs, axis=0)

# plot spectrum, (omegas, Greens), (x_grid, Tfs)
# for nn, we should keep T_pred,for predict
# i need to add chebyshev alphas in file to analysis results errors.
h5 = h5py.File(filename, 'w')
omegas = np.expand_dims(omegas, axis=0)
x_grid = np.expand_dims(x_grid, axis=0)
h5.create_dataset('omegas', data=omegas, dtype='float64')
h5.create_dataset('Greens', data=Greens, dtype='float64')
h5.create_dataset('x_grid', data=x_grid, dtype='float64')
h5.create_dataset('Tfs', data=Tfs, dtype='float64')
h5.create_dataset('T_pred', data=T_pred, dtype='float64')
h5.create_dataset('cheby_alphas', data=alphas, dtype='float64')
h5.close()

# test
if test:
    h5r = h5py.File(filename, 'r')
    omegas_r = h5r['omegas'][:]
    Greens_r = h5r['Greens'][:]
    x_grid_r = h5r['x_grid'][:]
    Tfs_r = h5r['Tfs'][:]
    T_pred_r = h5r['T_pred'][:]
    alphas_r = h5r['cheby_alphas'][:]
    print(f'{omegas_r.shape}')
    assert omegas_r[0, :5].sum() == omegas[0, :5].sum(), 'omegas not the same.'
    assert Greens_r[0, :].sum() == Greens[0, :].sum(), 'Greens not the same.'
    assert x_grid_r.sum() == x_grid.sum(), 'x_grid not the same.'
    assert all(Tfs_r.sum(axis=0) == Tfs.sum(axis=0)), 'Tfs not the same.'
    assert all(T_pred_r.sum(axis=0) == T_pred.sum(axis=0)), 'T_pred not the same.'
    assert all(alphas_r.sum(axis=0) == alphas.sum(axis=0)), 'alphas not the same.'
    h5r.close()
