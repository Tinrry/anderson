import numpy as np
from quspin.basis import spinful_fermion_basis_1d, spinful_fermion_basis_general
from quspin.operators import hamiltonian, quantum_LinearOperator
import scipy.sparse as sp
import matplotlib.pyplot as plt


np.random.seed(123)

def scale_up(z, x_min, x_max):
    """
    Scales up z \in [-1,1] to x \in [x_min,x_max]
    where z = (2 * (x - x_min) / (x_max - x_min)) - 1
    """

    return x_min + (z + 1) * (x_max - x_min) / 2


def scale_down(x, x_min, x_max):
    """
    Scales down x \in [x_min,x_max] to z \in [-1,1]
    where z = f(x) = (2 * (x - x_min) / (x_max - x_min)) - 1
    """

    return (2 * (x - x_min) / (x_max - x_min)) - 1

class LHS(sp.linalg.LinearOperator):
    #
    def __init__(self, H, omega, eta, E0, isparticle=True, kwargs={}):
        if isparticle:
            self._H = H  # Hamiltonian
            self._z = omega + 1j * eta + E0  # complex energy
        else: 
            self._H = -H  # Hamiltonian
            self._z = omega + 1j * eta - E0  # complex energy
        self._kwargs = kwargs  # arguments

    #
    @property
    def shape(self):
        return (self._H.Ns, self._H.Ns)

    #
    @property
    def dtype(self):
        return np.dtype(self._H.dtype)

    #
    def _matvec(self, v):
        # left multiplication
        return self._z * v - self._H.dot(v, **self._kwargs)

    #
    def _rmatvec(self, v):
        # right multiplication
        return self._z.conj() * v - self._H.dot(v, **self._kwargs)
    
    
class Anderson():
    def __init__(self, l, size):
        # system size
        self.L = l
        self.size = size

    @property
    def get_u(self, low=0, high=10):
        u = np.random.uniform(low=low, high=high, size=(1, self.size))
        return u
    
    @property
    def get_ef(self, low=-2.5, high=2.5):
        ef = np.random.uniform(low=low, high=high, size=(1, self.size))
        return ef
    
    @property 
    def get_ei(self, low=-5, high=5):
        ei_up = np.random.uniform(low=low, high=high, size=(self.L, self.size))
        ei_down = np.random.uniform(low=low, high=high, size=(self.L, self.size))
        return np.vstack((ei_up, ei_down))

    @property
    def get_ti(self, low=0, high=1.5):
        ti_up = np.random.uniform(low=low, high=high, size=(self.L, self.size))
        ti_down = np.random.uniform(low=low, high=high, size=(self.L, self.size))
        return np.vstack((ti_up, ti_down))

    def spectral_function_fermion(self, omegas, paras, eta=0.6):
        """
        params:
        paras: 当前处理的是(dim,) 后续要改成(N, dim)
        eta: spectral peaks broadening factor

        """
        # config
        part = self.L // 2  # 根据对称性，进行参数缩减
        # L=6, paras = np.array([9.0, 0.0, 2.0, 1.8, 4.0, 0.0, 0.2, 0.0])
        U = paras[0]
        ef = paras[1]
        eis_part = paras[2:2 + part]          # 不考虑spin_up, spin_down的对称性
        hoppings_part = paras[2 + part:]
        eis = np.concatenate((eis_part, -1 * eis_part))
        hoppings = np.concatenate((hoppings_part, hoppings_part))

        # hop_to_xxx， hop_from_xxx都是正符号，同一个系数，在hamiltonian中已经定好了符号
        hop_to_impurity = [[hoppings[i], 0, i + 1] for i in range(self.L)]
        hop_from_impurity = [[hoppings[i], i + 1, 0] for i in range(self.L)]
        pot = [[ef, 0]] + [[eis[i], i + 1] for i in range(self.L)]
        interaction = [[U, 0, 0]]
        # end config

        # 在符号上 都用‘-+’ 或者‘+-’，不可以掺杂
        static = [
            ['-+|', hop_from_impurity],
            ['-+|', hop_to_impurity],
            ['|-+', hop_from_impurity],
            ['|-+', hop_to_impurity],
            ['n|', pot],  # up on-site potention
            ['|n', pot],  # down on-site potention
            ['z|z', interaction]  # up-down interaction
        ]

        Green = np.zeros_like(omegas, dtype=np.complex128)

        cdagger_op = [["+|", [0], 1.0]]
        c_op = [["-|", [0], 1.0]]

        # this is for (3, 4) basis
        occupancy = self.L + 1
        N_up = occupancy // 2
        N_down = occupancy - N_up

        dynamic = []
        no_checks = dict(check_pcon=False, check_symm=False, check_herm=False)
        # construct basis
        basis_GS = spinful_fermion_basis_general(N=occupancy, Nf=(N_up, N_down))

        H0 = hamiltonian(static, dynamic, basis=basis_GS, dtype=np.float64, **no_checks)
        # calculate ground state
        [E0], GS = H0.eigsh(k=1, which="SA")
    
        # 产生算符，会导致电子增加，所以要加1
        basis_H1 = spinful_fermion_basis_general(N=occupancy, Nf=(N_up + 1, N_down))
        H1 = hamiltonian(static, [], basis=basis_H1, dtype=np.complex128, **no_checks)

        # shift sectors, |A> = c^\dagger |GS>
        psiA = basis_H1.Op_shift_sector(basis_GS, cdagger_op, GS)
        # solve (z-H)|x> = |A> solve for |x>  using iterative solver for each omega
        #   |x> = (z+E0-H)^-1 c^\dagger |GS>
        for i, omega in enumerate(omegas):
            lhs = LHS(H1, omega, eta, E0)
            x, exitCode = sp.linalg.bicg(lhs, psiA)
            assert exitCode == 0
            np.allclose(lhs._matvec(x), psiA)
            Green[i] += -np.vdot(psiA, x) / np.pi

        # 湮灭算符，会导致电子减少，所以要减1
        basis_H2 = spinful_fermion_basis_general(N=occupancy, Nf=(N_up - 1, N_down))
        H2 = hamiltonian(static, [], basis=basis_H2, dtype=np.complex128, **no_checks)

        # shift sectors, |A> = c |GS>
        psiA = basis_H2.Op_shift_sector(basis_GS, c_op, GS)
        # solve (z-H)|x> = |A> solve for |x>  using iterative solver for each omega
        #   |x> = (z-E0+H)^-1 c |GS>
        for i, omega in enumerate(omegas):
            lhs = LHS(H2, omega, eta, E0, isparticle=False)
            x, exitCode = sp.linalg.bicg(lhs, psiA)
            assert exitCode == 0
            np.allclose(lhs._matvec(x), psiA)
            Green[i] += -np.vdot(psiA, x) / np.pi

        # this is for (4, 3) basis

        if 2 *(occupancy//2) != occupancy:
            N_down = occupancy // 2
            N_up = occupancy - N_down

            dynamic = []
            no_checks = dict(check_pcon=False, check_symm=False, check_herm=False)

            # construct basis
            basis_GS = spinful_fermion_basis_general(N=occupancy, Nf=(N_up, N_down))

            H0 = hamiltonian(static, dynamic, basis=basis_GS, dtype=np.float64, **no_checks)
            # calculate ground state
            [E0], GS = H0.eigsh(k=1, which="SA")

            # 产生算符，会导致电子增加，所以要加1
            basis_H1 = spinful_fermion_basis_general(N=occupancy, Nf=(N_up + 1, N_down))
            H1 = hamiltonian(static, [], basis=basis_H1, dtype=np.complex128, **no_checks)

            # shift sectors, |A> = c^\dagger |GS>
            psiA = basis_H1.Op_shift_sector(basis_GS, cdagger_op, GS)
            # solve (z-H)|x> = |A> solve for |x>  using iterative solver for each omega
            #   |x> = (z+E0-H)^-1 c^\dagger |GS>
            for i, omega in enumerate(omegas):
                lhs = LHS(H1, omega, eta, E0)
                x, exitCode = sp.linalg.bicg(lhs, psiA)
                assert exitCode == 0
                np.allclose(lhs._matvec(x), psiA)
                Green[i] += -np.vdot(psiA, x) / np.pi
                
            # 湮灭算符，会导致电子减少，所以要减1
            basis_H2 = spinful_fermion_basis_general(N=occupancy, Nf=(N_up - 1, N_down))
            H2 = hamiltonian(static, [], basis=basis_H2, dtype=np.complex128, **no_checks)

            # shift sectors, |A> = c^ |GS>
            psiA = basis_H2.Op_shift_sector(basis_GS, c_op, GS)
            # solve (z-H)|x> = |A> solve for |x>  using iterative solver for each omega
            #   |x> = (z-E0+H)^-1 c |GS>
            for i, omega in enumerate(omegas):
                lhs = LHS(H2, omega, eta, E0, isparticle=False)
                x, exitCode = sp.linalg.bicg(lhs, psiA)
                assert exitCode == 0
                np.allclose(lhs._matvec(x), psiA)
                Green[i] += -np.vdot(psiA, x) / np.pi
            Green *= 0.5

        return Green[:].imag


class Chebyshev():
    def __init__(self, n=250, x_min=-25, x_max=25):
        """ chebyshev parameters
        n : order (degree, highest power) of the approximating polynomial
        m : number of Chebyshev nodes (having m > n doesn't matter for the approximation it seems)
        """
        self.n = n
        self.m = n+1
        self.x_min = x_min
        self.x_max = x_max

    def x_grid(self, size=1000):
        x_grid = np.linspace(self.x_min, self.x_max, size)
        return x_grid
    
    @property
    def r_k(self):
        r_k = np.polynomial.chebyshev.chebpts1(self.m)
        return r_k

    def T(self, r_k):
        # builds the Vandermonde matrix of Chebyshev polynomial expansion at the r_k nodes
        # using the recurrence relation
        T = np.polynomial.chebyshev.chebvander(r_k, self.n)
        return T

    def T_pred(self, z_grid):
        # Use coefficients to compute an approximation of $f(x)$ over the grid of $x$:
        T_pred = np.zeros((len(x_grid), self.n + 1))
        T_pred[:, 0] = np.ones((len(x_grid), 1)).T
        T_pred[:, 1] = z_grid.T
        for i in range(1, self.n):
            T_pred[:, i + 1] = 2 * z_grid * T_pred[:, i] - T_pred[:, i - 1]
        return T_pred
    

from numpy import savetxt, loadtxt
from tqdm import tqdm

if __name__ == "__main__":
    debug = False
    # PARAMETERS
    L, SIZE = 6, 5000
    N, X_MIN, X_MAX = 255, -25, 25
    training_size = int(SIZE * 0.8)       # training: testing = 8: 2
    training_file = f"L{L}N{N}_training{training_size}.csv"
    testing_file = f"L{L}N{N}_training{SIZE - training_size}.csv"

    # generate anderson model parameters
    model = Anderson(l=L, size=SIZE)     # band=3 , parameters_size = 3*2*2+2=14
    parameters = np.vstack((model.get_u, model.get_ef, model.get_ei, model.get_ti)).T     # shape(N, L+2)

    # calculate the Chebyshev coefficients
    chebyshev = Chebyshev(n=N, x_min= X_MIN, x_max= X_MAX)

    # compute setup
    x_k = scale_up(chebyshev.r_k, X_MIN, X_MAX)
    omegas = x_k.copy()
    x_grid = chebyshev.x_grid()
    z_grid = scale_down(x_grid, X_MIN, X_MAX)
    T_pred = chebyshev.T_pred(z_grid)

    Greens = np.array([])
    alphas = np.array([])
    Tfs = np.array([])
    for paras in tqdm(parameters, desc=f"generate data", leave=False):
        Green = model.spectral_function_fermion(omegas, paras, eta=0.55)
        Greens = np.row_stack((Greens, Green.T)) if Greens.size else Green

        # chebyshev & anderson translate
        # 拟合alpha
        y_k = Green
        T = chebyshev.T(chebyshev.r_k)
        alpha = np.linalg.inv(T.T @ T) @ T.T @ y_k
        alphas = np.row_stack((alphas, alpha.T)) if alphas.size else alpha

        # 计算Tf
        Tf = T_pred @ alpha
        Tfs = np.row_stack((Tfs, Tf.T)) if Tfs.size else Tf

    # 制作数据集
    dataset = np.concatenate((parameters, alphas, Greens), axis=1)

    savetxt(training_file, dataset[ : training_size], delimiter=',')
    savetxt(testing_file, dataset[training_size: ], delimiter=',')
    loaddata = loadtxt(training_file, delimiter=',')
    print(f"dataset[0] : {dataset[0]}")
    print(f"loaddata[0] : {loaddata[0]}")
    
    assert dataset.shape == (SIZE, L + 2 + 2 * (N + 1)), f'{dataset.shape} error'
    assert dataset[0].sum()-loaddata[0].sum()<0.0001, "save and load data not the same."
    
    if debug == True:
        fig, axs = plt.subplots(2)
        axs[0].set_title('two cases anderson and chebyshev spectrum')
        for i in range(2):
            axs[i].plot(x_grid, Tfs[i])
            axs[i].plot(omegas, Greens[i])
        plt.show()
        # inspect that area should be close to 1
        area = np.sum((omegas[1:] - omegas[0:-1]) * (Green[1:]+Green[0:-1]))/2.0
        print(f"area is : {area}")
        print(omegas[:10])
        print(Green[:10])
        print(20*'-')
        print(x_grid[:10])
        print(y_k[:10])

