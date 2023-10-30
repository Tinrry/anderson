import os, sys
import numpy as np
from quspin.basis import spinful_fermion_basis_1d, spinful_fermion_basis_general
from quspin.operators import hamiltonian, quantum_LinearOperator
import scipy.sparse as sp
import numexpr, cProfile
import matplotlib as mpl
import matplotlib.pyplot as plt

parameters = [[float(x) for x in d.strip().split(',')[1:]] for d in open('paras.csv').readlines()]

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

def spectral_function_fermion(omegas, paras, eta=0.6):
    # config
    L = 6  # system size
    # L = 4  # system size
    part = L // 2  # 根据对称性，进行参数缩减
    # paras = np.array([9.0, 0.0, 2.0, 1.8, 4.0, 0.0, 0.2, 0.0])

    # paras = np.array([9.0, 0.0, 2.0, 1.8, 0.0, 0.2])
    U = paras[0]
    ef = paras[1]
    eis_part = paras[2:2 + part]
    hoppings_part = paras[2 + part:]
    eis = np.concatenate((eis_part, -1 * eis_part))
    hoppings = np.concatenate((hoppings_part, hoppings_part))

    # hop_to_xxx， hop_from_xxx都是正符号，同一个系数，在hamiltonian中已经定好了符号
    hop_to_impurity = [[hoppings[i], 0, i + 1] for i in range(L)]
    hop_from_impurity = [[hoppings[i], i + 1, 0] for i in range(L)]
    pot = [[ef, 0]] + [[eis[i], i + 1] for i in range(L)]
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

    # spectral peaks broadening factor
    #eta = 0.9

    Green = np.zeros_like(omegas, dtype=np.complex128)

    cdagger_op = [["+|", [0], 1.0]]
    c_op = [["-|", [0], 1.0]]

    # this is for (3, 4) basis
    occupancy = L + 1
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

    # if 2 *(occupancy//2) != occupancy:
    if 0:
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




N = 7

paras = np.array(parameters[N])


# chebyshev parameters
order_n = 256
n = 250  # order (degree, highest power) of the approximating polynomial
# m = 113  # number of Chebyshev nodes (having m > n doesn't matter for the approximation it seems)
m = n + 1

x_min = -25
x_max = 25
x_grid = np.linspace(x_min, x_max, 1000)

r_k = np.polynomial.chebyshev.chebpts1(m)

# builds the Vandermonde matrix of Chebyshev polynomial expansion at the r_k nodes
# using the recurrence relation
T = np.polynomial.chebyshev.chebvander(r_k, n)

# calculate the Chebyshev coefficients
x_k = scale_up(r_k, x_min, x_max)



plt.figure(figsize=[8,4])
plt.plot(paras, '-.o')
plt.ylim([-5,10])
plt.grid()


omegas = x_k.copy()

Green = spectral_function_fermion(omegas, paras, eta=0.55)

plt.figure(figsize=[8,4])
plt.plot(omegas, Green)
plt.grid()
plt.xlabel('$\\omega$')
plt.ylabel('A')
plt.xlim([-12, 12])
plt.ylim([0, 0.45])
area = np.sum((omegas[1:] - omegas[0:-1]) * (Green[1:]+Green[0:-1]))/2.0
print(f"area is : {area}")


# NN = 2
# paras = np.array(parameters[NN])

# 这里的x_k的值是非均匀的值，所以面积算出来的不准
#y_k = spectral_function_fermion(x_k, paras)
y_k = Green
alpha = np.linalg.inv(T.T @ T) @ T.T @ y_k

# Use coefficients to compute an approximation of $f(x)$ over the grid of $x$:
T_pred = np.zeros((len(x_grid), n + 1))
T_pred[:, 0] = np.ones((len(x_grid), 1)).T
z_grid = scale_down(x_grid, x_min, x_max)
T_pred[:, 1] = z_grid.T
for i in range(1, n):
    T_pred[:, i + 1] = 2 * z_grid * T_pred[:, i] - T_pred[:, i - 1]
Tf = T_pred @ alpha

# this plot coefficients of the Chebyshev polynomial
plt.figure(figsize=[8,6])
plt.plot(alpha/alpha.max(), 'ro', ms=3)
plt.xlabel('i')
plt.ylabel('alpha')
plt.xlim([0, 256])
plt.grid()

plt.plot(T_pred @ alpha)
plt.plot(Tf)

plt.show()
