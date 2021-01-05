# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import autograd
import autograd.numpy as np
from qiskit.quantum_info import *
import scipy
from scipy.optimize import *

from decomposition import *

#Goal:
#C_i s.t.
#* C_i hermitian
#* C_i>>0
#* tr_2[C_i] = Id*a_i
#* sum(C_i) == ideal_choi
# minimize sum(a_i)

def anp_partial_trace(rho, dims, axis=0):
    """
    Takes partial trace over the subsystem defined by 'axis'
    rho: a matrix
    dims: a list containing the dimension of each subsystem
    axis: the index of the subsytem to be traced out
    (We assume that each subsystem is square)
    """
    dims_ = np.array(dims)
    reshaped_rho = np.reshape(rho, np.concatenate((dims_, dims_), axis=None))

    reshaped_rho = np.moveaxis(reshaped_rho, axis, -1)
    reshaped_rho = np.moveaxis(reshaped_rho, len(dims)+axis-1, -1)

    assert reshaped_rho.shape[-1] == reshaped_rho.shape[-2]
    traced_out_rho = sum([reshaped_rho[:,:,i,i] for i in range(reshaped_rho.shape[-1])])

    dims_untraced = np.delete(dims_, axis)
    rho_dim = np.prod(dims_untraced)
    return traced_out_rho.reshape([rho_dim, rho_dim])

def burer_monteiro(target_choi, n_decomp, rank, n_qubits, initial_guess=None, cfac_tol=1.):
    choi_dim = target_choi.shape[0]

    # expanding and flattening
    entries = choi_dim*rank
    def extract_matrix(x):
        mat_re = x[0:entries].reshape((rank,choi_dim))
        mat_im = x[entries:2*entries].reshape((rank,choi_dim))
        return mat_re+1j*mat_im

    matlen = 2*entries
    def expand(x):
        arr_Y_pos = list()
        arr_Y_neg = list()
        arr_a_pos = list()
        arr_a_neg = list()
        for i in range(n_decomp):
            arr_Y_pos.append(extract_matrix(x[matlen*i : matlen*(i+1)]))
        for i in range(n_decomp):
            arr_Y_neg.append(extract_matrix(x[matlen*(n_decomp+i) : matlen*(n_decomp+i+1)]))
        for i in range(n_decomp):
            arr_a_pos.append(x[matlen*2*n_decomp + i])
            arr_a_neg.append(x[matlen*2*n_decomp + n_decomp + i])
        return arr_Y_pos, arr_Y_neg, arr_a_pos, arr_a_neg


    def flatten_matrix(mat):
        mat_re = np.real(mat).flatten()
        mat_im = np.imag(mat).flatten()
        return [mat_re, mat_im]

    def flatten(arr_Y_pos, arr_Y_neg, arr_a_pos, arr_a_neg):
        tot_list = list()
        for Y in arr_Y_pos:
            tot_list += flatten_matrix(Y)
        for Y in arr_Y_neg:
            tot_list += flatten_matrix(Y)
        tot_list += arr_a_pos
        tot_list += arr_a_neg

        return np.hstack(tot_list)


    # optimization function
    def loss(x):
        arr_Y_pos, arr_Y_neg, arr_a_pos, arr_a_neg = expand(x)
        return np.sum(np.abs(arr_a_pos)) + np.sum(np.abs(arr_a_neg))

    def constraint(x):
        arr_Y_pos, arr_Y_neg, arr_a_pos, arr_a_neg = expand(x)
        arr_C_pos = list()
        arr_C_neg = list()
        def conj(z): return np.real(z) - 1j * np.imag(z)
        for i in range(n_decomp):
            arr_C_pos.append(conj(arr_Y_pos[i].T) @ arr_Y_pos[i])
            arr_C_neg.append(conj(arr_Y_neg[i].T) @ arr_Y_neg[i])

        retvec = np.array([])

        # TP constraint
        for i in range(n_decomp):
            pt = anp_partial_trace(arr_C_pos[i], [2**n_qubits,2**n_qubits], 1)
            vec = (pt - arr_a_pos[i]*np.identity(2**n_qubits)).flatten()
            retvec = np.hstack([retvec, vec])

            pt = anp_partial_trace(arr_C_neg[i], [2**n_qubits,2**n_qubits], 1)
            vec = (pt - arr_a_neg[i]*np.identity(2**n_qubits)).flatten()
            retvec = np.hstack([retvec, vec])

        # equality constraint
        C_sum = np.zeros_like(target_choi)
        for i in range(n_decomp):
            C_sum += arr_C_pos[i] - arr_C_neg[i]
        vec = (C_sum - target_choi).flatten()
        retvec = np.hstack([retvec, vec])

        # separate complex and real part
        retvec = np.hstack([np.real(retvec), np.imag(retvec)])
        return retvec

    constraint_jac = autograd.jacobian(constraint)
    constraint_hess = autograd.hessian(lambda x,v : np.dot(constraint(x), v), argnum=0)

    # initial guess
    res = minimize(lambda z: np.linalg.norm(np_partial_trace(target_choi,[2**n_qubits,2**n_qubits], 1).data - z*np.eye(2**n_qubits)), [1.])
    scale = res.x
    #assert res.fun < 1e-6
    arr_Y_pos = list()
    arr_Y_neg = list()
    arr_a_pos = list()
    arr_a_neg = list()
    if initial_guess is not None:
        for i in range(n_decomp):
            arr_Y_pos.append(initial_guess["Y_pos"][i])
            arr_Y_neg.append(initial_guess["Y_neg"][i])
            arr_a_pos.append(initial_guess["a_pos"][i])
            arr_a_neg.append(initial_guess["a_neg"][i])
    else:
        for i in range(n_decomp):
            arr_Y_pos.append(scale*np.random.normal(size=(rank,choi_dim)) + 1j*scale*np.random.normal(size=(rank,choi_dim)))
            arr_Y_neg.append(scale*np.random.normal(size=(rank,choi_dim)) + 1j*scale*np.random.normal(size=(rank,choi_dim)))
            arr_a_pos.append(scale*np.random.uniform())
            arr_a_neg.append(scale*np.random.uniform())
    x0 = flatten(arr_Y_pos, arr_Y_neg, arr_a_pos, arr_a_neg)
    len_x0 = x0.shape[0]

    # check flatten+expand
    Yp,Yn,ap,an = expand(x0)
    for i in range(n_decomp):
        assert np.linalg.norm(arr_Y_pos[i] - Yp[i]) < 1e-10
        assert np.linalg.norm(arr_Y_neg[i] - Yn[i]) < 1e-10
        assert np.linalg.norm(arr_a_pos[i] - ap[i]) < 1e-10
        assert np.linalg.norm(arr_a_neg[i] - an[i]) < 1e-10

    # solve
    def new_loss(x): return np.sum(np.square(constraint(x)))
    new_loss_grad = autograd.grad(new_loss)

    lc_mat_dense = np.zeros((1, x0.shape[0]))
    lc_mat_dense[0,matlen*2*n_decomp:] = np.ones((n_decomp*2))
    indices_x = np.zeros((n_decomp*2))
    indices_y = list(range(matlen*2*n_decomp, x0.shape[0]))
    vals = np.ones((n_decomp*2))
    lc_mat = scipy.sparse.csr_matrix((vals, (indices_x, indices_y)), shape=(1, x0.shape[0]))
    assert np.linalg.norm(lc_mat_dense - lc_mat.toarray()) < 1e-10


    if np.max(np.abs(constraint(x0))) < 1e-8 and np.abs(loss(x0) - 1.) < 1e-8:
        res = OptimizeResult()
        res.x = x0
    else:
        con = LinearConstraint(lc_mat, 1., cfac_tol)
        res = minimize(new_loss, x0, jac=new_loss_grad, constraints=con, options={"verbose":0, "maxiter":10000000, "gtol":1e-12, "xtol":1e-16}, method='trust-constr')
        #assert np.max(np.abs(constraint(res.x))) < 1e-6

    # return
    arr_Y_pos, arr_Y_neg, arr_a_pos, arr_a_neg = expand(res.x)
    arr_C_pos = list()
    arr_C_neg = list()
    for i in range(n_decomp):
        arr_C_pos.append(np.conj(arr_Y_pos[i].T) @ arr_Y_pos[i])
        arr_C_neg.append(np.conj(arr_Y_neg[i].T) @ arr_Y_neg[i])

    return arr_a_pos + arr_a_neg, arr_C_pos+arr_C_neg
