# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import scipy
from qiskit import *
from qiskit.quantum_info import *
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.utils import insert_noise
from qiskit.quantum_info.operators.predicates import is_unitary_matrix

from basis_ops import *
from decomposition import *
from channels import *
from json_tools import *
from burer_monteiro import burer_monteiro

def get_remainder(fixed_ops, n_qubits, target_choi, budget, disp=False):
    if len(fixed_ops) == 0: return target_choi, 1.

    coeffs,dn = bqpd_c(target_choi, fixed_ops, n_qubits, cp_constraint=False, c_fac_budget=budget, use_mosek=True)
    if disp: print('current dn:', dn, 'with c_factor', np.sum(np.abs(coeffs)))

    expanded = np.zeros_like(target_choi)
    for i in range(len(fixed_ops)):
        expanded += coeffs[i] * fixed_ops[i]
    return target_choi - expanded, dn

def get_new_ops(target_choi, n_qubits, disp=False):
    """ channel decomposition without rank constraint """
    # empirically I observe that this is enough
    n_ops = 1

    F_pos = list()
    a_pos = list()
    F_neg = list()
    a_neg = list()
    for i in range(n_ops):
        F_pos.append(cvxpy.Variable(target_choi.shape, hermitian=True))
        a_pos.append(cvxpy.Variable())
        F_neg.append(cvxpy.Variable(target_choi.shape, hermitian=True))
        a_neg.append(cvxpy.Variable())

    summed =  sum([F_pos[i] for i in range(n_ops)]) \
            + sum([F_neg[i] for i in range(n_ops)])

    constraints = list()
    constraints.append(target_choi == summed)
    for i in range(n_ops):
        constraints += [
            F_pos[i] >> 0,
            F_neg[i] << 0,
            cvxpy_partial_trace(F_pos[i], [2**n_qubits, 2**n_qubits], 1) == a_pos[i]*np.identity(2**n_qubits),
            cvxpy_partial_trace(F_neg[i], [2**n_qubits, 2**n_qubits], 1) == a_neg[i]*np.identity(2**n_qubits),
            ]

    objective = sum([a_pos[i] - a_neg[i] for i in range(n_ops)])
    prob = cvxpy.Problem(cvxpy.Minimize(objective), constraints)
    if disp: print('start solving (w/out rank constraint)...')
    prob.solve(solver="MOSEK", mosek_params={"MSK_DPAR_BASIS_TOL_X":1e-8})
    if disp: print('done!')
    assert prob.status == 'optimal'
    c_fac = objective.value

    # remove duplicates
    def rem_dupl(Flist, alist):
        for i in range(n_ops):
            for j in range(i+1, n_ops):
                if np.linalg.norm(Flist[i]/alist[i]-Flist[j]/alist[j]) < 1e-6:
                    Flist[j].value = np.zeros_like(target_choi)
                    alist[j].value = 0.
    rem_dupl(F_pos, a_pos)
    rem_dupl(F_neg, a_neg)

    # Determine entries to be fixed
    to_be_fixed_pos = list()
    to_be_fixed_neg = list()
    for i in range(n_ops):
        if a_pos[i].value > 1e-8:
            to_be_fixed_pos.append(F_pos[i].value / a_pos[i].value)
        if a_neg[i].value < -1e-8:
            to_be_fixed_neg.append(F_neg[i].value / a_neg[i].value)
    return to_be_fixed_pos, to_be_fixed_neg, c_fac

def get_mb_initial_guess(new_ops_pos, new_ops_neg, remainder, n_qubits, rank=2):
    """ get initial guess for rank-constrained channel decomposition """ 
    assert rank in [1,2]

    def decompose(mat):
        evals,evecs = np.linalg.eig(mat)
        assert np.max(np.abs(np.imag(evals))) < 1e-15
        evals = np.real(evals)
        #assert np.min(evals) > -1e-8
        order = np.flip(np.argsort(evals)) # index of eigenvalues largest to smallest

        mat_list = list()
        a_list = list()
        for i in range(int(0.5*len(evals)) if rank==2 else len(evals)):
            if rank == 1:
                m = np.hstack([
                    evecs[:,order[i]][np.newaxis].T * np.sqrt(evals[order[i]] + 0j),
                    ])
            elif rank == 2:
                m = np.hstack([
                    evecs[:,order[2*i  ]][np.newaxis].T * np.sqrt(evals[order[2*i  ]] + 0j),
                    evecs[:,order[2*i+1]][np.newaxis].T * np.sqrt(evals[order[2*i+1]] + 0j),
                    ])
            m = np.conj(m.T)
            expanded = np.conj(m.T) @ m
            a = np.real(np.sum(np.linalg.eigvals(expanded))) / 2**n_qubits
            mat_list.append(m)
            a_list.append(a)

        expanded_tot = np.zeros_like(mat)
        for i in range(len(mat_list)):
            expanded = np.conj(mat_list[i].T) @ mat_list[i]
            expanded_tot += expanded
            #assert np.abs( np.sum(np.linalg.eigvals(expanded/a_list[i])) - 2**n_qubits) < 1e-8
        #assert np.linalg.norm( expanded_tot - mat) < 1e-6

        return mat_list, a_list

    initial_guess = {}
    initial_guess["Y_pos"] = list()
    initial_guess["Y_neg"] = list()
    initial_guess["a_pos"] = list()
    initial_guess["a_neg"] = list()

    try:
        coeffs,_ = qpd(remainder, new_ops_pos+new_ops_neg, n_qubits)
    except:
        # Due to numerical inaccuracy, the above might fail
        coeffs,dn = bqpd_c(remainder, new_ops_pos+new_ops_neg, n_qubits, cp_constraint=False)
        assert dn < 1e-7

    Y_zero = np.zeros((rank,4**n_qubits))
    if len(new_ops_pos)>0:
        mat_list, a_list = decompose(new_ops_pos[0]*coeffs[0])
        for i in range(len(mat_list)):
            initial_guess["Y_pos"].append(mat_list[i])
            initial_guess["a_pos"].append(a_list[i])
    else:
        if n_qubits == 1:
            initial_guess["Y_pos"] += [Y_zero]*2
            initial_guess["a_pos"] += [0.]*2
        elif n_qubits == 2:
            initial_guess["Y_pos"] += [Y_zero]*8
            initial_guess["a_pos"] += [0.]*8

    if len(new_ops_neg)>0:
        mat_list, a_list = decompose(-new_ops_neg[0]*coeffs[len(new_ops_pos)])
        for i in range(len(mat_list)):
            initial_guess["Y_neg"].append(mat_list[i])
            initial_guess["a_neg"].append(a_list[i])
    else:
        if n_qubits == 1:
            initial_guess["Y_neg"] += [Y_zero]*2
            initial_guess["a_neg"] += [0.]*2
        elif n_qubits == 2:
            initial_guess["Y_neg"] += [Y_zero]*8
            initial_guess["a_neg"] += [0.]*8

    return initial_guess

def get_new_ops_burer_monteiro(target_choi, initial_guess, n_qubits, scale=None, cfac_tol=1.2, rank=2, n_decomp=8, disp=False):
    """ channel decomposition with rank constraint """
    if scale is not None:
        target_choi *= scale
        if initial_guess is not None:
            for i in range(len(initial_guess["Y_pos"])):
                initial_guess["Y_pos"][i] = initial_guess["Y_pos"][i] * scale
                initial_guess["Y_neg"][i] = initial_guess["Y_neg"][i] * scale
                initial_guess["a_pos"][i] = initial_guess["a_pos"][i] * scale
                initial_guess["a_neg"][i] = initial_guess["a_neg"][i] * scale

    if disp: print('start solving BM...')
    coeffs, arr_C = burer_monteiro(target_choi, n_decomp=n_decomp, rank=rank, n_qubits=n_qubits, initial_guess=initial_guess, cfac_tol=cfac_tol)
    if disp: print('done!')

    # remove duplicates
    def rem_dupl(Flist, alist):
        for i in range(2*n_decomp):
            for j in range(i+1, 2*n_decomp):
                if np.abs(alist[i])>1e-8 and np.abs(alist[j])>1e-8:
                    if np.linalg.norm(Flist[i]/alist[i]-Flist[j]/alist[j]) < 1e-6:
                        Flist[j] = np.zeros_like(target_choi)
                        alist[j] = 0.
    rem_dupl(arr_C, coeffs)

    # Determine entries to be fixed
    to_be_fixed = list()
    for i in range(2*n_decomp):
        if np.abs(coeffs[i]) > 1e-8:
            to_be_fixed.append(arr_C[i] / coeffs[i])
    if disp: print('BM used {}/{} ops.'.format(len(to_be_fixed), 2*n_decomp))

    return to_be_fixed, np.sum(np.abs(coeffs))


def get_stinespring_unitary(choi, target_choi, target_unitary, n_qubits):
    """ get the Stinespring dilation unitary given the Choi matrix choi"""
    unitary = Stinespring(Choi(choi)).data

    # this fixes some numerical issues
    if np.linalg.norm(choi.data - target_choi) < 1e-8:
        unitary = target_unitary

    if type(unitary) == tuple:
        assert np.linalg.norm(unitary[0]-unitary[1]) < 1e-8
        unitary = unitary[0]
    assert unitary.shape[1] == 2**n_qubits

    # increase hilbert space size to be a power of two
    next_pow_2 = int(np.power(2, np.ceil(np.log(unitary.shape[0])/np.log(2))))
    if unitary.shape[0] != next_pow_2:
        diff = next_pow_2 - unitary.shape[0]
        unitary = np.vstack([unitary, np.zeros((diff, 2**n_qubits))])
        assert unitary.shape[0] == next_pow_2
        assert unitary.shape[1] == 2**n_qubits
    num_anc = int(np.log2(unitary.shape[0])) - n_qubits

    completed = complete_unitary(unitary)
    # due to numerical issues, the matrix is not always perfectly unitary
    if not is_unitary_matrix(completed, atol=1e-12): completed,_ = scipy.linalg.polar(completed)

    return completed, num_anc


def stinespring_algorithm(target_unitary, n_qubits, noise_oracle, disp=True, dn_tol=1e-7, rank_constraint=2, bm_ops=8, cfac_tol=1.2, cfac_budget=1.5):
    """
    target_choi: Choi matrix of operation to be decomposed
    n_qubits: may be 1 or 2
    noise_oracle: function taking a unitary U and returning the choi matrix of the noisy execution of U
    disp: set to True to display debug info
    dn_tol: tolerance (diamond norm) when to stop iteraction
    rank_constraint: rank of Choi matrices. Set to None if no constraint shall be used
    bm_ops: how many ops to use in the Burer Monteiro expansion

    returns: list of stinespring unitaries, choi_matrices (noisy ones) and coefficients
    """
    target_choi = Choi(Kraus([target_unitary])).data

    fixed_unitaries = list()
    fixed_noisy_ops = list()
    while True:
        if disp: print("="*10)

        remainder, dn = get_remainder(fixed_noisy_ops, n_qubits, target_choi, cfac_budget, disp)
        # make sure remainder is hermitian and TP
        assert np.linalg.norm(remainder - np.conj(remainder.T)) < 1e-8
        assert np.linalg.norm(partial_trace(remainder, list(range(n_qubits))).data - (0.5/n_qubits)*np.trace(remainder)*np.identity(2**n_qubits)) < 1e-8
        # tolerance condition
        if dn < dn_tol: break

        # solve problem without rank constraint
        new_ops_pos, new_ops_neg, cfac = get_new_ops(remainder, n_qubits, disp=disp)
        if disp: print('Without rank constraint: got a solution with gamma factor', cfac)

        if rank_constraint is not None:
            # scale the problem for numerical reasons
            scale = 1. / cfac
            if disp: print('using scale',scale)

            initial_guess = get_mb_initial_guess(new_ops_pos, new_ops_neg, remainder, n_qubits, rank=rank_constraint)
            new_ops, cfac2 = get_new_ops_burer_monteiro(remainder, initial_guess, n_qubits, scale=scale, cfac_tol=cfac_tol, rank=rank_constraint, n_decomp=bm_ops, disp=disp)
            if disp: print('With rank constraint: got a solution with gamma factor', cfac2/scale)
        else:
            new_ops = new_ops_pos + new_ops_neg

        for op in new_ops:
            U, n_anc = get_stinespring_unitary(op, target_choi, target_unitary, n_qubits)
            N = noise_oracle(U, n_anc)
            fixed_unitaries.append(U)
            fixed_noisy_ops.append(N)
        if disp: print("added", len(new_ops), "new ops")
        if disp: print("="*10)

    # Run final qpd
    try:
        coeffs,_ = qpd(target_choi, fixed_noisy_ops, n_qubits)
    except:
        # The above might fail if dn_threshold is high
        coeffs,dn = bqpd_c(target_choi, fixed_noisy_ops, n_qubits, cp_constraint=False)
    if disp: print("FINAL RESULT: cfactor =", np.sum(np.abs(coeffs)))

    return fixed_unitaries, fixed_noisy_ops, coeffs

