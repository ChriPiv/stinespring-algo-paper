# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import numpy as true_np
import cvxpy
from qiskit.quantum_info import *
from cvxpy.expressions.expression import Expression

def expr_as_np_array(cvx_expr):
    if hasattr(cvx_expr, 'shape'):
        shape = cvx_expr.shape
    else:
        shape = cvx_expr.size

    if cvx_expr.is_scalar():
        return true_np.array(cvx_expr)
    elif len(shape) == 1:
        return true_np.array([v for v in cvx_expr])
    else:
        # then cvx_expr is a 2d array
        rows = []
        for i in range(shape[0]):
            row = [cvx_expr[i,j] for j in range(shape[1])]
            rows.append(row)
        arr = true_np.array(rows)
        return arr


def np_array_as_expr(np_arr):
    aslist = np_arr.tolist()
    expr = cvxpy.bmat(aslist)
    return expr


def np_partial_trace(rho, dims, axis=0):
    dims_ = true_np.array(dims)
    reshaped_rho = true_np.reshape(rho, true_np.concatenate((dims_, dims_), axis=None))

    reshaped_rho = true_np.moveaxis(reshaped_rho, axis, -1)
    reshaped_rho = true_np.moveaxis(reshaped_rho, len(dims)+axis-1, -1)

    traced_out_rho = true_np.trace(reshaped_rho, axis1=-2, axis2=-1)

    dims_untraced = true_np.delete(dims_, axis)
    rho_dim = true_np.prod(dims_untraced)
    return traced_out_rho.reshape([rho_dim, rho_dim])


def cvxpy_partial_trace(rho, dims, axis=0):
    if not isinstance(rho, Expression):
        rho = cvxpy.Constant(shape=rho.shape, value=rho)
    rho_np = expr_as_np_array(rho)
    traced_rho = np_partial_trace(rho_np, dims, axis)
    traced_rho = np_array_as_expr(traced_rho)
    return traced_rho

def cvxpy_partial_trace_re(rho_re, rho_im, dims, axis=0):
    if not isinstance(rho_re, Expression):
        rho_re = cvxpy.Constant(shape=rho_re.shape, value=rho_re)
    if not isinstance(rho_im, Expression):
        rho_im = cvxpy.Constant(shape=rho_im.shape, value=rho_im)
    rho_np_re = expr_as_np_array(rho_re)
    rho_np_im = expr_as_np_array(rho_im)
    traced_rho = true_np.real(np_partial_trace(rho_np_re, dims, axis)) \
               - true_np.imag(np_partial_trace(rho_np_im, dims, axis))
    traced_rho = np_array_as_expr(traced_rho)
    return traced_rho

def cvxpy_partial_trace_im(rho_re, rho_im, dims, axis=0):
    if not isinstance(rho_re, Expression):
        rho_re = cvxpy.Constant(shape=rho_re.shape, value=rho_re)
    if not isinstance(rho_im, Expression):
        rho_im = cvxpy.Constant(shape=rho_im.shape, value=rho_im)
    rho_np_re = expr_as_np_array(rho_re)
    rho_np_im = expr_as_np_array(rho_im)
    traced_rho = true_np.imag(np_partial_trace(rho_np_re, dims, axis)) \
               + true_np.real(np_partial_trace(rho_np_im, dims, axis))
    traced_rho = np_array_as_expr(traced_rho)
    return traced_rho


def qpd(choi_ideal, choi_mats, n_qubits, eps=1e-6):
    """ LP for optimal QPD from Temme et al. """
    n_ops = len(choi_mats)

    a = list()
    for i in range(n_ops): a.append(cvxpy.Variable())

    summed = sum([a[i]*choi_mats[i] for i in range(n_ops)])
    constraints = [
        summed == choi_ideal
    ]
    loss = sum([cvxpy.abs(a[i]) for i in range(n_ops)])

    prob = cvxpy.Problem(cvxpy.Minimize(loss), constraints)
    prob.solve(solver="SCS", eps=eps, max_iters=int(1e8))
    assert prob.status == "optimal"
    return [float(a[i].value) for i in range(n_ops)], loss.value

def bqpd_c(choi_ideal, choi_mats, n_qubits, c_fac_budget=None, cp_constraint=True, tp_constraint=False, eps=1e-6, use_mosek=False):
    """ approximate QPD with gamma factor factor budget c_fac_budget"""
    n_ops = len(choi_mats)
    dim = choi_ideal.shape[0]
    halfdim = int(true_np.sqrt(dim))
    Y0 = cvxpy.Variable((dim,dim), hermitian=True)
    Y1 = cvxpy.Variable((dim,dim), hermitian=True)
    a = list()
    for i in range(n_ops): a.append(cvxpy.Variable())

    C1 = sum([a[i]*choi_mats[i] for i in range(n_ops)])
    C = C1 - choi_ideal
    bigmatrix = cvxpy.bmat([[Y0, -C],[-cvxpy.conj(C.T), Y1]])
    constraints = [
        bigmatrix >> 0
    ]
    if cp_constraint: constraints.append( C1 >> 0)
    if tp_constraint: constraints.append( cvxpy_partial_trace(C1, (2**n_qubits,2**n_qubits), 1) == true_np.identity(2**n_qubits) )
    if c_fac_budget is not None:
        constraints.append( sum([cvxpy.abs(a[i]) for i in range(n_ops)]) <= c_fac_budget )
    loss = 0.5*cvxpy.norm(cvxpy_partial_trace(Y0, (halfdim,halfdim), 1)) + 0.5*cvxpy.norm(cvxpy_partial_trace(Y1, (halfdim,halfdim), 1))

    prob = cvxpy.Problem(cvxpy.Minimize(loss), constraints)
    if use_mosek:
        prob.solve(solver="MOSEK")
    else:
        prob.solve(max_iters=int(1e8), eps=eps, solver="SCS")
    assert prob.status == "optimal"
    return [float(a[i].value) for i in range(n_ops)], loss.value
