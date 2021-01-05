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


def dnorm(c):
    dim = c.shape[0]
    halfdim = int(true_np.sqrt(dim))
    Y0 = cvxpy.Variable((dim,dim), hermitian=True)
    Y1 = cvxpy.Variable((dim,dim), hermitian=True)
    bigmatrix = cvxpy.bmat([[Y0, -c], [-true_np.conj(c.T),Y1]])
    constraints = [
        bigmatrix >> 0,
        Y0 >> 0,
        Y1 >> 0
    ]
    objective = 0.5*cvxpy.norm(cvxpy_partial_trace(Y0, (halfdim,halfdim), 1)) + 0.5*cvxpy.norm(cvxpy_partial_trace(Y1, (halfdim,halfdim), 1))
    
    prob = cvxpy.Problem(cvxpy.Minimize(objective), constraints)
    prob.solve(solver="MOSEK")
    assert prob.status == "optimal"
    return objective.value
