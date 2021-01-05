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
from scipy.optimize import minimize
from qiskit import *
from qiskit.quantum_info import *
from qiskit.aqua.components.variational_forms import *
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.utils import insert_noise

from json_tools import *
from channels import *
import autograd.numpy as np

REDUCED_UNITARY = True

# circuit construction routines
def rx(param):
    return np.array(
            [[np.cos(param*0.5),     -1j * np.sin(param*0.5)],
             [-1j*np.sin(param*0.5), np.cos(param*0.5)]])
def ry(param):
    return (1.+0j)*np.array(
            [[np.cos(param*0.5), -np.sin(param*0.5)],
             [np.sin(param*0.5), np.cos(param*0.5)]])

def rz(param):
    return np.array(
            [[1., 0.],
             [0, np.exp(1j*param)]])

def TP(matrices):
    if len(matrices)==1: return matrices[0]
    retval = np.kron(matrices[1], matrices[0])
    for i in range(2, len(matrices)):
        retval = np.kron(matrices[i], retval)
    return retval

P0 = np.array([[1,0],[0,0]])
P1 = np.array([[0,0],[0,1]])
Id = np.array([[1,0],[0,1]])
X  = np.array([[0,1],[1,0]])
CX = TP([P0,Id]) + TP([P1,X])
CXI = TP([P0,Id,Id]) + TP([P1,X,Id])
CIX = TP([P0,Id,Id]) + TP([P1,Id,X])
ICX = TP([Id,P0,Id]) + TP([Id,P1,X])

def get_varform_unitary(params, n_qubits, depth, full_connectivity=True):
    num_parameters = (depth + 1) * 2 * n_qubits
    unitary = np.eye(2**n_qubits)

    def ryrz_row(param_idx):
        val1 = TP([ ry(params[param_idx+2*i]) for i in range(n_qubits) ])
        val2 = TP([ rz(params[param_idx+2*i+1]) for i in range(n_qubits) ])
        return val2 @ val1
    def entanglement_row():
        if full_connectivity:
            if n_qubits == 1: return np.identity(2)
            elif n_qubits == 2: return CX
            elif n_qubits == 3: return ICX @ CIX @ CXI
        else:
            if n_qubits == 1: return np.identity(2)
            elif n_qubits == 2: return CX
            elif n_qubits == 3: return ICX @ CXI
        raise

    param_idx = 0
    unitary = ryrz_row(param_idx) @ unitary
    param_idx += 2*n_qubits
    for i in range(depth):
        unitary = entanglement_row() @ unitary
        unitary = ryrz_row(param_idx) @ unitary
        param_idx += 2*n_qubits

    assert param_idx == num_parameters
    return unitary

def get_varform_circuit(params, n_qubits, depth, full_connectivity=True):
    entanglement = 'full' if full_connectivity else 'linear'
    varform = RYRZ(n_qubits, depth=depth, entanglement_gate='cx', entanglement=entanglement)
    circ = varform.construct_circuit(params)
    # remove barriers
    for i in reversed(range(len(circ.data))):
        if type(circ.data[i][0]) == qiskit.circuit.barrier.Barrier:
            del circ.data[i]

    return circ


# optimization routines
def error_l2(u1, u2, n_qubits):
    def norm(x): return np.mean( np.square(np.real(x)) + np.square(np.imag(x)) )
    if n_qubits == 2:
        return norm(u1[:,0] - u2[:,0]) + \
                norm(u1[:,2] - u2[:,2])
    elif n_qubits == 3:
        return norm(u1[:,0] - u2[:,0]) + \
                norm(u1[:,2] - u2[:,2]) + \
                norm(u1[:,4] - u2[:,4]) + \
                norm(u1[:,6] - u2[:,6])
    else: raise

def error_mean(u1, u2, n_qubits):
    if not REDUCED_UNITARY: return np.mean(np.abs(u1-u2))
    if n_qubits == 2:
        return np.mean(np.abs(u1[:,0] - u2[:,0])) + \
                np.mean(np.abs(u1[:,2] - u2[:,2]))
    elif n_qubits == 3:
        return np.mean(np.abs(u1[:,0] - u2[:,0])) + \
                np.mean(np.abs(u1[:,2] - u2[:,2])) + \
                np.mean(np.abs(u1[:,4] - u2[:,4])) + \
                np.mean(np.abs(u1[:,6] - u2[:,6]))
    else: raise

def get_approx_circuit(unitary, n_qubits, depth, full_connectivity=True):
    def loss(x):
        ux = get_varform_unitary(x, n_qubits, depth, full_connectivity)
        err = error_l2(ux, unitary, n_qubits)
        return err
    loss_grad = autograd.grad(loss)
    loss_hess = autograd.hessian(loss)

    num_parameters = (depth + 1) * 2 * n_qubits
    bestval = 1e10
    for _ in range(5):
        x0 = np.random.uniform(low=0., high=2.*np.pi, size=(num_parameters))
        res = minimize(loss, x0, jac=loss_grad, options={"maxiter":int(1e8)}, method='BFGS')
        if res.fun < bestval:
            bestval = res.fun
            best = res.x

    u_final = get_varform_unitary(best, n_qubits, depth, full_connectivity)
    return u_final, best


if __name__ == "__main__":
    full_connectivity = True
    n_qubits = 3
    depth = 10
    num_parameters = (depth + 1) * 2 * n_qubits
    x0 = np.random.uniform(low=0., high=2.*np.pi, size=(num_parameters))
    U = get_varform_unitary(x0, n_qubits, depth, full_connectivity)
    circ = get_varform_circuit(x0, n_qubits, depth, full_connectivity)
    U_ref = Operator(circ).data
    assert np.linalg.norm(U-U_ref) < 1e-10

    full_connectivity = False
    U = get_varform_unitary(x0, n_qubits, depth, full_connectivity)
    circ = get_varform_circuit(x0, n_qubits, depth, full_connectivity)
    U_ref = Operator(circ).data
    assert np.linalg.norm(U-U_ref) < 1e-10
    print("done.")
