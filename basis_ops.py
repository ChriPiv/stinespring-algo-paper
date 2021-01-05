# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import autograd.numpy as np
from qiskit import *
from qiskit.quantum_info import *
from qiskit.providers.aer.utils import insert_noise

from channels import *

Id = np.array([[1., 0.],[0.,1.]])
X = np.array([[0.,1.],[1.,0.]])
Y = np.array([[0.,-1j],[1j,0.]])
Z = np.array([[1.,0.],[0.,-1.]])
H = np.array([[1.,1.],[1.,-1.]])/np.sqrt(2.)
P0 = np.array([[1.,0.],[0.,0.]])
P1 = np.array([[0.,0.],[0.,1.]])

class BasisOp:
    def __init__(self, name, u1, project=None, u2=None):
        self.name = name
        if project is None:
            self.is_unitary = True
            self.u1 = u1
            self.u = u1
        else:
            self.is_unitary = False
            self.u1 = u1
            self.u2 = u2 if u2 is not None else Id

def get_basis_ops(endo_unitaries=True, endo_projections=True):

    basis_ops = list()

    if endo_unitaries:
        basis_ops.append(BasisOp("Id", Id))
        basis_ops.append(BasisOp("X", X))
        basis_ops.append(BasisOp("Y", Y))
        basis_ops.append(BasisOp("Z", Z))
        basis_ops.append(BasisOp("Rx", (Id+1j*X)/np.sqrt(2.)))
        basis_ops.append(BasisOp("Ry", (Id+1j*Y)/np.sqrt(2.)))
        basis_ops.append(BasisOp("Rz", (Id+1j*Z)/np.sqrt(2.)))
        basis_ops.append(BasisOp("Ryz", (Y+Z)/np.sqrt(2.)))
        basis_ops.append(BasisOp("Rzx", (Z+X)/np.sqrt(2.)))
        basis_ops.append(BasisOp("Rxy", (X+Y)/np.sqrt(2.)))

    if endo_projections:
        basis_ops.append(BasisOp("PX", H, True, H))
        basis_ops.append(BasisOp("PY",
            np.array([[1.,-1j],[-1j,1.]])/np.sqrt(2.),
            True,
            np.array([[1.,1.],[1j,-1j]])/np.sqrt(2.)))
        basis_ops.append(BasisOp("PZ", Id, True))
        basis_ops.append(BasisOp("PYZ",
            np.array([[1j, -1j],[-1j, -1j]])/np.sqrt(2.),
            True,
            np.array([[1., -1j],[1., 1j]])/np.sqrt(2.)))
        basis_ops.append(BasisOp("PZX",
            np.array([[0.5+0.5j, -0.5+0.5j],[0.5-0.5j, -0.5-0.5j]]),
            True,
            np.array([[0.5-0.5j, 0.5+0.5j],[0.5+0.5j, 0.5-0.5j]])))
        basis_ops.append(BasisOp("PXY", X, True))

    return basis_ops


def noisy_unitary(u, noise_model, n_q=1):
    qc = QuantumCircuit(n_q)
    qc.unitary(u, list(range(n_q)))
    qc_noisy = insert_noise(qc, noise_model, transpile=True)
    return Choi(qc_noisy)

def apply_noise_model(basis_ops, noise_model):
    noisy_ops = list()

    for i in range(len(basis_ops)):
        op = basis_ops[i]
        if op.is_unitary:
            noisy_ops.append(noisy_unitary(op.u, noise_model).data)
        else:
            ch = noisy_unitary(op.u1, noise_model)
            ch = ch.compose(channel_project(1, 0))
            if op.u2 is not None:
                ch = ch.compose(noisy_unitary(op.u2, noise_model))
            noisy_ops.append(Choi(ch).data)
    return noisy_ops

def gen_two_qubit_basis(basis_ops):
    """
    computes tensor products.
    """
    n_ops = len(basis_ops)
    out = list()
    for i in range(n_ops):
        for j in range(n_ops):
            ch1 = Choi(basis_ops[i])
            ch2 = Choi(basis_ops[j])
            out.append( ch1.expand(ch2).data )

    return out

