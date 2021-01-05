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
from qiskit.quantum_info.operators.predicates import *

# Note on convention:
# We map |x> to |0x>
# In qiskit this means we append the ancillas after the previous qubits
# (see https://github.com/Qiskit/qiskit-terra/issues/1148)

def channel_expand(num_q, num_anc):
    dim_in = 2**num_q
    dim_out = 2**num_anc * 2**num_q

    if num_q == 1:
        # single kraus op: |0>|0><0| + |1>|0><1|
        mat = np.zeros((dim_out, dim_in))
        mat[0,0] = 1.
        mat[int(0.5*dim_out),1] = 1
        return Kraus([mat])

    elif num_q == 2:
        # single kraus op: |00>|0><00| + |01>|0><01| + |10>|0><10| + |11>|0><11|
        mat = np.zeros((dim_out, dim_in))
        mat[0,0] = 1.
        mat[int(0.25*dim_out),1] = 1
        mat[int(0.5*dim_out),2] = 1
        mat[int(0.75*dim_out),3] = 1
        return Kraus([mat])

    elif num_q == 3:
        mat = np.zeros((dim_out, dim_in))
        mat[0,0] = 1.
        mat[int(0.125*dim_out),1] = 1
        mat[int(0.25*dim_out),2] = 1
        mat[int(0.375*dim_out),3] = 1
        mat[int(0.5*dim_out),4] = 1
        mat[int(0.625*dim_out),5] = 1
        mat[int(0.75*dim_out),6] = 1
        mat[int(0.875*dim_out),7] = 1
        return Kraus([mat])


def channel_trace(num_q, num_anc):
    # kraus ops: Id x <x|
    kraus_ops = list()
    for x in range(2**num_anc):
        xbra = np.eye(1, 2**num_anc, x)
        kraus_ops.append(np.kron(np.identity(2**num_q), xbra))
    return Kraus(kraus_ops)

def modifiedGramSchmidt(A):
    # assuming A is a square matrix
    dim = A.shape[0]
    Q = np.zeros(A.shape, dtype=A.dtype)
    for j in range(0, dim):
        q = A[:,j]
        for i in range(0, j):
            rij = np.vdot(Q[:,i], q)
            q = q - rij*Q[:,i]
        rjj = np.linalg.norm(q, ord=2)
        if np.isclose(rjj,0.0):
            raise ValueError("invalid input matrix")
        else:
            Q[:,j] = q/rjj
    return Q

def complete_unitary(unitary):
    if unitary.shape[1] == 2:
        dim = unitary.shape[0]
        assert dim in [2, 4, 8]
        print(np.abs(np.dot(np.conj(unitary[:,0]), unitary[:,1])))
        #assert np.abs(np.dot(np.conj(unitary[:,0]), unitary[:,1])) < 1e-6
        if dim==2: return unitary

        # complete matrix
        onb = np.zeros((dim,dim), dtype=np.complex128)
        onb[:,0] = unitary[:,0]
        onb[:,1] = unitary[:,1]
        for i in range(2, dim):
            # Not really elegant, but it should work.
            onb[:,i] = np.random.normal(size=(dim))
        # generate orthonormal vectors
        onb = modifiedGramSchmidt(onb)
        # swap columns
        temp = onb[:,1].copy()
        onb[:,1] = onb[:,int(0.5*dim)].copy()
        onb[:,int(0.5*dim)] = temp

        U = onb
        #assert is_unitary_matrix(U)
        #assert np.linalg.norm(
        #            U @ np.eye(1,dim, 0).T - \
        #            unitary @ np.array([[1,0]]).T \
        #            ) < 1e-5
        #assert np.linalg.norm(
        #            U @ np.eye(1,dim, int(0.5*dim)).T - \
        #            unitary @ np.array([[0,1]]).T\
        #            ) < 1e-5
        return U

    if unitary.shape[1] == 4:
        dim = unitary.shape[0]
        assert dim in [4, 8, 16, 32, 64]
        if dim==4: return unitary

        # complete matrix
        onb = np.zeros((dim,dim), dtype=np.complex128)
        onb[:,0] = unitary[:,0]
        onb[:,1] = unitary[:,1]
        onb[:,2] = unitary[:,2]
        onb[:,3] = unitary[:,3]
        for i in range(4, dim):
            # Not really elegant, but it should work.
            onb[:,i] = np.random.normal(size=(dim))
        # generate orthonormal vectors
        onb = modifiedGramSchmidt(onb)
        # swap columns
        quarter = int(0.25*dim)
        def swap_col(col1, col2):
            temp = onb[:,col1].copy()
            onb[:,col1] = onb[:,col2].copy()
            onb[:,col2] = temp
        swap_col(3, 3*quarter)
        swap_col(2, 2*quarter)
        swap_col(1, quarter)

        U = onb
        assert is_unitary_matrix(U)
        #assert np.linalg.norm(
        #            U @ np.eye(1,dim, 0).T - \
        #            unitary @ np.array([[1,0,0,0]]).T \
        #            ) < 1e-6
        #assert np.linalg.norm(
        #            U @ np.eye(1,dim, quarter).T - \
        #            unitary @ np.array([[0,1,0,0]]).T \
        #            ) < 1e-6
        #assert np.linalg.norm(
        #            U @ np.eye(1,dim, 2*quarter).T - \
        #            unitary @ np.array([[0,0,1,0]]).T \
        #            ) < 1e-6
        #assert np.linalg.norm(
        #            U @ np.eye(1,dim, 3*quarter).T - \
        #            unitary @ np.array([[0,0,0,1]]).T \
        #            ) < 1e-6
        return U


def channel_project(n_qubits, proj_qubit):
    assert proj_qubit == 0 # TODO currently only this is supported
    # |0><0| x Id
    bra = np.array([[1.,0.],[0.,0.]])
    mat = np.kron(np.identity(2**(n_qubits-1)), bra)
    return Kraus([mat])

if __name__ == "__main__":
    ch = random_quantum_channel(2, 2)
    unitary = Stinespring(ch).data
    num_anc = int(np.log2(unitary.shape[0])) - 1

    exp = channel_expand(1, num_anc)
    qc = QuantumCircuit(1+num_anc)
    qc.unitary(complete_unitary(unitary), list(range(1+num_anc)))
    uni = SuperOp(qc)
    tr = channel_trace(1, num_anc)

    # test 1: check that expansion and tracing out cancel
    assert np.sum(np.abs( Choi(Kraus([np.identity(2)])).data - Choi(exp.compose(tr)).data )) < 1e-10

    # test 2: check full stinespring dilation
    ch2 = exp.compose(uni.compose(tr))
    assert np.sum(np.abs(Choi(ch).data - Choi(ch2).data)) < 1e-10

    # test3: projection
    qc1 = QuantumCircuit(2)
    qc1.reset(0)
    qc1.reset(1)
    qc1.h(0)
    qc1.x(1)
    qc1 = SuperOp(qc1)
    qc2 = QuantumCircuit(2)
    qc2.reset(0)
    qc2.reset(1)
    qc2.x(1)
    qc2 = SuperOp(qc2)

    proj = channel_project(2, 0)
    assert np.sum(np.abs( 0.5*Choi(qc2).data - Choi(qc1.compose(proj)).data )) < 1e-10


    # Now redo the same for a 2 qubit channel
    print("="*10)
    ch = random_quantum_channel(4, 4, rank=2)
    unitary = Stinespring(ch).data
    num_anc = int(np.log2(unitary.shape[0])) - 2

    exp = channel_expand(2, num_anc)
    qc = QuantumCircuit(2+num_anc)
    completed = complete_unitary(unitary)
    qc.unitary(completed, list(range(2+num_anc)))
    uni = SuperOp(qc)

    tr = channel_trace(2, num_anc)

    # test 1: check that expansion and tracing out cancel
    assert np.sum(np.abs( Choi(Kraus([np.identity(4)])).data - Choi(exp.compose(tr)).data )) < 1e-10

    # test 2: check full stinespring dilation
    ch2 = exp.compose(uni.compose(tr))
    assert np.sum(np.abs(Choi(ch).data - Choi(ch2).data)) < 1e-10


