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

sys.path.append("..")
from json_tools import *
from basis_ops import *
from decomposition import *
from channels import *
from stinespring import stinespring_algorithm
from variational_approximation import get_approx_circuit, get_varform_circuit

qc = QuantumCircuit(2)
qc.swap(0,1)
target_unitary = Operator(qc).data
target_choi = Choi(qc).data
n_qubits = 2

noise_model = NoiseModel.from_dict(json_from_file("2020_04_08.json"))
noise_model.add_quantum_error(noise_model._local_quantum_errors['cx']['2,3'], 'cx', [0,2])
noise_model.add_quantum_error(noise_model._local_quantum_errors['cx']['3,2'], 'cx', [2,0])

depth = 6
cfac_budget = None
full_connectivity = False

saved_circuits = list()
def noise_oracle(U, num_anc):
    if num_anc == 0:
        qc = QuantumCircuit(n_qubits)
        if U.shape[0]==4 and np.linalg.norm(U-target_unitary) < 1e-8:
            qc.swap(0,1)
        else:
            qc.unitary(U, list(range(n_qubits)))
        qc = qiskit.compiler.transpile(qc, basis_gates=noise_model.basis_gates,
                                           coupling_map=[[0,1]])
        saved_circuits.append(qc)
        qc_noisy = insert_noise(qc, noise_model)
        return Choi(qc_noisy).data
    elif num_anc == 1:
        exp = channel_expand(n_qubits, num_anc)
        tr = channel_trace(n_qubits, num_anc)
        _,params = get_approx_circuit(U, n_qubits+num_anc, depth, full_connectivity)
        qc = get_varform_circuit(params, n_qubits+num_anc, depth, full_connectivity)
        coupling_map = [[0,1],[1,2],[0,2]] if full_connectivity else [[0,1],[1,2]]
        qc = qiskit.compiler.transpile(qc, basis_gates=noise_model.basis_gates,
                                           coupling_map=coupling_map)
        saved_circuits.append(qc)
        qc_noisy = insert_noise(qc, noise_model)
        qc_noisy = SuperOp(qc_noisy)
        return Choi( exp.compose(qc_noisy.compose(tr)) ).data
    else: raise



# Stinespring algorithm
fixed_unitaries, fixed_choi, coeffs = stinespring_algorithm(target_unitary, n_qubits, noise_oracle, disp=True, cfac_tol=1.2, bm_ops=8, cfac_budget=cfac_budget)
print("STINESPRING:", np.sum(np.abs(coeffs)))
#np.savez("noisemodel_cnot.npz", fixed_unitaries, fixed_choi, coeffs)
#for i in range(len(saved_circuits)):
#    saved_circuits[i].qasm(filename="data/sim_circ{}.qasm".format(i))

# Endo basis reference
endo_ops = list()
basis_ops = get_basis_ops(endo_unitaries=True, endo_projections=True)
def noisy_unitary(u, n_q=1):
    qc = QuantumCircuit(n_q)
    if u.shape[0]==4 and np.linalg.norm(u-target_unitary) < 1e-8:
        qc.swap(0,1)
    else:
        qc.unitary(u, list(range(n_q)))
    qc_noisy = insert_noise(qc, noise_model, transpile=True)
    return Choi(qc_noisy)
for i in range(len(basis_ops)):
    op = basis_ops[i]
    if op.is_unitary:
        endo_ops.append(noisy_unitary(op.u).data)
    else:
        ch = noisy_unitary(op.u1)
        ch = ch.compose(channel_project(1, 0))
        if op.u2 is not None:
            ch = ch.compose(noisy_unitary(op.u2))
        endo_ops.append(Choi(ch).data)
endo_ops = gen_two_qubit_basis(endo_ops)
endo_ops.append(noisy_unitary(target_unitary, n_q=2).data)

coeffs,_ = qpd(target_choi, endo_ops, 2)
print("ENDO REF:", np.sum(np.abs(coeffs)))
