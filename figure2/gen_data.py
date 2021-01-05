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
from diamond_norm import *

noise_model = NoiseModel.from_dict(json_from_file("2020_04_08.json"))
noise_model.add_quantum_error(noise_model._local_quantum_errors['cx']['2,3'], 'cx', [0,2])
noise_model.add_quantum_error(noise_model._local_quantum_errors['cx']['3,2'], 'cx', [2,0])


# target ops
qc = QuantumCircuit(1)
qc.ry(2.*np.arccos(np.sqrt(0.56789)), 0)
qc_noisy = insert_noise(qc, noise_model, transpile=True)
ry_unitary = Operator(qc).data
ry_choi = Choi(qc).data
ry_noisy = Choi(qc_noisy).data


qc = QuantumCircuit(2)
qc.cx(0,1)
qc_noisy = insert_noise(qc, noise_model, transpile=True)
cnot_unitary = Operator(qc).data
cnot_choi = Choi(qc).data
cnot_noisy = Choi(qc_noisy).data

qc = QuantumCircuit(2)
qc.swap(0,1)
qc_noisy = insert_noise(qc, noise_model, transpile=True)
swap_unitary = Operator(qc).data
swap_choi = Choi(qc).data
swap_noisy = Choi(qc_noisy).data


# build endo basis
def noisy_unitary(u, n_q=1):
    qc = QuantumCircuit(n_q)
    qc.unitary(u, list(range(n_q)))
    qc_noisy = insert_noise(qc, noise_model, transpile=True)
    return Choi(qc_noisy)
basis_ops = get_basis_ops(endo_unitaries=True, endo_projections=True)

endo_ops_1 = list()
for i in range(len(basis_ops)):
    op = basis_ops[i]
    if op.is_unitary:
        endo_ops_1.append(noisy_unitary(op.u).data)
    else:
        ch = noisy_unitary(op.u1)
        ch = ch.compose(channel_project(1, 0))
        if op.u2 is not None:
            ch = ch.compose(noisy_unitary(op.u2))
        endo_ops_1.append(Choi(ch).data)
endo_ops_2 = gen_two_qubit_basis(endo_ops_1)


# generate data
def gen_data(gate_choi, gate_noisy, basis, n_qubits, cbudget_list, filename):
    xvals = list()
    yvals = list()
    for cbudget in cbudget_list:
        coeffs,dn = bqpd_c(gate_choi, basis, n_qubits, c_fac_budget=cbudget, cp_constraint=False, use_mosek=True)
        xvals.append(cbudget)
        yvals.append(dn)
    ref_dn = dnorm(gate_noisy - gate_choi)
    np.savez(filename, np.array(xvals), np.array(yvals), ref_dn)


clist1 = np.linspace(0.99, 1.015, 100)
clist2 = np.linspace(0.9, 1.22, 100)
clist3 = np.linspace(0.8, 2.3, 100)
print("Generating Ry data...")
gen_data(ry_choi, ry_noisy, endo_ops_1+[ry_noisy], 1, clist1, "data/data_ry.npz")
print("Generating CNOT data...")
gen_data(cnot_choi, cnot_noisy, endo_ops_2+[cnot_noisy], 2, clist2, "data/data_cnot.npz")
print("Generating SWAP data...")
gen_data(swap_choi, swap_noisy, endo_ops_2+[swap_noisy], 2, clist3, "data/data_swap.npz")
