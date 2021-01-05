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

qc = QuantumCircuit(1)
qc.ry(2.*np.arccos(np.sqrt(0.56789)), 0)
target_unitary = Operator(qc).data
target_choi = Choi(qc).data
n_qubits = 1

noise_model = NoiseModel.from_dict(json_from_file("2020_04_08.json"))

def noise_oracle(U, num_anc):
    exp = channel_expand(n_qubits, num_anc)
    tr = channel_trace(n_qubits, num_anc)
    qc = QuantumCircuit(num_anc+n_qubits)

    qc.unitary(U, list(range(num_anc+n_qubits)))
    qc_noisy = insert_noise(qc, noise_model, transpile=True)
    qc_noisy = SuperOp(qc_noisy)

    return Choi( exp.compose(qc_noisy.compose(tr)) ).data

# Stinespring algorithm
fixed_unitaries, fixed_choi, coeffs = stinespring_algorithm(target_unitary, n_qubits, noise_oracle, disp=True, cfac_tol=1.0, bm_ops=2, cfac_budget=1.007)
print("STINESPRING:", np.sum(np.abs(coeffs)))
