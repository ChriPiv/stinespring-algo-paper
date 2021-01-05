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

sys.path.append("..")
from json_tools import *
from channels import *
from variational_approximation import error_mean, get_approx_circuit, get_varform_circuit
from diamond_norm import *
import autograd.numpy as np

n_qubits = 3
full_connectivity = False


U = random_unitary(2**n_qubits, seed=1234).data
noise_model = NoiseModel.from_dict(json_from_file("2020_04_08.json"))
noise_model.add_quantum_error(noise_model._local_quantum_errors['cx']['2,3'], 'cx', [0,2])
noise_model.add_quantum_error(noise_model._local_quantum_errors['cx']['3,2'], 'cx', [2,0])

def dilation_channel(data, is_unitary=True, ideal=False):
    exp = channel_expand(n_qubits-1,1)
    if is_unitary:
        qc = QuantumCircuit(n_qubits)
        qc.unitary(data, list(range(n_qubits)))
    else:
        qc = data
    if not ideal:
        if not full_connectivity:
            qc = qiskit.compiler.transpile(qc, basis_gates=noise_model.basis_gates,
                                               coupling_map=[[0,1],[1,2]])
        qc = insert_noise(qc, noise_model, transpile=True)
    qc = SuperOp(qc)
    tr = channel_trace(n_qubits-1,1)
    channel = exp.compose(qc.compose(tr))
    return Choi(channel).data

ch_ideal = dilation_channel(U, ideal=True)
ch_ref = dilation_channel(U)
assert Choi(ch_ideal).is_tp()
assert Choi(ch_ideal).is_cp()
assert Choi(ch_ref).is_tp()
assert Choi(ch_ref).is_cp()
print("Ref:", dnorm(ch_ideal - ch_ref))

if full_connectivity:
    depth_list = [1,2,3,4,5,6,7,8,9,10,15]
else:
    depth_list = [1,3,4,5,6,7,8,9,10,15,20,30,40]
for depth in depth_list:
    U_approx,params = get_approx_circuit(U, n_qubits, depth, full_connectivity)
    qc = get_varform_circuit(params, n_qubits, depth, full_connectivity)
    ch = dilation_channel(qc, is_unitary=False)
    print(depth, error_mean(U, U_approx, 2), dnorm(ch - ch_ideal))

