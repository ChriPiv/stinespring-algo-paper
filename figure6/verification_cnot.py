# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import numpy as np
import scipy
import glob
from qiskit import *
from qiskit.quantum_info import *
from qiskit.providers.aer.noise import *
from qiskit.providers.aer.utils import insert_noise

sys.path.append("..")
from json_tools import *

"""
The goal of this script is to verify that the circuits generated by the
Stinespring algorithm indeed do produce a valid QPD.
For that we consider a circuit with some random (non-noisy) single-qubit
unitaries and a noisy CNOT that we correct with the quasiprobability method.
"""

noise_model = NoiseModel.from_dict(json_from_file("2020_04_08.json"))
noise_model.add_quantum_error(noise_model._local_quantum_errors['cx']['2,3'], 'cx', [0,2])
noise_model.add_quantum_error(noise_model._local_quantum_errors['cx']['3,2'], 'cx', [2,0])

# load quasiprobability coefficients generated by Stinespring algorithm
coeffs = np.load("data/final_cnot.npz")["arr_2"]

# load circuits generated by Strinespring algorithm
def read_file(path):
    with open(path, 'r') as f:
        return f.read()
num_files = len(glob.glob("data/*.qasm"))
qasm_circuits = [read_file("data/final_cnot_sim_circ"+str(f)+".qasm") for f in range(num_files)]
circuits = [QuantumCircuit.from_qasm_str(s) for s in qasm_circuits]

# compute ideal output (without noise)
backend = Aer.get_backend('qasm_simulator')
def get_probab(qc):
    qc.measure([1], [0])
    shots = 10000 # make this higher to increase accuracy!
    job = execute(qc, backend, shots=shots)
    counts = job.result().get_counts(qc)
    if '1' not in counts: counts['1'] = 0
    return counts['1'] / shots

U = random_unitary(2, seed=1234) # 1237
qc = QuantumCircuit(2, 1)
qc.unitary(U, [0])
qc.cx(0, 1)
probab_ref = get_probab(qc)
print(probab_ref)

# run circuits of different quasiprobability branches
probabs = list()
for qc in circuits:
    qc_noisy = insert_noise(qc, noise_model)
    if qc.num_qubits == 2:
        qctot = QuantumCircuit(2, 1)
        qctot.unitary(U, [0])
    elif qc.num_qubits == 3:
        qctot = QuantumCircuit(3, 1)
        qctot.unitary(U, [0])
    else: raise
    qctot += qc_noisy
    probabs.append( get_probab(qctot) )
    #print(qctot)
probabs = np.array(probabs)

# check
print(np.sum(probabs * coeffs))
