This repository contains all the code to reproduce the data and figures for the paper _Quasiprobability decompositions with reduced sampling overhead_ by Christophe Piveteau, David Sutter and Stefan Woerner ([https://arxiv.org/abs/2101.09290](https://arxiv.org/abs/2101.09290)).


Setup
=====
To reproduce the experiments, I highly suggest you create a new Python3 environment (e.g. with the `venv` tool). Python3.7 was used for the paper, though other versions of Python3 will probably also work fine. In your new environment install following packages (the exact version numbers are important!):
* numpy 1.18.3
* autograd 1.3
* cvxpy 1.0.31
* qiskit-terra 0.15.0
* qiskit-aer 0.6.0
* qiskit-ignis 0.4.0
* qiskit-ibmq-provider 0.7.1
* qiskit-aqua 0.7.0
* mosek 9.2.8

If you wish to regenerate the plots, you will additionaly require matplotlib and latex to be installed.

The used version of qiskit-terra contains a bug that will prevent some code from running. Fix it by including following three lines before line `162` in `qiskit/quantum_info/operators/channel/kraus.py` (this file is part of qiskit-terra, so you have to change it inside your environment):

```python
if len(kraus[0][0].shape) == 3:
    for i in range(len(kraus[0])):
            kraus[0][i] = kraus[0][i][:,:,0] + 1j* kraus[0][i][:,:,1]
```

Overview of code
================
The parent directory contains python files with routines that are commonly reused throughout the paper. The results in table 2 are generated as part of the code of figure 6. Each subdirectory contains a file `gen_data.py` which performs the simulations and a file `plot.py` which reproduces the plots from the paper.
