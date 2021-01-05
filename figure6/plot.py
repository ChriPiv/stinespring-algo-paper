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
import matplotlib.pyplot as plt
import matplotlib

data_cnot = [0.031102744141666073,0.004038771580653574,0.00035862078629081184,2.515689649341933e-05,1.4567221275703946e-06,6.02040927971054e-08]
data_swap = [0.09310967570120603,0.009636916535298558,0.0008533404474038529,6.12248559322307e-05,4.081308986629326e-06,2.372185229109708e-07,9.421773413540917e-09]
data_ry = [0.002705542457872339,4.806365125741009e-05,5.207868980331364e-07,7.666074849647217e-10]

plt.rc('font', family='serif', serif='Palatino', size=9)
plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=9)
plt.rc('ytick', labelsize=9)
plt.rc('axes', labelsize=9)
COLORS = [
        '#02A5CF',
        '#DE2019',
        '#FFBF00',
        '#29BF12',
        '#574AE2'
    ]

fig, ax = plt.subplots()
fig.subplots_adjust(left=.18, bottom=.16, right=.98, top=.9)
ax.plot([0., 10.], [1e-7, 1e-7], color='black', linewidth=1)
ax.plot(np.arange(len(data_ry))+1., data_ry, color=COLORS[0], ls='--', marker='o', markersize=4, label='\emph{Ry}')
ax.plot(np.arange(len(data_cnot))+1., data_cnot, color=COLORS[2], ls='--', marker='o', markersize=4, label='\emph{CNOT}')
ax.plot(np.arange(len(data_swap))+1., data_swap, color=COLORS[3], ls='--', marker='o', markersize=4, label='\emph{SWAP}')
#ax.plot([0., 4.], [refval, refval], color=COLORS[1], ls='--')
ax.set_title('Convergence of approximation error in Stinespring algorithm')
ax.set_xlim([0, 8])
ax.set_xlabel('Number of steps')
ax.set_ylabel('Error $\Delta$ (diamond norm)')
ax.set_yscale('log')
#ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(xax_tick))
#ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(yax_tick))
#ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
ax.legend(loc='best')
ax.text(1., 2e-7, "Threshold", color='black')
fig.set_size_inches(5., 5./1.618)
plt.savefig("plots/stinespring_convergence.pdf")

