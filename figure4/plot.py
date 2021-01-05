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

data_depth = [1,3,4,5,6,7,8,9,10,15,20,30,40]
data_Uerr = [0.628245746335422,0.22290757952981,0.149242822500311,0.08143287647172,0.029886812723847,6.45E-05,2.46E-05,3.78E-05,4.34E-05,2.70E-05,1.21E-05,1.73E-05,1.22E-05]
data_dn = [1.9230726255549,0.862176266043871,0.557000208494517,0.491780449406918,0.385602812975596,0.427080269538043,0.461409922345587,0.522259333008194,0.538863530077179,0.769370902842745,0.972278468049599,1.24556563035902,1.41580566008791]
data_ref = 1.777408917389572 


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
fig.subplots_adjust(left=.21, bottom=.16, right=.97, top=.9)
ax.plot(data_depth, data_dn, color=COLORS[4])
ax.plot([0., 100.], [data_ref, data_ref], color=COLORS[1], ls='--')
ax.set_title("(a)")
ax.set_xlim([0, np.max(data_depth)])
ax.set_ylim([0, 2.])
ax.set_xlabel('Number of layers')
ax.set_ylabel('Error (diamond norm)')
#ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(xax_tick))
#ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(yax_tick))
#ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
ax.text(15., 1.8, "Reference", color=COLORS[1])
fig.set_size_inches(2.5, 2.5)
plt.savefig('plots/vua_dn.pdf')


fig, ax = plt.subplots()
fig.subplots_adjust(left=.21, bottom=.16, right=.97, top=.9)
ax.plot(data_depth, data_Uerr, color=COLORS[0])
ax.set_title("(b)")
ax.set_xlim([0, 15.])
ax.set_xlabel('Number of layers')
ax.set_ylabel('L2 Unitary Approximation Error')
#ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(xax_tick))
#ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(yax_tick))
#ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
fig.set_size_inches(2.5, 2.5)
plt.savefig('plots/vua_Uerr.pdf')
