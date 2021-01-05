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

data_ry = np.load('data/data_ry.npz')
data_cnot = np.load('data/data_cnot.npz')
data_swap = np.load('data/data_swap.npz')


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

def plot(xval, yval, refval, title, filename, xax_tick, yax_tick, color):
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=.18, bottom=.16, right=.98, top=.9)
    ax.plot([0., 4.], [0., 0.], color='black', linewidth=1)
    ax.plot(xval, yval, color=COLORS[color])
    ax.plot([0., 4.], [refval, refval], color=COLORS[1], ls='--')
    ax.set_title(title)
    ax.set_xlim([np.min(xval), np.max(xval)])
    ax.set_xlabel('$\gamma$ budget')
    ax.set_ylabel('Error (diamond norm)')
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(xax_tick))
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(yax_tick))
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax.text(0.7*np.max(xval)+0.3*np.min(xval), refval*1.1, "Reference", color=COLORS[1])
    fig.set_size_inches(2.5, 2.5)
    plt.savefig(filename)

plot(data_ry['arr_0'], data_ry['arr_1'], data_ry['arr_2'],
     title="\emph{Ry} Gate", filename="plots/tradeoff_ry.pdf",
     xax_tick=0.01, yax_tick=0.005, color=0)
plot(data_cnot['arr_0'], data_cnot['arr_1'], data_cnot['arr_2'],
     title="\emph{CNOT} Gate", filename="plots/tradeoff_cnot.pdf",
     xax_tick=0.1, yax_tick=0.05, color=2)
plot(data_swap['arr_0'], data_swap['arr_1'], data_swap['arr_2'],
     title="\emph{SWAP} Gate", filename="plots/tradeoff_swap.pdf",
     xax_tick=0.5, yax_tick=0.1, color=3)
