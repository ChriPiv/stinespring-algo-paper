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
from scipy.optimize import minimize

data_ry = np.load('data/data_ry.npz')
data_cnot = np.load('data/data_cnot.npz')


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

def plot(xval, yval, refval, title, filename, xax_tick, yax_tick):
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=.18, bottom=.16, right=.98, top=.9)
    ax.plot([0., 4.], [0., 0.], color='black', linewidth=1)
    ax.plot(xval, yval, color=COLORS[0])
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

# plot
plot(data_ry['arr_0'], data_ry['arr_1'], data_ry['arr_2'],
     title="\emph{Ry} Gate", filename="plots/tradeoff_ry.pdf",
     xax_tick=0.01, yax_tick=0.005)
plot(data_cnot['arr_0'], data_cnot['arr_1'], data_cnot['arr_2'],
     title="\emph{CNOT} Gate", filename="plots/tradeoff_cnot.pdf",
     xax_tick=0.1, yax_tick=0.05)


# run optimization
print(len(data_ry['arr_0']))
print(len(data_ry['arr_1']))
def Ry_eps(C):
    return np.interp(C, data_ry['arr_0'], data_ry['arr_1'], left=2., right=data_ry['arr_1'][-1])
def CNOT_eps(C):
    return np.interp(C, data_cnot['arr_0'], data_cnot['arr_1'], left=2., right=data_cnot['arr_1'][-1])

X = np.linspace(0.8, 1.3, 1000)
plot(X, CNOT_eps(X), data_cnot['arr_2'],
     title="\emph{CNOT} Gate", filename="plots/temp.pdf",
     xax_tick=0.1, yax_tick=0.05)


list_ctot = list()
list_cry = list()
list_ccnot = list()
for Ctot in np.linspace(1., 1.19139, 50):
    print(Ctot)
    def loss(Cry):
        Ccnot = Ctot / Cry
        return Ry_eps(Cry) + CNOT_eps(Ccnot)
    x0 = (Ctot + 1.)*0.5
    res = minimize(loss, x0, method='BFGS')

    list_ctot.append(Ctot)
    list_cry.append(res.x[0])
    list_ccnot.append(Ctot / res.x[0])

# plot optimization result
fig, ax = plt.subplots()
fig.subplots_adjust(left=.15, bottom=.15, right=.99, top=0.9)
ax.plot([0., 4.], [1., 1.], color='black', linewidth=1)
ax.plot([0., 4.], [1.0106, 1.0106], color=COLORS[1], ls='--')
ax.plot([0., 4.], [1.1789, 1.1789], color=COLORS[1], ls='--')
ax.text(1.05, 1.02, "perfect Ry", color=COLORS[1])
ax.text(1.05, 1.189, "perfect CNOT", color=COLORS[1])
ax.plot(list_ctot, list_ctot, color='black', linewidth=1)
ax.plot(list_ctot, list_ccnot, color=COLORS[2], label="CNOT")
ax.plot(list_ctot, list_cry, color=COLORS[0], label="Ry")

ax.set_title("Optimal distribution of $\gamma$-factor budget")
ax.set_xlim([1., 1.19])
ax.set_xlabel('Total $\gamma$-factor budget $\gamma_{tot}$')
ax.set_ylabel('Local $\gamma$-factor budget')
#ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(xax_tick))
#ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(yax_tick))
#ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
fig.legend(loc='center right')
fig.set_size_inches(5., 5./1.618)
plt.savefig("plots/budget_distribution.pdf")

