import matplotlib.pyplot as plt
import numpy as np
import utils as utils
from matplotlib.ticker import FormatStrFormatter

def freiling(env, inf_trader, broker):
    t = env.timesteps
    z = inf_trader.z
    f3 = z[9] / inf_trader.k
    f2 = z[2] / 2 / inf_trader.k
    f1 = z[1] / 2 / inf_trader.k
    f0 = z[0] / 2 / inf_trader.k
    
    C = np.array([[0, 0], [0, 1]])
    D = np.array([[-1,0], [0, -1]])
    L = np.zeros((broker.N, 4, 4))
    BigL = np.zeros((broker.N, 4, 4))
    U = np.zeros((broker.N, 2 , 2))
    V = np.zeros((broker.N, 2 , 2))
    B = np.zeros((broker.N, 2 , 2))
    eig_vals = np.zeros((broker.N, 4))
    max_eig_vals = np.zeros((broker.N,))
    determinants = np.zeros((broker.N,))

    for t in range(broker.N):
        U[t, 0, 0] = (1 - f2[t])**2
        U[t, 0, 1] = f2[t] * (1 - f2[t])
        U[t, 1, 0] = U[t, 0, 1]
        U[t, 1, 1] = f2[t]**2

        U[t, :, :] /= broker.k - inf_trader.k * f2[t]**2

        V[t, 0, 0] = env.b *(1-f2[t])/ 2
        V[t, 0, 1] = -f3[t]*(broker.k - inf_trader.k *f2[t])
        V[t, 1, 0] = env.b * f2[t] / 2
        V[t, 1, 1] = f3[t] * broker.k

        V[t, :, :] /= broker.k - inf_trader.k * f2[t]**2

        B[t, 0, 0] = .25 * env.b**2 - (broker.k - inf_trader.k * f2[t]**2) * (broker.rho0 + broker.rho1 * broker.V_B[t] )
        B[t, 0, 1] = 0.5 * env.b * f2[t] * f3[t] * inf_trader.k
        B[t, 1, 0] = B[t, 0, 1]
        B[t, 1, 1] = broker.k * inf_trader.k * f3[t]**2

        B[t, :, :] /= broker.k - inf_trader.k * f2[t]**2

        L[t, :, :] = np.block([[C@V[t,:,:] + B[t,:,:], C@U[t,:,:]],
                               [np.zeros((2,2)), -U[t,:,:]]])

        BigL[t,:,:] = L[t, :, :] + (L[t,:,:]).T

        eig_vals[t, :] = np.linalg.eigvals(BigL[t,:,:])
        
    eig_vals_sorted = np.sort(eig_vals, axis=-1)
    eig_vals_sorted[:, -1] = 0
    cond_0 = not (1 + inf_trader.k * f3 <= 0).any()

    fig, ax = plt.subplots(1, 3, figsize=(14,4))
    t = env.timesteps

    font0 = 25
    font1 = 25
    font2 = 23

    for j in range(3):
        ax[j].plot(env.timesteps, eig_vals_sorted[:, j], linewidth = 2)
        ax[j].set_xlim(0,1)
        if j == 1:
            ax[j].axhline(0, linestyle='-.', color = 'black', linewidth=1)
        #if j == 2:
            #ax[j].yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
        ax[j].set_xlabel(r'$t$', fontsize = font1)
        ax[j].tick_params(axis='both', which='major', labelsize=font2)

    fig.suptitle(rf'$\mathrm{{Eigenvalues}}$ $\mathrm{{of}}$ $L(t) + L(t)^\top$ ($1 + \mathfrak{{b}}\, f_3 > 0$ $\mathrm{{is}}$ $\mathrm{{{cond_0}}}$)', fontsize=font0) 
    plt.tight_layout()
    plt.savefig(f'figures/freiling condition.pdf',format='pdf',bbox_inches='tight')
    plt.show()