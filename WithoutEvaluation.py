from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy import linalg
import time
import random
import math
import cmath

# def RandomSimplex(d):
#     r = sorted([random.random() for i in range(d-1)])
#     r1 = r + [1]
#     r2 = [0] + r
#     return [a - b for a, b in zip(r1, r2)]

# def RandomKet(n):
#     p = [math.sqrt(val) for val in RandomSimplex(n)]
#     ph = [cmath.exp(1j * random.uniform(0, 2 * math.pi)) for _ in range(n - 1)]
#     ph = [1] + ph
#     return [a * b for a, b in zip(p, ph)]

start = time.time()

# Omega1 = np.array(qload('without_no_Omega1'))
# Delta1 = np.array(qload('without_no_Delta1'))
# Omega2 = np.array(qload('without_no_Omega2'))
# Delta2 = np.array(qload('without_no_Delta2'))
# J = np.array(qload('without_no_J'))
# t = np.array(qload('without_no_time'))

Omega1 = np.array(qload('without_gauss_Omega1'))
Delta1 = np.array(qload('without_gauss_Delta1'))
Omega2 = np.array(qload('without_gauss_Omega2'))
Delta2 = np.array(qload('without_gauss_Delta2'))
J = np.array(qload('without_gauss_J'))
t = np.array(qload('without_gauss_time'))

# Omega1 = np.array(qload('without_dropout_Omega1'))
# Delta1 = np.array(qload('without_dropout_Delta1'))
# Omega2 = np.array(qload('without_dropout_Omega2'))
# Delta2 = np.array(qload('without_dropout_Delta2'))
# J = np.array(qload('without_dropout_J'))
# t = np.array(qload('without_dropout_time'))

# strength = 1
# Omega1 = Omega1 * 2*strength
# Delta1 = (2 * Delta1 - 1) * 1.6 *strength
# Omega2 = Omega2 * 2*strength
# Delta2 = (2 * Delta2 - 1) * 1 *strength
# J = (2 * J - 1) * 1.7*strength


strength = 2*np.pi
Omega1 = Omega1 * strength
Delta1 = (2 * Delta1 - 1) * strength
Omega2 = Omega2 * strength
Delta2 = (2 * Delta2 - 1) * strength
J = (2 * J - 1) * strength


_max_episode_timesteps = 100
tf = t[-1]
omega = 0.5*np.pi
target = (1j * np.pi * tensor(sigmay(), sigmay()) / 4).expm()
# t = np.linspace(0,tf,_max_episode_timesteps+1)

# evotime=[0,tf/_max_episode_timesteps]
# phi0 = tensor(basis(2,0),basis(2,1))
# phit = target*phi0

nx = 21
F = np.zeros([nx, nx])
lam_J = 0

# %%
for i in range(nx):
    for j in range(nx):
        lam_Delta = -0.04+0.004*i
        lam_Omega = -0.04+0.004*j
        gate = tensor(qeye(2), qeye(2))
        # phi0 = Qobj(np.array(RandomKet(4)),[[2,2],[1,1]])
        # phit = target*phi0
        for timestep in range(_max_episode_timesteps):
            H1 = 0.5 * (Omega1[timestep]*(1+lam_Omega)*np.cos(omega*t[timestep])*sigmax() + Omega1[timestep]*(
                1+lam_Omega)*np.sin(omega*t[timestep])*sigmay() + Delta1[timestep]*(1+lam_Delta)*sigmaz())
            H2 = 0.5 * (Omega2[timestep]*(1+lam_Omega)*np.cos(omega*t[timestep])*sigmax() + Omega2[timestep]*(
                1+lam_Omega)*np.sin(omega*t[timestep])*sigmay() + Delta2[timestep]*(1+lam_Delta)*sigmaz())
            H3 = 0.5 * J[timestep]*(1+lam_J)*tensor(sigmaz(), sigmaz())
            H = tensor(H1, qeye(2)) + tensor(qeye(2), H2) + H3
            result = (-1j * H * tf / _max_episode_timesteps).expm()
            # print(H1)
            gate = result*gate
            # H = Qobj(np.kron(H1,np.eye(2)) + np.kron(np.eye(2),H2) + H3, [[2,2],[2,2]])
            # result = mesolve(H, phi0, evotime, [], [])
            # phi0 = result.states[1]

        Fidelity = abs(np.trace(target.dag()*gate)/4)**2
        # Fidelity=fidelity(phi0,phit)**2
        print('sigmaz deviation=', lam_Delta, 'sigmax deviation=',
              lam_Omega, 'Fidelity=', Fidelity)
        F[i, j] = Fidelity


# %%
fig, ax = plt.subplots()
norm = matplotlib.colors.Normalize(vmin=0.1, vmax=2.5)
im = ax.imshow(-np.log10(1-F),
               interpolation='bilinear',
               origin='lower',
               cmap='coolwarm',
               extent=(-0.2, 0.2, -0.2, 0.2),
               norm=norm)
ax.set_title('$-log_{10}(1-F)$', fontsize=12, y=1.02)
ax.set_xlabel('$\delta \Omega$', fontsize=16)
ax.set_ylabel('$\delta \Delta$', fontsize=16)
plt.xticks(np.arange(-0.2, 0.201, step=0.1), fontsize=14)  # x轴坐标
plt.yticks(np.arange(-0.2, 0.201, step=0.1), fontsize=14)  # y轴坐标
f_level=np.arange(0.1, 2.6, 0.4)
CB = fig.colorbar(im, ticks=f_level)

# fig, ax = plt.subplots()
nd = 101
delt = np.linspace(-0.2, 0.2, nd)
levels = np.array([0.5, 1, 2])
CS = ax.contour(-np.log10(1-F),
                levels=levels,
                origin='lower',
                extend='both',
                linewidths=0.8,
                colors='k',
                # cmap='coolwarm',
                extent=(delt[0], delt[-1], delt[0], delt[-1]))
ax.set_aspect(9/10)
ax.clabel(CS, inline=True, fontsize=12)
# %%
"""
##########
fig.savefig('without_no.pdf',dpi=800,bbox_inches='tight',format='pdf')
qsave(F,'without_no_fidelity')
F = qload('without_no_fidelity7')
np.savetxt("MingleF.txt",F)
#########


###########
fig,ax=plt.subplots(1,1)
ax.plot(np.linspace(1, 100000,100000),unpickled_array)
ax.legend(("$\Delta$", "$\Omega$", "$J$"))
###########


import pylab as pl
import numpy as np

a = np.array([[0.1,2.6]])
pl.figure(figsize=(0.5,10))
img = pl.imshow(a, cmap="coolwarm")
pl.gca().set_visible(False)
cax = pl.axes([0.1, 0.2, 0.8, 0.6])
cbar=pl.colorbar(orientation="vertical", cax=cax)
cbar.ax.tick_params(labelsize=14)
pl.savefig("colorbar.pdf",bbox_inches='tight')


import numpy as np
import matplotlib.pylab as pl

x = np.linspace(0, 2*np.pi, 64)
y = np.cos(x) 

pl.figure()
pl.plot(x,y)

n = 20
colors = pl.cm.jet(np.linspace(0,1,n))

for i in range(n):
    pl.plot(x, i*y, color=colors[i])
    
    
norm = matplotlib.colors.Normalize(vmin=0,vmax=0.8)  # 设置colorbar显示的最大最小值
CB = fig.colorbar(im, ax=ax, shrink=0.8, ticks=f_level)  # 添加cbar
plt.xticks(np.arange(0, 2.2, step=0.2),list('abcdefghigk'),rotation=45)  #自定义刻度标签值，刻度显示为您想要的一切（日期，星期等等）
CS.collections[n].set_linewidth(m)  # 选取第 n 个等高线设置宽度 m
ax.clabel(CS, levels[1::2], inline=True, fmt='%1.1f', fontsize=14)  # label every second level

plt.show()
"""

end = time.time()
print('Running time: %s seconds' % (end-start))
