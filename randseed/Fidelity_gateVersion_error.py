from qutip import *
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import linalg
import time

def single(gate0, Omega, Delta, omega, delta1, delta2, nt, timestep):
    # omega = 2 * np.pi / tf
    # Delta = omega * np.cos(beta) ** 2
    # Omega = np.sqrt(Delta * omega - Delta ** 2)
    tf = 2 * np.pi / omega
    t = np.linspace(0, tf, nt)  # Define time vector
    u = Delta * (1 + delta2) * sigmaz() / 2 + \
        Omega * (1 + delta1) * np.cos(omega * t[timestep]) * sigmax() / 2 + \
        Omega * (1 + delta1) * np.sin(omega * t[timestep]) * sigmay() / 2
    gate = (-1j * u * tf / (nt-1)).expm()
    output = gate*gate0
    return output


start = time.time()


nd = 101
delt1 = np.linspace(-0.2-0.004*0, 0.2-0.004*0, nd)
delt2 = np.linspace(-0.2-0.004*0, 0.2-0.004*0, nd)
f = np.zeros((nd, nd))
Omega = 2
nt = 40


# delta1 is the error for Omega, delta2 is the error for Delta

jj = 0
for delta1 in delt1:
    ii = 0
    for delta2 in delt2:
        # gate V1
        output = qeye(2)
        beta = 0.521341
        Delta = Omega * np.cos(beta) / np.sin(beta)
        omega = Omega / (np.sin(beta) * np.cos(beta))
        for timestep in range(1,nt):
            output = single(output, Omega, Delta, omega, delta1, delta2, nt, timestep)

        beta = 0.287802
        Delta = Omega * np.cos(beta) / np.sin(beta)
        omega = Omega / (np.sin(beta) * np.cos(beta))
        for timestep in range(1,nt):
            output = single(output, Omega, Delta, omega, delta1, delta2, nt, timestep)

        beta = 0.912661
        Delta = Omega * np.cos(beta) / np.sin(beta)
        omega = Omega / (np.sin(beta) * np.cos(beta))
        for timestep in range(1,nt):
            output = single(output, Omega, Delta, omega, delta1, delta2, nt, timestep)

        beta = 0.128691
        Delta = Omega * np.cos(beta) / np.sin(beta)
        omega = Omega / (np.sin(beta) * np.cos(beta))
        for timestep in range(1,nt):
            output = single(output, Omega, Delta, omega, delta1, delta2, nt, timestep)
        # print(output)
        gate1 = output

        # gate V2
        output = qeye(2)

        beta = 0.478711
        Delta = Omega * np.cos(beta) / np.sin(beta)
        omega = Omega / (np.sin(beta) * np.cos(beta))
        for timestep in range(1,nt):
            output = single(output, Omega, Delta, omega, delta1, delta2, nt, timestep)

        beta = 0.231424
        Delta = Omega * np.cos(beta) / np.sin(beta)
        omega = Omega / (np.sin(beta) * np.cos(beta))
        for timestep in range(1,nt):
            output = single(output, Omega, Delta, omega, delta1, delta2, nt, timestep)

        beta = 0.686241
        Delta = Omega * np.cos(beta) / np.sin(beta)
        omega = Omega / (np.sin(beta) * np.cos(beta))
        for timestep in range(1,nt):
            output = single(output, Omega, Delta, omega, delta1, delta2, nt, timestep)

        beta = 0.299632
        Delta = Omega * np.cos(beta) / np.sin(beta)
        omega = Omega / (np.sin(beta) * np.cos(beta))
        for timestep in range(1,nt):
            output = single(output, Omega, Delta, omega, delta1, delta2, nt, timestep)

        beta = 0.90449
        Delta = Omega * np.cos(beta) / np.sin(beta)
        omega = Omega / (np.sin(beta) * np.cos(beta))
        for timestep in range(1,nt):
            output = single(output, Omega, Delta, omega, delta1, delta2, nt, timestep)

        beta = 0.314679
        Delta = Omega * np.cos(beta) / np.sin(beta)
        omega = Omega / (np.sin(beta) * np.cos(beta))
        for timestep in range(1,nt):
            output = single(output, Omega, Delta, omega, delta1, delta2, nt, timestep)

        beta = 0.456121
        Delta = Omega * np.cos(beta) / np.sin(beta)
        omega = Omega / (np.sin(beta) * np.cos(beta))
        for timestep in range(1,nt):
            output = single(output, Omega, Delta, omega, delta1, delta2, nt, timestep)

        # print(output)
        gate2 = output

        # gate Mikio

        output = Qobj(qeye(4), [[2, 2], [2, 2]])
        # omega = 1
        # J = 0.3187
        omega = 2.59561 * Omega  # Omega=sqrt((omega/2)^2-J^2), J=0.3187*omega
        J = 0.3187 * omega
        D = np.sqrt(Omega ** 2 + J ** 2)
        T = 1.21  # 2 * np.pi / omega # Eq(21) is the requirement of geometric gate
        # D = omega / 2
        # Omega1 = np.sqrt(D ** 2 - J ** 2)
        for timestep in range(1,nt):
            t = np.linspace(0, T, nt)  # Define time vector
            Hprim = J * tensor(sigmaz(), sigmaz()) / 2 + D * (1 + delta2) * tensor(qeye(2), sigmaz()) / 2 + \
                    Omega * (1 + delta1) * np.cos(omega * t[timestep]) * tensor(qeye(2), sigmax()) / 2 + \
                    Omega * (1 + delta1) * np.sin(omega * t[timestep]) * tensor(qeye(2), sigmay()) / 2
            gate = (-1j * Hprim * T / (nt - 1)).expm()
            output = gate * output
        # print(output)
        gatemikio = output

        # gate V3
        output = qeye(2)

        beta = 0.810056
        Delta = Omega * np.cos(beta) / np.sin(beta)
        omega = Omega / (np.sin(beta) * np.cos(beta))
        for timestep in range(1,nt):
            output = single(output, Omega, Delta, omega, delta1, delta2, nt, timestep)

        beta = 0.294323
        Delta = Omega * np.cos(beta) / np.sin(beta)
        omega = Omega / (np.sin(beta) * np.cos(beta))
        for timestep in range(1,nt):
            output = single(output, Omega, Delta, omega, delta1, delta2, nt, timestep)

        beta = 0.810074
        Delta = Omega * np.cos(beta) / np.sin(beta)
        omega = Omega / (np.sin(beta) * np.cos(beta))
        for timestep in range(1,nt):
            output = single(output, Omega, Delta, omega, delta1, delta2, nt, timestep)

        beta = 0.241139
        Delta = Omega * np.cos(beta) / np.sin(beta)
        omega = Omega / (np.sin(beta) * np.cos(beta))
        for timestep in range(1,nt):
            output = single(output, Omega, Delta, omega, delta1, delta2, nt, timestep)

        beta = 0.289496
        Delta = Omega * np.cos(beta) / np.sin(beta)
        omega = Omega / (np.sin(beta) * np.cos(beta))
        for timestep in range(1,nt):
            output = single(output, Omega, Delta, omega, delta1, delta2, nt, timestep)

        beta = 0.558668
        Delta = Omega * np.cos(beta) / np.sin(beta)
        omega = Omega / (np.sin(beta) * np.cos(beta))
        for timestep in range(1,nt):
            output = single(output, Omega, Delta, omega, delta1, delta2, nt, timestep)

        beta = 0.238617
        Delta = Omega * np.cos(beta) / np.sin(beta)
        omega = Omega / (np.sin(beta) * np.cos(beta))
        for timestep in range(1,nt):
            output = single(output, Omega, Delta, omega, delta1, delta2, nt, timestep)
        # print(output)
        gate3 = output

        # gate V4
        output = qeye(2)
        beta = 0.619671
        Delta = Omega * np.cos(beta) / np.sin(beta)
        omega = Omega / (np.sin(beta) * np.cos(beta))
        for timestep in range(1,nt):
            output = single(output, Omega, Delta, omega, delta1, delta2, nt, timestep)

        beta = 0.15835
        Delta = Omega * np.cos(beta) / np.sin(beta)
        omega = Omega / (np.sin(beta) * np.cos(beta))
        for timestep in range(1,nt):
            output = single(output, Omega, Delta, omega, delta1, delta2, nt, timestep)

        beta = 0.89884
        Delta = Omega * np.cos(beta) / np.sin(beta)
        omega = Omega / (np.sin(beta) * np.cos(beta))
        for timestep in range(1,nt):
            output = single(output, Omega, Delta, omega, delta1, delta2, nt, timestep)

        beta = 0.272815
        Delta = Omega * np.cos(beta) / np.sin(beta)
        omega = Omega / (np.sin(beta) * np.cos(beta))
        for timestep in range(1,nt):
            output = single(output, Omega, Delta, omega, delta1, delta2, nt, timestep)

        beta = 1.10936
        Delta = Omega * np.cos(beta) / np.sin(beta)
        omega = Omega / (np.sin(beta) * np.cos(beta))
        for timestep in range(1,nt):
            output = single(output, Omega, Delta, omega, delta1, delta2, nt, timestep)
        # print(output)

        gate4 = output

        # fidelity
        target = (1j * np.pi * tensor(sigmay(), sigmay()) / 4).expm()
        final = np.dot(np.kron(gate3, gate4), np.dot(gatemikio, np.kron(gate1, gate2)))
        f[jj, ii] = abs(np.trace(target.dag()*final)/4)**2
        # print(f)
        ii = ii + 1
        print('ii: %s' % ii)
    jj = jj + 1
    print('jj: %s' % jj)


# %%
fig, ax = plt.subplots()
norm = matplotlib.colors.Normalize(vmin=0.1, vmax=2.5)
im = ax.imshow(-np.log10(1-f),
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
# f_level=np.arange(0.1, 2.6, 0.4)
# CB = fig.colorbar(im, ticks=f_level)

# fig, ax = plt.subplots()
# nd = 101
delt = np.linspace(-0.2, 0.2, nd)
levels = np.array([0.5, 1, 2])
CS = ax.contour(-np.log10(1-f),
                levels=levels,
                origin='lower',
                extend='both',
                linewidths=0.8,
                colors='k',
                # cmap='coolwarm',
                extent=(delt[0], delt[-1], delt[0], delt[-1]))
ax.set_aspect(9/10)
ax.clabel(CS, inline=True, fontsize=12)

plt.scatter([-0.004], [-0.004], color='black')

ax.arrow(-0.004, -0.004, 0.06+, 0.12, head_width=0.01, head_length=0.025)
ax.arrow(-0.004, -0.004, 0.12, -0.07, head_width=0.01, head_length=0.025)

# %%
"""
CB = fig.colorbar(im, ax=ax, shrink=0.8, ticks=f_level)  # 添加cbar
plt.xticks(np.arange(0, 2.2, step=0.2),list('abcdefghigk'),rotation=45)  #自定义刻度标签值，刻度显示为您想要的一切（日期，星期等等）
CS.collections[n].set_linewidth(m)  # 选取第 n 个等高线设置宽度 m
ax.clabel(CS, levels[1::2], inline=True, fmt='%1.1f', fontsize=14)  # label every second level

plt.show()
fig.savefig('gauss.pdf',dpi=800,bbox_inches='tight',format='pdf')

XDelta, YOmega = np.meshgrid(delt, delt)
qsave(f,'fidelity')
qsave(XDelta,'xframe')
qsave(YOmega,'yframe')

f = qload('fidelity')
print((f[50,50]-f[50,60])/0.04)

"""

end = time.time()
print('Running time: %s seconds' % (end-start))

