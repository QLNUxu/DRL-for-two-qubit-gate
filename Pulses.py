#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 23:07:55 2023

@author: xutianniu
"""

from qutip import *
import numpy as np
import matplotlib.pyplot as plt

# Omega1no = np.concatenate((np.array([0]),np.array(qload('without_no_Omega1'))))
# Delta1no = np.concatenate((np.array([0.5]),np.array(qload('without_no_Delta1'))))
# Omega2no = np.concatenate((np.array([0]),np.array(qload('without_no_Omega2'))))
# Delta2no = np.concatenate((np.array([0.5]),np.array(qload('without_no_Delta2'))))
# Jno = np.concatenate((np.array([0.5]),np.array(qload('without_no_J'))))

# Omega1gau = np.concatenate((np.array([0]),np.array(qload('without_gauss_Omega1'))))
# Delta1gau = np.concatenate((np.array([0.5]),np.array(qload('without_gauss_Delta1'))))
# Omega2gau = np.concatenate((np.array([0]),np.array(qload('without_gauss_Omega2'))))
# Delta2gau = np.concatenate((np.array([0.5]),np.array(qload('without_gauss_Delta2'))))
# Jgau = np.concatenate((np.array([0.5]),np.array(qload('without_gauss_J'))))

# Omega1dp = np.concatenate((np.array([0]),np.array(qload('without_dropout_Omega1'))))
# Delta1dp = np.concatenate((np.array([0.5]),np.array(qload('without_dropout_Delta1'))))
# Omega2dp = np.concatenate((np.array([0]),np.array(qload('without_dropout_Omega2'))))
# Delta2dp = np.concatenate((np.array([0.5]),np.array(qload('without_dropout_Delta2'))))
# Jdp = np.concatenate((np.array([0.5]),np.array(qload('without_dropout_J'))))


Omega1no = np.array(qload('without_no_Omega1'))
Delta1no = np.array(qload('without_no_Delta1'))
Omega2no = np.array(qload('without_no_Omega2'))
Delta2no = np.array(qload('without_no_Delta2'))
Jno = np.array(qload('without_no_J'))

Omega1gau = np.array(qload('without_gauss_Omega1'))
Delta1gau = np.array(qload('without_gauss_Delta1'))
Omega2gau = np.array(qload('without_gauss_Omega2'))
Delta2gau = np.array(qload('without_gauss_Delta2'))
Jgau = np.array(qload('without_gauss_J'))

Omega1dp = np.array(qload('without_dropout_Omega1'))
Delta1dp = np.array(qload('without_dropout_Delta1'))
Omega2dp = np.array(qload('without_dropout_Omega2'))
Delta2dp = np.array(qload('without_dropout_Delta2'))
Jdp = np.array(qload('without_dropout_J'))
# t = np.array(qload('without_no_time'))



####################################
colors = plt.cm.coolwarm(np.linspace(0,1,2))


Y = np.linspace(0, 100, 100)


fig,ax=plt.subplots()
ax.step(Y,Omega1no, color=colors[0],linewidth=1.2,linestyle='solid',label=r"$\tilde{\Omega}_1^{no}$")
ax.step(Y,Omega1gau, color=colors[1],linewidth=1.2,linestyle='dashed',label=r"$\tilde{\Omega}_1^{gau}$")
ax.step(np.delete(Y,-1),Omega1dp, color='black',linewidth=1.2,linestyle='dotted',label=r"$\tilde{\Omega}_1^{dp}$")
ax.set_xlabel(r'$t/T$', fontsize=14)  # Add an x-label to the axes.
ax.set_ylabel(r'$\Omega_1(t)/\Omega_{max}$',fontsize=14)  # Add a y-label to the axes.
ax.set_yticks([0.2, 0.4, 0.6, 0.8])
ax.set_yticklabels([0.2, 0.4, 0.6, 0.8], fontsize='14')
ax.set_xticks([0, 25, 50, 75, 100])
ax.set_xticklabels([0, 0.25, 0.5, 0.75, 1], fontsize='14')
ax.set_xlim(-2, 102)
ax.set_ylim(0.18, 0.82)
# plt.legend(loc='upper right',fontsize='12')
fig.savefig('Puls_Omega1.pdf',dpi=800,bbox_inches='tight',format='pdf')


fig,ax=plt.subplots()
ax.step(Y,Omega2no, color=colors[0],linewidth=1.2,linestyle='solid',label=r"$\tilde{\Omega}_2^{no}$")
ax.step(Y,Omega2gau, color=colors[1],linewidth=1.2,linestyle='dashed',label=r"$\tilde{\Omega}_2^{gau}$")
ax.step(np.delete(Y,-1),Omega2dp, color='black',linewidth=1.2,linestyle='dotted',label=r"$\tilde{\Omega}_2^{dp}$")
ax.set_xlabel(r'$t/T$', fontsize=14)  # Add an x-label to the axes.
ax.set_ylabel(r'$\Omega_2(t)/\Omega_{max}$',fontsize=14)  # Add a y-label to the axes.
ax.set_yticks([0.2, 0.4, 0.6, 0.8])
ax.set_yticklabels([0.2, 0.4, 0.6, 0.8], fontsize='14')
ax.set_xticks([0, 25, 50, 75, 100])
ax.set_xticklabels([0, 0.25, 0.5, 0.75, 1], fontsize='14')
ax.set_xlim(-2, 102)
ax.set_ylim(0.18, 0.82)
# plt.legend(loc='best',fontsize='12')
fig.savefig('Puls_Omega2.pdf',dpi=800,bbox_inches='tight',format='pdf')


fig,bx=plt.subplots()
bx.step(Y,Delta1no, color=colors[0],linewidth=1.2,linestyle='solid',label=r"$\tilde{\Delta}_1^{no}$")
bx.step(Y,Delta1gau, color=colors[1],linewidth=1.2,linestyle='dashed',label=r"$\tilde{\Delta}_1^{gau}$")
bx.step(np.delete(Y,-1),Delta1dp, color='black',linewidth=1.2,linestyle='dotted',label=r"$\tilde{\Delta}_1^{dp}$")
bx.set_xlabel(r'$t/T$', fontsize=14)  # Add an x-label to the bxes.
bx.set_ylabel(r'$\Delta_1(t)/\Delta_{max}$',fontsize=14)  # Add a y-label to the bxes.
bx.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
bx.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1], fontsize='14')
bx.set_xticks([0, 25, 50, 75, 100])
bx.set_xticklabels([0, 0.25, 0.5, 0.75, 1], fontsize='14')
bx.set_xlim(-2, 102)
bx.set_ylim(-0.02, 0.82)
# plt.legend(loc='best',fontsize='12')
fig.savefig('Puls_Delta1.pdf',dpi=800,bbox_inches='tight',format='pdf')


fig,bx=plt.subplots()
bx.step(Y,Delta2no, color=colors[0],linewidth=1.2,linestyle='solid',label=r"$\tilde{\Delta}_2^{no}$")
bx.step(Y,Delta2gau, color=colors[1],linewidth=1.2,linestyle='dashed',label=r"$\tilde{\Delta}_2^{gau}$")
bx.step(np.delete(Y,-1),Delta2dp, color='black',linewidth=1.2,linestyle='dotted',label=r"$\tilde{\Delta}_2^{dp}$")
bx.set_xlabel(r'$t/T$', fontsize=14)  # Add an x-label to the bxes.
bx.set_ylabel(r'$\Delta_2(t)/\Delta_{max}$',fontsize=14)  # Add a y-label to the bxes.
bx.set_yticks([0.1, 0.3, 0.5, 0.7])
bx.set_yticklabels([0.1, 0.3, 0.5, 0.7], fontsize='14')
bx.set_xticks([0, 25, 50, 75, 100])
bx.set_xticklabels([0, 0.25, 0.5, 0.75, 1], fontsize='14')
bx.set_xlim(-2, 102)
bx.set_ylim(0.08, 0.72)
# plt.legend(loc='best',fontsize='12')
fig.savefig('Puls_Delta2.pdf',dpi=800,bbox_inches='tight',format='pdf')


fig,cx=plt.subplots()
cx.step(Y,Jno, color=colors[0],linewidth=1.2,linestyle='solid',label=r"$\tilde{J}^{no}$")
cx.step(Y,Jgau, color=colors[1],linewidth=1.2,linestyle='dashed',label=r"$\tilde{J}^{gau}$")
cx.step(np.delete(Y,-1),Jdp, color='black',linewidth=1.2,linestyle='dashdot',label=r"$\tilde{J}^{dp}$")
cx.set_xlabel(r'$t/T$', fontsize=14)  # Add an x-label to the cxes.
cx.set_ylabel(r'$J(t)/J_{max}$',fontsize=14)  # Add a y-label to the cxes.
cx.set_yticks([0.1, 0.3, 0.5, 0.7, 0.9])
cx.set_yticklabels([0.1, 0.3, 0.5, 0.7, 0.9], fontsize='14')
cx.set_xticks([0, 25, 50, 75, 100])
cx.set_xticklabels([0, 0.25, 0.5, 0.75, 1], fontsize='14')
cx.set_xlim(-2, 102)
cx.set_ylim(0.08, 0.92)
# plt.legend(loc='best',fontsize='12')
fig.savefig('Puls_J.pdf',dpi=800,bbox_inches='tight',format='pdf')

###################################
"""
fig.savefig('Puls_Omega.pdf',dpi=800,bbox_inches='tight',format='pdf')
fig.savefig('Puls_Delta.pdf',dpi=800,bbox_inches='tight',format='pdf')
fig.savefig('Puls_J.pdf',dpi=800,bbox_inches='tight',format='pdf')
"""
