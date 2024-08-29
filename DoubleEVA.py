from tensorforce.agents import TRPOAgent
from tensorforce.agents import PPOAgent
from tensorforce.agents import Agent
import DoubleENV
import numpy as np
from tensorforce.execution import Runner
import pickle
from qutip import *
import matplotlib.pyplot as plt
# from scipy import linalg
# import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'


YY=DoubleENV.YY()

#hidden layer
network_spec=[
        dict(type='dense',size=64,activation='relu'),
        dict(type='dense',size=64,activation='relu'),
        dict(type='dense',size=64,activation='relu'),
        ]


agent=PPOAgent(
        states=YY.states(),
        actions=YY.actions(),
        network=network_spec,
        max_episode_timesteps=YY.max_episode_timesteps(),
        batch_size=32,
        learning_rate=1*1e-4,
        )


agent.restore(directory='model_15',filename='agent-3813135')


sum_rewards = 0.0
# qstate=np.zeros(36)
states = YY.reset()
terminal = False
statelist=[]
actionlist=[]
while not terminal:
    YY.lam_Delta = 0.0
    YY.lam_Omega = 0.0
    YY.lam_J = 0
    actions = agent.act(states=states, evaluation=True)
    states, terminal, reward = YY.execute(actions=actions)
    sum_rewards += reward
    statelist.append(states)
    actionlist.append(actions)
print(reward)


time=[]
Omega1=[]
Delta1=[]
Omega2=[]
Delta2=[]
J=[]
for i in range(len(statelist)):
    Omega1.append(statelist[i][-6])
    Delta1.append(statelist[i][-5])
    Omega2.append(statelist[i][-4])
    Delta2.append(statelist[i][-3])
    J.append(statelist[i][-2])
    time.append(statelist[i][-1])
    
fig,ax=plt.subplots(1,1)
ax.plot(time,Omega1,time,Delta1,time,Omega2,time,Delta2,time,J)
ax.legend(("$\Omega1$","$\Delta1$","$\Omega2$","$\Delta2$","$J$"))
# ax.plot(time,Omega1,time,Omega2,time,J)
# ax.legend(("$\Omega1$","$\Omega2$","$J$"))


final_real = np.reshape(np.array_split(np.delete(statelist[-1], [-1, -2, -3,-4,-5,-6,]),2)[0],[4,4])
final_imag = np.reshape(np.array_split(np.delete(statelist[-1], [-1, -2, -3,-4,-5,-6,]),2)[1],[4,4])
# final_real = np.reshape(np.array_split(np.delete(statelist[-1],[-1,-2,-3]),2)[0],[2,2])
# final_imag = np.reshape(np.array_split(np.delete(statelist[-1],[-1,-2,-3]),2)[1],[2,2])

final = final_real+1j*final_imag

target= (1j * np.pi * tensor(sigmay(), sigmay()) / 4).expm()
# target = Qobj(np.array([[1,-1j],[1j,-1]]/np.sqrt(2),dtype='complex'))

f= abs(np.trace(target.dag()*final)/4)**2


print('delta error=',YY.lam_Delta,'omega error=', YY.lam_Omega, 'fidelity=',f)

"""
qsave(Omega1,'without_no_Omega1')
qsave(Delta1,'without_no_Delta1')
qsave(Omega2,'without_no_Omega2')
qsave(Delta2,'without_no_Delta2')
qsave(J,'without_no_J')
qsave(time,'without_no_time')

qsave(Omega1,'without_gauss_Omega1')
qsave(Delta1,'without_gauss_Delta1')
qsave(Omega2,'without_gauss_Omega2')
qsave(Delta2,'without_gauss_Delta2')
qsave(J,'without_gauss_J')
qsave(time,'without_gauss_time')

qsave(Omega1,'without_dropout_Omega1')
qsave(Delta1,'without_dropout_Delta1')
qsave(Omega2,'without_dropout_Omega2')
qsave(Delta2,'without_dropout_Delta2')
qsave(J,'without_dropout_J')
qsave(time,'without_dropout_time')
"""
