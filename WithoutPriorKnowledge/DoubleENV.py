import importlib
import json
import os
from threading import Thread
from tensorforce import TensorforceError, util
import numpy as np
import math
import random
from scipy.linalg import expm, norm
from tensorforce.environments import Environment
from qutip import *



class YY(Environment):
    """
    Tensorforce environment interface.
    """

    def __init__(self):
        # first two arguments, if applicable: level, visualize=False
        self.observation = None
        self.timestep = 0
        self.tf = 2 #17.05
        self.omega = 0.5*np.pi#2*np.pi/self.tf #fixed
        self.strength = 2*np.pi
        self._max_episode_timesteps = 100 #100 pulses
        self.dt = self.tf/self._max_episode_timesteps #evolution time per step
        
        
        self._states = dict(shape=(38,), type='float') #no information on the controller encoded in RL states
        self.U = tensor(qeye(2), qeye(2))
        self.qtarget = (1j * np.pi * tensor(sigmay(), sigmay()) / 4).expm() # target gate
        
        
        self.lam_Delta = 0#np.random.normal(0, 0.1)
        self.lam_Omega = 0#np.random.normal(0, 0.1)
        self.lam_J = 0#np.random.normal(0, 0.15)
        
        # self._states = dict(shape=(11,), type='float') 
        # self.U = qeye(2) #initialized to I2*2
        # self.qtarget = Qobj(np.array([[1,-1j],[1j,-1]]/np.sqrt(2),dtype='complex'))
        

    def __str__(self):
        return self.__class__.__name__

    def states(self):
        """
        Returns the state space specification.
        Returns:
            specification: Arbitrarily nested dictionary of state descriptions with the following
            attributes:
            <ul>
            <li><b>type</b> (<i>"bool" | "int" | "float"</i>) &ndash; state data type
            (<span style="color:#00C000"><b>default</b></span>: "float").</li>
            <li><b>shape</b> (<i>int | iter[int]</i>) &ndash; state shape
            (<span style="color:#C00000"><b>required</b></span>).</li>
            <li><b>num_states</b> (<i>int > 0</i>) &ndash; number of discrete state values
            (<span style="color:#C00000"><b>required</b></span> for type "int").</li>
            <li><b>min_value/max_value</b> (<i>float</i>) &ndash; minimum/maximum state value
            (<span style="color:#00C000"><b>optional</b></span> for type "float").</li>
            </ul>
        """
        
        return self._states

    def actions(self):
        """
        Returns the action space specification.
        Returns:
            specification: Arbitrarily nested dictionary of action descriptions with the following
            attributes:
            <ul>
            <li><b>type</b> (<i>"bool" | "int" | "float"</i>) &ndash; action data type
            (<span style="color:#C00000"><b>required</b></span>).</li>
            <li><b>shape</b> (<i>int > 0 | iter[int > 0]</i>) &ndash; action shape
            (<span style="color:#00C000"><b>default</b></span>: scalar).</li>
            <li><b>num_actions</b> (<i>int > 0</i>) &ndash; number of discrete action values
            (<span style="color:#C00000"><b>required</b></span> for type "int").</li>
            <li><b>min_value/max_value</b> (<i>float</i>) &ndash; minimum/maximum action value
            (<span style="color:#00C000"><b>optional</b></span> for type "float").</li>
            </ul>
        """
        
        return dict(shape=(5,), type='float',min_value=0.0, max_value=1.0) 
        # return dict(shape=(2,), type='float',min_value=0.0, max_value=1.0) 

    def max_episode_timesteps(self):
        """
        Returns the maximum number of timesteps per episode.
        Returns:
            int: Maximum number of timesteps per episode.
        """
        return self._max_episode_timesteps

    def close(self):
        """
        Closes the environment.
        """
     
        
        self.environment = None

    def reset(self):
        """
        Resets the environment to start a new episode.
        Returns:
            dict[state]: Dictionary containing initial state(s) and auxiliary information.
        """
        
        self.STATES=np.array([1, 0, 0, 0,  # real part of 1st row
                              0, 1, 0, 0,  # real part of 2st row
                              0, 0, 1, 0,  # real part of 3st row
                              0, 0, 0, 1,  # real part of 4st row
                              0, 0, 0, 0,  # imaginary part of 1st row
                              0, 0, 0, 0,  # imaginary part of 2st row
                              0, 0, 0, 0,  # imaginary part of 3st row
                              0, 0, 0, 0,  # imaginary part of 4st row
                              0, 0.5, 0, 0.5, 0.5, 0])# Omega1 Delta1 Omega2 Delta2 J omega1 omega2 t
        self.U = tensor(qeye(2),qeye(2))
        
        
        # self.STATES=np.array([1, 0, 0, 1, 0, 0, 0, 0, 0, 0.5, 0])
        # self.U = qeye(2) #initialized to I2*2 
        
        
        self.lam_Delta = 0#np.random.normal(0, 0.1)
        self.lam_Omega = 0#np.random.normal(0, 0.1)
        self.lam_J = 0#np.random.normal(0, 0.15)
        
        
        self.timestep = 0
                                 
               
        return self.STATES


    def execute(self, actions):
        """
        Executes the given action(s) and advances the environment by one step.
        Args:
            actions (dict[action]): Dictionary containing action(s) to be executed
                (<span style="color:#C00000"><b>required</b></span>).
        Returns:
            ((dict[state], bool | 0 | 1 | 2, float)): Dictionary containing next state(s), whether
            a terminal state is reached or 2 if the episode was aborted, and observed reward.
        """
        
        systime = (self.timestep+1)*self.dt
        
        Rabi1 = actions[0]*self.strength
        Detuning1 = 2*(actions[1]-0.5)*self.strength
        # phi1 = actions[5]*self.strength
        H1 = 0.5*( Rabi1*(1+self.lam_Omega)*np.cos(self.omega*systime)*sigmax() + Rabi1*(1+self.lam_Omega)*np.sin(self.omega*systime)*sigmay() + Detuning1*(1+self.lam_Delta)*sigmaz())      
        # H1 = 0.5*( Rabi1*(1+self.lam_Omega)*np.cos(phi1)*sigmax() + Rabi1*(1+self.lam_Omega)*np.sin(phi1)*sigmay() + Detuning1*(1+self.lam_Delta)*sigmaz())      


        Rabi2 = actions[2]*self.strength
        Detuning2 = 2*(actions[3]-0.5)*self.strength
        # phi2 = actions[6]*self.strength
        H2 = 0.5*( Rabi2*(1+self.lam_Omega)*np.cos(self.omega*systime)*sigmax() + Rabi2*(1+self.lam_Omega)*np.sin(self.omega*systime)*sigmay() + Detuning2*(1+self.lam_Delta)*sigmaz())
        # H2 = 0.5*( Rabi2*(1+self.lam_Omega)*np.cos(phi2)*sigmax() + Rabi2*(1+self.lam_Omega)*np.sin(phi2)*sigmay() + Detuning2*(1+self.lam_Delta)*sigmaz())


        J = 2*(actions[4]-0.5)*self.strength
        H3 = 0.5 * J*(1+self.lam_J) * tensor(sigmaz(),sigmaz())
        
        
        H = tensor(H1,qeye(2)) + tensor(qeye(2),H2) + H3
        
        
        prop = (-1j * H * self.dt).expm()
        self.U = prop * self.U
        
        #now we write the qobj into array for encoding in RL state
        reU = np.real(np.array(self.U)).flatten()
        imU = np.imag(np.array(self.U)).flatten()
        self.timestep += 1
        
        
        self.STATES = np.concatenate((reU,imU,np.array([actions[0],actions[1],actions[2],actions[3],actions[4],self.timestep*self.dt])),axis=0)
        # self.STATES = np.concatenate((reU,imU,np.array([actions[0],actions[1],self.timestep*self.dt])),axis=0)
         

        f = np.abs(np.trace(self.qtarget.dag() * self.U)/4)**2
        
        reward = -np.log10(1-f)
        
        # reward = 0

        if self.timestep == self._max_episode_timesteps or f>=0.99:
            terminal = 1
            
            if 0.95<f<0.99:
                reward += 1
            elif f>=0.99:
                reward += 10
            else:
                reward = -np.log10(1-f)
                
        else:
            terminal = 0
            reward = 0
        
    
        return self.STATES, terminal, reward
            