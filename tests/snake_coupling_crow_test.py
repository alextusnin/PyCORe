#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 09:36:43 2021

@author: tusnin
"""

import matplotlib.pyplot as plt
import numpy as np
import sys, os
curr_dir = os.getcwd()
PyCore_dir = os.path.dirname(curr_dir)
sys.path.append(PyCore_dir)
import PyCORe_main as pcm
from scipy.constants import hbar
import time

start_time = time.time()

Num_of_modes = 2**9
N_crow = 3

D2 = 4.1e6#-1*beta2*L/Tr*D1**2 ## From beta2 to D2

D3 = 0
mu = np.arange(-Num_of_modes/2,Num_of_modes/2)
Dint_single = 2*np.pi*(mu**2*D2/2 + mu**3*D3/6)
Dint = np.zeros([mu.size,N_crow])
Dint = (Dint_single*np.ones([mu.size,N_crow]).T).T#Making matrix of dispersion with dispersion profile of j-th resonator on the j-th column





J = 0*5e9*2*np.pi*np.ones([mu.size,(N_crow)])

#dNu_ini = -2*J.max()/2/np.pi-100e6
dNu_ini = -1e3
dNu_end = 1e9#3*J.max()/2/np.pi+2e9

#dNu_ini = 0#3*J.max()/2/np.pi+2e9
#dNu_end = 3*J.max()/2/np.pi+1e9

nn = 10000
ramp_stop = 1
dOm = 2*np.pi*np.concatenate([np.linspace(dNu_ini,dNu_end, int(nn*ramp_stop)),dNu_end*np.ones(int(np.round((1-ramp_stop)*nn)))])

#delta = 0.1e9*2*np.pi
kappa_ex_ampl = 200e6*2*np.pi
kappa_ex = np.zeros([Num_of_modes,N_crow])

for ii in range(0,N_crow,2):
    kappa_ex[:,ii] = kappa_ex_ampl*np.ones([Num_of_modes])

Delta = np.zeros([mu.size,(N_crow)])

N_cells = (N_crow+1)//2
bus_coupling=np.zeros([mu.size,N_cells])
bus_phases = np.ones(N_cells-1)*np.pi/2
for ii in range(0,N_crow,2):
    bus_coupling[:,ii//2] = -kappa_ex[:,ii]

    
#Delta[:,0] = 2*np.pi*1e9*np.ones([Num_of_modes])

PhysicalParameters = {'Inter-resonator_coupling': J,
                      'Snake bus coupling' : bus_coupling,
                      'Snake bus phases' : bus_phases,
                      'Resonator detunings' : Delta,
                      'n0' : 1.9,
                      'n2' : 2.4e-19,### m^2/W
                      'FSR' : 181.7e9 ,
                      'w0' : 2*np.pi*192e12,
                      'width' : 1.5e-6,
                      'height' : 0.85e-6,
                      'kappa_0' : 50e6*2*np.pi,
                      'kappa_ex' : kappa_ex,
                      'Dint' : Dint}

simulation_parameters = {'slow_time' : 1e-6,
                         'detuning_array' : dOm,
                         'noise_level' : 1e-8,
                         'output' : 'map',
                         'absolute_tolerance' : 1e-8,
                         'relative_tolerance' : 1e-8,
                         'max_internal_steps' : 2000}

P0 = .5### W

Pump = np.zeros([len(mu),N_crow],dtype='complex')

phase = 0
Pump[0,0] = np.sqrt(P0)
#for ii in range(2,N_crow,2):
#    phase+=bus_phases[ii//2-1]
#    Pump[0,ii] = np.sqrt(P0)*np.exp(1j*phase)

#%%
crow = pcm.CROW()
crow.Init_From_Dict(PhysicalParameters)
#ev = crow.Linear_analysis()

#%%

map2d = crow.Propagate_SAMCLIB(simulation_parameters, Pump, BC='PERIODIC')
#map2d = crow.Propagate_SAMCLIB_PSEUD_SPECT(simulation_parameters, Pump)
#map2d = crow.Propagate_SAM(simulation_parameters, Pump)
#%%
plt.figure()
plt.plot(dOm/2/np.pi,np.mean(np.abs(map2d[:,:,-2])**2,axis=1))
plt.plot(dOm/2/np.pi,np.mean(np.abs(map2d[:,:,1])**2,axis=1))
pcm.Plot_Map(np.fft.ifft(map2d[:,:,-2],axis=1),dOm/2/np.pi)

print("--- %s seconds ---" % (time.time() - start_time))
#%%
S = np.sqrt(P0)/np.sqrt(crow.w0*hbar)*np.exp(1j*np.sum(bus_phases))#*Num_of_modes
trans = np.zeros_like(dOm,dtype=complex)
for ii in range(nn):
    trans[ii]=S
    for jj in range(0,N_crow,2):
        field = np.mean(map2d[ii,:,jj])#/np.sqrt(Num_of_modes)
        trans[ii]-=np.sqrt(kappa_ex[0,jj])*field*np.exp(1j*np.sum(bus_phases[jj//2:]))
        
#%%
fig = plt.figure(figsize=[3.6*2,2.2*2],frameon=True)
ax = fig.add_subplot(1,1,1)
ax.plot(dOm,abs(trans/S)**2)
ax.set_ylim(0,1.1)
ax.set_ylabel("Transmission")
ax.set_xlabel("Detuning (abs. units)")