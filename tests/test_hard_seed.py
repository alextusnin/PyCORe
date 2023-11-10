#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import sys,os
from scipy.constants import c,hbar

sys.path.append(os.path.abspath(__file__)[:-23])

def SeedSol(device,D2,det, P): 

    result = np.zeros([device.N_points],dtype = complex)
    DKS = np.zeros(device.N_points,dtype=complex)
    CW = np.zeros(device.N_points,dtype=complex)
    phi = device.phi
    
    
    
    f = np.sqrt(P/hbar/device.w0)*np.sqrt(8*device.g0*device.kappa_ex/device.kappa**3)
    zeta_0 = det*2/device.kappa
    DKS_phase = np.arccos(np.sqrt(8*zeta_0)/f/np.pi)
  
    
    CW = f/(1+1j*zeta_0)*0
    DKS = np.sqrt(2*zeta_0)*1/np.cosh(np.sqrt(2*zeta_0)*(phi-np.pi)*np.sqrt(device.kappa/2/D2))*np.exp(1j*DKS_phase)
    
    result[:] = np.fft.fft(CW+DKS)
    return result*np.sqrt(device.kappa/2/device.g0)
#%%
import PyCORe_main as pcm
import time

start_time = time.time()

map2d_scan = np.zeros([],dtype=complex)#np.load('map2d_scan.npy')
dOm_scan = np.zeros([])
Pump=np.zeros([],dtype=complex)
simulation_parameters={}
device = pcm.Resonator()
#device=pcm.CROW()
simulation_parameters,map2d_scan,dOm_scan,Pump=device.Init_From_File('./data/')

idet = 1500
nn = 30000
dOm = np.ones(nn)*dOm_scan[idet]
simulation_parameters['slow_time']=1e-6
simulation_parameters['detuning_array']=dOm

#Seed = SeedSol(device, single_ring.D2, dOm[idet], abs(Pump[0])**2)
Seed = map2d_scan[idet,:]
#Seed_old = map2d_scan[idet,:]#/device.N_points

#Seed = SeedSol(device,single_ring.D2,dOm[idet],abs(Pump[0])**2)
#%%
#map2d = device.Propagate_SAM(simulation_parameters, Pump)
#map2d = device.Propagate_SplitStepCLIB(simulation_parameters, Pump,Seed=Seed,dt=0.5e-3, HardSeed=True)
#map2d = device.Propagate_SAMCLIB(simulation_parameters, Pump,Seed=Seed,HardSeed=True,BC='OPEN')
#map2d = device.Propagate_PSEUDO_SPECTRAL_SAMCLIB(simulation_parameters, Pump,Seed=Seed,HardSeed=True, BC='OPEN')
map2d = device.Propagate_PseudoSpectralSAMCLIB(simulation_parameters, Pump,Seed=Seed,HardSeed=True,dt=0.5e-3,lib='NR')
#map2d = device.Propagate_SAMCLIB(simulation_parameters, Pump,Seed=Seed,HardSeed=True)
#map2d = device.Propagate_SplitStep(simulation_parameters, Pump,dt=1e-3)
#%%
#plt.figure()
#plt.plot(dOm/2/np.pi,np.mean(np.abs(map2d)**2,axis=1))
#%%

#pcm.Plot_Map(np.fft.ifft(map2d,axis=1),dOm*2/device.kappa)
#pcm.Plot_Map(np.fft.ifft(map2d,axis=1),np.arange(nn))
#np.save('map2d_'+str(idet),map2d[:,:],allow_pickle=True)
pcm.Plot_Map(np.fft.ifft(map2d[:,:,0],axis=1),np.arange(dOm.size))
print("--- %s seconds ---" % (time.time() - start_time))
#%%
res,rel_diff = device.NewtonRaphsonFixedD1(map2d[-1,:],dOm[-1],Pump,tol=1e-6,max_iter=25)
#%%
eig_vals, eig_vecs = device.LinearStability(res,dOm[-1])
#%%
#GoldStone_index = np.argmax(np.real(eig_vals))
#GoldStone_mode = eig_vecs[:,np.argmax(np.real(eig_vals))]

#%%
#seed = res+0.1*np.fft.fft(np.roll(np.fft.ifft(GoldStone_mode),0))
#map2d = device.Propagate_SAMCLIB(simulation_parameters, Pump,Seed=seed,HardSeed=True)
#pcm.Plot_Map(np.fft.ifft(map2d,axis=1),np.arange(nn))

#plt.figure()
#plt.plot(device.phi[np.argmax(np.abs(np.fft.ifft(map2d[:,:],axis=1))**2,axis=1)]-np.pi)
