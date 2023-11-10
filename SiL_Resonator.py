#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is the Resonator with self-injection locked laser class for PyCORe
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import complex_ode,solve_ivp, ode
from scipy.sparse.linalg import expm
from scipy.sparse.linalg import inv as inv_sparse
from scipy.sparse.linalg import spsolve as solve_sparse
from scipy.linalg import dft
from scipy.linalg import solve as solve_dense
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
from scipy.constants import pi, c, hbar, e
from matplotlib.widgets import Slider, Button, TextBox
from matplotlib.animation import FuncAnimation
import matplotlib.image as mpimg
from scipy.optimize import curve_fit
import time
import sys, os
from scipy.sparse import block_diag,identity,diags, eye, csc_matrix, dia_matrix, isspmatrix
from scipy.sparse.linalg import eigs as scp_eigs
import ctypes
from scipy.linalg import eig, inv, solve, lu_factor, lu_solve
from scipy.optimize import minimize
from matplotlib.patches import Wedge
from matplotlib.collections import PatchCollection
import matplotlib.cm as cm

class_path = os.path.abspath(__file__)

# Get the directory containing the script
PyCORe_directory = os.path.dirname(class_path)

class SiL_Resonator(Resonator):
    def __init__(self):
        Resonator.__init__(self)
        self.alpha_h = 0 #Linewidth enhancement factor
        self.a = 0 ##differential gain
        self.N0 = 0 #Carrier density at transparency
        self.kappa_laser= 0 #Laser cavity loss rate
        self.kappa_inj = 0 #Laser-microresonator coupling rate
        self.I = 0 #Laser biased current
        self.gamma = 0 #Carrier recombination rate
        self.V = 0 #Volume of active section
        self.eta = 0# Conversion factor
        self.theta = 0 #optical feedback
        self.kappa_sc = 0#cw-ccw coupling rate
        self.zeta = 0 #Hz/A, Current-frequency tuning coefficient
    def Init_From_Dict(self,ResonatorParameters,LaserParameters):
        Resonator.Init_From_Dict(self,ResonatorParameters)
        self.alpha_h = LaserParameters['alpha_h']
        self.a = LaserParameters['a'] ##differential gain
        self.N0 = LaserParameters['N0'] #Carrier density at transparency
        self.kappa_laser= LaserParameters['kappa_laser'] #Laser cavity loss rate
        self.kappa_inj = LaserParameters['kappa_inj'] #Laser-microresonator coupling rate
        self.I = LaserParameters['I'] #Laser biased current
        self.gamma = LaserParameters['gamma'] #Carrier recombination rate
        self.V = LaserParameters['V'] #Volume of active section
        self.eta = LaserParameters['eta']# Conversion factor
        self.theta = LaserParameters['theta']#optical feedback
        self.kappa_sc = ResonatorParameters['kappa_sc']
        self.zeta = LaserParameters['zeta']#Hz/A, Current-frequency tuning coefficient
    def seed_level (self, detuning):
        resonator_field = np.zeros(self.N_points,dtype=complex)
        N = self.N0+self.kappa_laser/self.a/self.V
        E_laser = np.sqrt((-self.I/e/self.V+self.gamma*N)/self.kappa_laser,dtype=complex)
        resonator_field[:] = self.eta*self.kappa_inj*np.exp(self.theta*1j)*E_laser/(self.kappa/2+1j*detuning)
        CCW_amplitude = np.complex_(1j*resonator_field[0]*self.kappa_sc/(self.kappa/2+1j*detuning))
        return resonator_field, CCW_amplitude, E_laser, N
        
    def Propagate_PseudoSpectralSAMCLIB(self, simulation_parameters, dt=5e-4, Seed=[0], CCW_seed=0., E_laser=0., N=0., HardSeed=False):
        #start_time = time.time()
        T = simulation_parameters['slow_time']
        out_param = simulation_parameters['output']
        detuning = simulation_parameters['detuning_array']
        eps = simulation_parameters['noise_level']
        abtol = simulation_parameters['absolute_tolerance']
        reltol = simulation_parameters['relative_tolerance']
        #dt = simulation_parameters['time_step']#in photon lifetimes
        
      
        
        if HardSeed == False:
            resonator_field, CCW_amplitude, E_laser, N = self.seed_level(detuning[0])
            seed = np.sqrt(2*self.g0/self.kappa)*resonator_field
            CCW_seed  = CCW_amplitude*np.sqrt(2*self.g0/self.kappa)
        else:
            seed = Seed*np.sqrt(2*self.g0/self.kappa)
            CCW_seed  = CCW_amplitude*np.sqrt(2*self.g0/self.kappa)
        ### renormalization
        T_rn = (self.kappa/2)*T
        
        f0_direct_sq = np.abs(self.eta*2*self.kappa_inj/self.kappa*np.sqrt(2*self.g0/self.kappa)*E_laser)**2
        print('f0^2 = ' + str(np.round(f0_direct_sq, 2)))
        print('xi [' + str(detuning[0]*2/self.kappa) + ',' +str(detuning[-1]*2/self.kappa)+ ']')
        
        
        t_st = float(T_rn)/len(detuning)
        #dt=1e-4 #t_ph
        
        sol_res = np.ndarray(shape=(len(detuning), self.N_points), dtype='complex') # define an array to store the data
        sol_CCW_res = np.zeros([len(detuning)],dtype=complex)
        sol_Laser = np.zeros([len(detuning)],dtype=complex)
        sol_N = np.zeros([len(detuning)])
        sol_res[0,:] = (seed)/self.N_points
        sol_CCW_res[0] = CCW_seed*self.N_points
        sol_Laser[0] = E_laser#*self.N_points
        sol_N[0] = N#*self.N_points
        print( sol_res[0,0],sol_CCW_res[0], sol_Laser[0] , sol_N[0] )
        
        #%% crtypes defyning
        LLE_core = ctypes.CDLL(PyCORe_directory+'/lib/lib_lle_core.so')
        
        LLE_core.Propagate_SiL_PseudoSpectralSAM.restype = ctypes.c_void_p
        #LLE_core.Propagate_PseudoSpectralSAM.restype = ctypes.c_void_p
        #%% defining the ctypes variables
        
        A = np.zeros(self.N_points+3,dtype=complex)
        A[:self.N_points] = np.fft.ifft(seed)#*self.N_points
        A[self.N_points] = CCW_seed
        A[self.N_points+1] = E_laser
        A[self.N_points+2] = N
        #plt.plot(abs(A))
        #print(A[0])
        
        In_val_RE = np.array(np.real(A),dtype=ctypes.c_double)
        In_val_IM = np.array(np.imag(A),dtype=ctypes.c_double)
        In_phi = np.array(self.phi,dtype=ctypes.c_double)
        In_Nphi = ctypes.c_int(self.N_points)
        In_kappa = ctypes.c_double(np.max(self.kappa))
        In_g0 = ctypes.c_double(self.g0)
        
        In_I_laser = ctypes.c_double(self.I)
        In_zeta = ctypes.c_double(self.zeta)
        In_a = ctypes.c_double(self.a)
        In_e = ctypes.c_double(e)
        In_alpha_h = ctypes.c_double(self.alpha_h)
        In_N0 = ctypes.c_double(self.N0)
        In_kappa_laser = ctypes.c_double(self.kappa_laser)
        In_kappa_inj = ctypes.c_double(self.kappa_inj)
        In_gamma = ctypes.c_double(self.gamma)
        In_V = ctypes.c_double(self.V)
        In_eta = ctypes.c_double(self.eta)
        In_theta = ctypes.c_double(self.theta)
        In_kappa_sc = ctypes.c_double(self.kappa_sc)
        
        
        
        In_atol = ctypes.c_double(abtol)
        In_rtol = ctypes.c_double(reltol)
      
        In_det = np.array(2/self.kappa*detuning,dtype=ctypes.c_double)
        In_Ndet = ctypes.c_int(len(detuning))
        In_Dint = np.array(self.Dint*2/self.kappa,dtype=ctypes.c_double)
        In_Tmax = ctypes.c_double(T_rn)
        In_Tstep = ctypes.c_double(t_st)
    
        In_Nt = ctypes.c_int(int(t_st/dt)+1)
        In_dt = ctypes.c_double(dt)
        In_noise_amp = ctypes.c_double(eps)
        
        
            

            
            
            
        In_res_RE = np.zeros(len(detuning)*(self.N_points+3),dtype=ctypes.c_double)
        In_res_IM = np.zeros(len(detuning)*(self.N_points+3),dtype=ctypes.c_double)
        
        double_p=ctypes.POINTER(ctypes.c_double)
        In_val_RE_p = In_val_RE.ctypes.data_as(double_p)
        In_val_IM_p = In_val_IM.ctypes.data_as(double_p)
        In_phi_p = In_phi.ctypes.data_as(double_p)
        In_det_p = In_det.ctypes.data_as(double_p)
        In_Dint_p = In_Dint.ctypes.data_as(double_p)
        
        
        In_res_RE_p = In_res_RE.ctypes.data_as(double_p)
        In_res_IM_p = In_res_IM.ctypes.data_as(double_p)
        #%%running simulations
        
        LLE_core.Propagate_SiL_PseudoSpectralSAM(In_val_RE_p, In_val_IM_p, In_det_p, In_kappa, In_kappa_laser, In_kappa_sc, In_kappa_inj, In_theta, In_g0, In_alpha_h, In_gamma, In_V, In_a, In_e, In_N0, In_eta, In_I_laser , In_zeta, In_Dint_p, In_Ndet, In_Nt, In_Tmax, In_Tstep, In_dt, In_atol, In_rtol, In_Nphi, In_noise_amp, In_res_RE_p, In_res_IM_p)
        indexes = np.arange(self.N_points+3)
        ind_modes = np.arange(self.N_points)
        for ii in range(0,len(detuning)):
            sol_res[ii,ind_modes] = np.fft.fft(In_res_RE[ii*(self.N_points+3)+ind_modes] + 1j*In_res_IM[ii*(self.N_points+3)+ind_modes])
            sol_CCW_res[ii] = In_res_RE[ii*(self.N_points+3)+self.N_points]  + 1j*In_res_IM[ii*(self.N_points+3)+self.N_points]
            sol_Laser[ii] = In_res_RE[ii*(self.N_points+3)+self.N_points+1]  + 1j*In_res_IM[ii*(self.N_points+3)+self.N_points+1]
            sol_N[ii] = In_res_RE[ii*(self.N_points+3)+self.N_points+2]  
        #sol = np.reshape(In_res_RE,[len(detuning),self.N_points]) + 1j*np.reshape(In_res_IM,[len(detuning),self.N_points])
                    
        if out_param == 'map':
            return sol_res/np.sqrt(2*self.g0/self.kappa), sol_CCW_res/np.sqrt(2*self.g0/self.kappa), sol_Laser, sol_N
        elif out_param == 'fin_res':
            return sol_res[-1, :]/np.sqrt(2*self.g0/self.kappa), sol_CCW_res[-1]/np.sqrt(2*self.g0/self.kappa), sol_Laser[-1], sol_N[-1]
        else:
            print ('wrong parameter')  