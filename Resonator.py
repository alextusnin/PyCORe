#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is the Resonator class for PyCORe
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


class Resonator:
    '''
    It's a main class for any resonator structure
    
    __init__(self) is just a constructor where we initialize all the physical parameters
    
    
    '''
    
    def __init__(self):
        '''
        Initialization of the physical parameters of a resonator 
        Parameters:
            n0: refractive index 
            n2: Nonlinear refractive index [m^2/W]
            FSR: Free Spectral Range [Hz]
            w0: Optical frequency  [rad Hz]
            width: waveguide width [m]
            height: weveguide height [m]
            Aeff = width*height effective mode area [m^2]
            kappa_0: internal resonator linewidth [rad Hz]
            kappa_ex: coupling rate to the bus waveguide [rad Hz]
            kappa = kappa_0 + kappa_ex is the loaded linewidth [rad Hz]
            Dint: integrated dispersion [rad Hz]
            Tr = 1/FSR: roundrtip time [s]
            Leff = c/n0*Tr: effective lentght of the mode [m] 
            Veff = Leff*Aeff: effective mode volume [m^3]
            g0 = hbar*w0^2*c*n2/n0^2/Veff: single-photon Kerr frequency shift [rad Hz]
            gamma = n2*w0/c/Aeff: Nonlinear coefficient in the Fiber notations for NLSE [1/W/m^2]
            N_points: number of longitudial modes we consider in simulations
            tau_r: Raman shock time [s]
            phi: azimuthal resonator coordinate going ranged from -pi to pi
            D2: GVD [rad Hz]
            D3: 3rd order dispersion [rad Hz]
            n2t: thermal nonlinearity coefficient
            t_th: thermal response time [s]
            J_EO: electro-optical coupling for resonant EO comb generation [rad Hz]            
            D: dispersion matrix 
            FirstDmat: auxilary matrix to compute first derivative on phi for Newton-Raphson method
            
            
            
        Returns
        -------
        None.

        '''
        self.n0 = 0
        self.n2 = 0
        self.FSR = 0
        self.w0 = 0
        self.width = 0
        self.height = 0
        self.kappa_0 = 0
        self.kappa_ex = 0
        self.Dint = np.array([])
        #Auxiliary physical parameters
        self.Tr = 0
        self.Aeff = 0
        self.Leff = 0
        self.Veff = 0
        self.g0 = 0
        self.gamma = 0
        self.kappa = self.kappa_0 + self.kappa_ex
        self.N_points = len(self.Dint)
        self.tau_r = 0
    
        self.phi = np.array([])
        
        self.D2 = 0
        self.D3 = 0
        
        self.D2_mod = 0
        
        self.n2t = 0
        self.t_th=0
        
        self.J_EO = 0
        self.D=np.zeros([0],dtype=complex)
        self.FirstDmat=np.zeros([0],dtype=complex)
    
    def Init_From_File(self,data_dir):
        '''

        Parameters
        ----------
        data_dir : string. Path to the directory with the parameters

        Returns
        -------
        simulation_parameters : dictionary with the simulation parameters
        map2d : 2D simulation data. 1st index stands for index along slow time (detuning), 2nd for intsanteneous spectrum. numpy complex
        dOm : detuning array. numpy complex
        Pump : pump array in spectrum representation. numpy complex

        '''
        simulation_parameters={}
        map2d=np.array([],dtype=complex)
        Pump=np.array([],dtype=complex)
        dOm=np.array([])
       
        for file in os.listdir(data_dir+'class_parameters/'):
            if file.endswith('.npy'):
                key = os.path.splitext(file)[0]
                print(file + " is open")
                self.__dict__[key] = np.load(data_dir+'class_parameters/'+file)
        for file in os.listdir(data_dir+'sim_parameters/'):
            if file.endswith('.npy'):
                key = os.path.splitext(file)[0]
                simulation_parameters[key] = np.load(data_dir+'sim_parameters/'+file)
        map2d=np.load(data_dir+'map2d.npy')
        dOm=np.load(data_dir+'dOm.npy')
        Pump=np.load(data_dir+'Pump.npy')
        
        return simulation_parameters, map2d, dOm, Pump
    def Init_From_Dict(self, resonator_parameters):
        '''
        

        Parameters
        ----------
        resonator_parameters :  resonator parameters in dict format. Example:
                                'n0',
                                'n2',
                                'FSR',
                                'w0',
                                'width',
                                'kappa_0'
                                'kappa_ex'
                                'Dint'
                                'Raman time'
                                

        Returns
        -------
        Initializes all the parameters to a given value

        '''
        #Physical parameters initialization
        self.n0 = resonator_parameters['n0']
        self.n2 = resonator_parameters['n2']
        self.FSR = resonator_parameters['FSR']
        self.w0 = resonator_parameters['w0']
        self.width = resonator_parameters['width']
        self.height = resonator_parameters['height']
        self.kappa_0 = resonator_parameters['kappa_0']
        self.kappa_ex = resonator_parameters['kappa_ex']
        self.Dint = np.fft.ifftshift(resonator_parameters['Dint'])
        self.tau_r = resonator_parameters['Raman time']
        
        #Auxiliary physical parameters
        self.Tr = 1/self.FSR #round trip time
        self.Aeff = self.width*self.height 
        self.Leff = c/self.n0*self.Tr 
        self.Veff = self.Aeff*self.Leff 
        self.g0 = hbar*self.w0**2*c*self.n2/self.n0**2/self.Veff
        self.gamma = self.n2*self.w0/c/self.Aeff
        self.kappa = self.kappa_0 + self.kappa_ex
        self.N_points = len(self.Dint)
        self.mu = np.fft.fftshift(np.arange(-self.N_points/2, self.N_points/2))
        self.phi = np.linspace(0,2*np.pi,self.N_points)
        self.D = self.DispersionMatrix(order=0)
        def func(x, a, b, c, d):
            return a + x*b + c*x**2/2 + d*x**3/6
        popt, pcov = curve_fit(func, self.mu, self.Dint)
        self.D2 = popt[2]
        self.D3 = popt[3]
        
        if 'Modulated D2' in resonator_parameters.keys():
            self.D2_mod = resonator_parameters['Modulated D2']
        else:
            self.D2_mod = 0
        
        if 'T thermal' in resonator_parameters.keys():
            self.t_th = resonator_parameters['T thermal']
            self.n2t = resonator_parameters['n2 thermal']
        
        if 'electro-optical coupling' in resonator_parameters.keys():
            self.J_EO =  resonator_parameters['electro-optical coupling']
        else:
            self.J_EO = 0
        
    def Save_Data(self,map2d,Pump,Simulation_Params,dOm=[0],directory='./'):
        params = self.__dict__
        try: 
            os.mkdir(directory+'class_parameters/')
            os.mkdir(directory+'sim_parameters/')
        except:
            pass
        for key in params.keys():
            np.save(directory+'class_parameters/'+key+'.npy',params[key])
        for key in Simulation_Params:
            np.save(directory+'sim_parameters/'+key+'.npy',Simulation_Params[key])
        np.save(directory+'map2d.npy',map2d)
        np.save(directory+'dOm.npy',dOm)
        np.save(directory+'Pump.npy',Pump)
        
        #print(params.keys())
        
    
    def noise(self, a):
#        return a*np.exp(1j*np.random.uniform(-1,1,self.N_points)*np.pi)
        return a*(np.random.uniform(-1,1,self.N_points) + 1j*np.random.uniform(-1,1,self.N_points))

    #   Propagate Using the Step Adaptive  Method
    def Propagate_SAM(self, simulation_parameters, Pump, Seed=[0], Normalized_Units=False):
        start_time = time.time()

        T = simulation_parameters['slow_time']
        abtol = simulation_parameters['absolute_tolerance']
        reltol = simulation_parameters['relative_tolerance']
        out_param = simulation_parameters['output']
        nmax = simulation_parameters['max_internal_steps']
        detuning = simulation_parameters['detuning_array']
        eps = simulation_parameters['noise_level']
        
        
        if Normalized_Units == False:
            pump = Pump*np.sqrt(1./(hbar*self.w0))
            if Seed[0] == 0:
                seed = self.seed_level(Pump, detuning[0])*np.sqrt(2*self.g0/self.kappa)
            else:
                seed = Seed*np.sqrt(2*self.g0/self.kappa)
            ### renormalization
            T_rn = (self.kappa/2)*T
            f0 = pump*np.sqrt(8*self.g0*self.kappa_ex/self.kappa**3)
            J = self.J_EO
            J*=2/self.kappa
            print('f0^2 = ' + str(np.round(max(abs(f0)**2), 2)))
            print('xi [' + str(detuning[0]*2/self.kappa) + ',' +str(detuning[-1]*2/self.kappa)+ ']')
        else:
            pump = Pump
            if Seed[0] == 0:
                seed = self.seed_level(Pump, detuning[0],Normalized_Units)
            else:
                seed = Seed
            T_rn = T
            f0 = pump
            print(r'Normalized pump power $f_0^2$ = ' + str(np.round(max(abs(f0)**2), 2)))
            print(r'Normalized detuning $\zeta_0$ = [' + str(detuning[0]) + ',' +str(detuning[-1])+ ']')
            detuning*=self.kappa/2
        noise_const = self.noise(eps) # set the noise level
        nn = len(detuning)
        ### define the rhs function
        def LLE_1d(Time, A):
            A = A - noise_const#self.noise(eps)
            A_dir = np.fft.ifft(A)*len(A)## in the direct space
            dAdT =  -1*(1 + 1j*(self.Dint + dOm_curr)*2/self.kappa)*A + 1j*np.fft.fft(A_dir*np.abs(A_dir)**2)/len(A) + 1j*np.fft.fft(J*2/self.kappa*np.cos(self.phi)*A_dir/self.N_points) + f0#*len(A)
            return dAdT
        
        t_st = float(T_rn)/len(detuning)
        r = complex_ode(LLE_1d).set_integrator('dop853', atol=abtol, rtol=reltol,nsteps=nmax)# set the solver
        #r = ode(LLE_1d).set_integrator('zvode', atol=abtol, rtol=reltol,nsteps=nmax)# set the solver
        r.set_initial_value(seed, 0)# seed the cavity
        sol = np.ndarray(shape=(len(detuning), self.N_points), dtype='complex') # define an array to store the data
        sol[0,:] = seed
        #printProgressBar(0, nn, prefix = 'Progress:', suffix = 'Complete', length = 50, fill='elapsed time = ' + str((time.time() - start_time)) + ' s')
        for it in range(1,len(detuning)):
            self.printProgressBar(it + 1, nn, prefix = 'Progress:', suffix = 'Complete,', time='elapsed time = ' + '{:04.1f}'.format(time.time() - start_time) + ' s', length = 50)
            #self.print('elapsed time = ', (time.time() - start_time))
            dOm_curr = detuning[it] # detuning value
            sol[it] = r.integrate(r.t+t_st)
            
        if out_param == 'map':
            if Normalized_Units == False :
                return sol/np.sqrt(2*self.g0/self.kappa)
            else:
                detuning/=self.kappa/2
                return sol
        elif out_param == 'fin_res':
            return sol[-1, :]/np.sqrt(2*self.g0/self.kappa)
        else:
            print ('wrong parameter')
    def Propagate_SAM_NEW(self, simulation_parameters, Pump, Seed=[0], Normalized_Units=False):
        start_time = time.time()

        T = simulation_parameters['slow_time']
        abtol = simulation_parameters['absolute_tolerance']
        reltol = simulation_parameters['relative_tolerance']
        out_param = simulation_parameters['output']
        nmax = simulation_parameters['max_internal_steps']
        detuning = simulation_parameters['detuning_array']
        eps = simulation_parameters['noise_level']
        if 'electro-optical coupling' in simulation_parameters.keys():
            J =  simulation_parameters['electro-optical coupling']
        else:
            J = 0
        
        if Normalized_Units == False:
            pump = Pump*np.sqrt(1./(hbar*self.w0))
            if Seed[0] == 0:
                seed = self.seed_level(Pump, detuning[0])*np.sqrt(2*self.g0/self.kappa)
            else:
                seed = Seed*np.sqrt(2*self.g0/self.kappa)
            ### renormalization
            T_rn = (self.kappa/2)*T
            f0 = pump*np.sqrt(8*self.g0*self.kappa_ex/self.kappa**3)
            J*=2/self.kappa
            print('f0^2 = ' + str(np.round(max(abs(f0)**2), 2)))
            print('xi [' + str(detuning[0]*2/self.kappa) + ',' +str(detuning[-1]*2/self.kappa)+ ']')
        else:
            pump = Pump
            if Seed[0] == 0:
                seed = self.seed_level(Pump, detuning[0],Normalized_Units)
            else:
                seed = Seed
            T_rn = T
            f0 = pump
            print(r'Normalized pump power $f_0^2$ = ' + str(np.round(max(abs(f0)**2), 2)))
            print(r'Normalized detuning $\zeta_0$ = [' + str(detuning[0]) + ',' +str(detuning[-1])+ ']')
            detuning*=self.kappa/2
        noise_const = self.noise(eps) # set the noise level
        nn = len(detuning)
        ### define the rhs function
        # def LLE_1d(Time, A):
        #     #A = A - noise_const#self.noise(eps)
        #     A_dir = np.fft.ifft(A)*len(A)## in the direct space
        #     dAdT =  -1*(1 + 1j*(self.Dint + dOm_curr)*2/self.kappa)*A + 1j*np.fft.fft(A_dir*np.abs(A_dir)**2)/len(A)  + f0#*len(A)
        #     return dAdT
        disp_operator = self.Dint*2/self.kappa
        f0_dir = np.fft.ifft(f0)*self.N_points
        print(r'Normalized pump power $f_0^2$ = ' + str(np.round(max(abs(f0_dir)**2), 2)))
        def LLE_1d(t,A):
            #A+=noise_const
            dAdt = np.fft.ifft((-1j*disp_operator-(self.kappa + 1j*dOm_curr*2)/self.kappa)*(np.fft.fft(A)) ) +1j*np.abs(A)**2*A+f0_dir -1j*self.tau_r*self.FSR*2*np.pi*A*np.fft.ifft(np.fft.fft(np.abs(A)**2))
            return dAdt
        
        t_st = float(T_rn)/len(detuning)
        #r = complex_ode(LLE_1d).set_integrator('dop853', atol=abtol, rtol=reltol,nsteps=nmax)# set the solver
        #r = ode(LLE_1d).set_integrator('zvode', atol=abtol, rtol=reltol,nsteps=nmax)# set the solver
        #r.set_initial_value(seed, 0)# seed the cavity
        
        sol = np.ndarray(shape=(len(detuning), self.N_points), dtype='complex') # define an array to store the data
        sol[0,:] = np.fft.ifft(seed+noise_const)#/self.N_points
        print(np.sum(abs(sol[0,:]))**2)
        #printProgressBar(0, nn, prefix = 'Progress:', suffix = 'Complete', length = 50, fill='elapsed time = ' + str((time.time() - start_time)) + ' s')
        T_span = np.linspace(0,T_rn,len(detuning))
        for it in range(1,len(detuning)):
            self.printProgressBar(it + 1, nn, prefix = 'Progress:', suffix = 'Complete,', time='elapsed time = ' + '{:04.1f}'.format(time.time() - start_time) + ' s', length = 50)
            #self.print('elapsed time = ', (time.time() - start_time))
            dOm_curr = detuning[it] # detuning value
            #sol[it] = r.integrate(r.t+t_st)
            sol[it,:]=(solve_ivp( LLE_1d, t_span=[T_span[it-1],T_span[it]], y0=sol[it-1,:], method='DOP853', t_eval=[T_span[it]], atol=abtol, rtol=reltol,max_step=nmax,first_step=1e-3, min_step=1e-5)).y.T
            #print(np.sum(abs(sol[it,:]))**2)
            
        if out_param == 'map':
            if Normalized_Units == False :
                return np.fft.fft(sol,axis=1)/np.sqrt(2*self.g0/self.kappa)/self.N_points
            else:
                detuning/=self.kappa/2
                return np.fft.fft(sol,axis=1)
        elif out_param == 'fin_res':
            return np.fft.fft(sol[-1, :])/np.sqrt(2*self.g0/self.kappa)
        else:
            print ('wrong parameter')   
    def Propagate_SplitStep(self, simulation_parameters, Pump, Seed=[0], dt=5e-4, Normalized_Units=False):
        start_time = time.time()
        T = simulation_parameters['slow_time']
        out_param = simulation_parameters['output']
        detuning = simulation_parameters['detuning_array']
        eps = simulation_parameters['noise_level']
        J =  self.J_EO
        
        #dt = simulation_parameters['time_step']#in photon lifetimes
        
        if Normalized_Units == False:
            pump = Pump*np.sqrt(1./(hbar*self.w0))
            if Seed[0] == 0:
                seed = self.seed_level(Pump, detuning[0],Normalized_Units)*np.sqrt(2*self.g0/self.kappa)
            else:
                seed = Seed*np.sqrt(2*self.g0/self.kappa)
            ### renormalization
            T_rn = (self.kappa/2)*T
            f0 = pump*np.sqrt(8*self.g0*self.kappa_ex/self.kappa**3)
            J*=2/self.kappa
            print('f0^2 = ' + str(np.round(max(abs(f0)**2), 2)))
            print('xi [' + str(detuning[0]*2/self.kappa) + ',' +str(detuning[-1]*2/self.kappa)+ ']')
        else:
            pump = Pump
            if Seed[0] == 0:
                seed = self.seed_level(Pump, detuning[0],Normalized_Units)
            else:
                seed = Seed
            T_rn = T
            f0 = pump
            print(r'Normalized pump power $f_0^2$ = ' + str(np.round(max(abs(f0)**2), 2)))
            print(r'Normalized detuning $\zeta_0$ = [' + str(detuning[0]) + ',' +str(detuning[-1])+ ']')
            detuning*=self.kappa/2
        
        noise_const = self.noise(eps) # set the noise level
        nn = len(detuning)
        
        print('J = ' + str(J))
        t_st = float(T_rn)/len(detuning)
        #dt=1e-4 #t_ph
        
        sol = np.ndarray(shape=(len(detuning), self.N_points), dtype='complex') # define an array to store the data
        sol[0,:] = seed
        #f0 = np.fft.ifft(f0)*self.N_points
        #f0*=self.N_points
        self.printProgressBar(0, nn, prefix = 'Progress:', suffix = 'Complete', length = 50)
        for it in range(1,len(detuning)):
            noise_const = self.noise(eps)
            self.printProgressBar(it + 1, nn, prefix = 'Progress:', suffix = 'Complete,', time='elapsed time = ' + '{:04.1f}'.format(time.time() - start_time) + ' s', length = 50)
            dOm_curr = detuning[it] # detuning value
            t=0
            buf = sol[it-1,:]
            buf-=noise_const
            buf = np.fft.ifft(buf)*len(buf)
            while t<t_st:
                
                # First step
                
                #buf = np.fft.fft(np.exp(dt*(1j*np.abs(buf)**2+1j*J*(np.cos(self.phi) + 0.*np.sin(2*self.phi)) + f0/buf))*buf)
                buf = np.fft.fft(np.exp(dt*(1j*np.abs(buf)**2+1j*J*(np.cos(self.phi) + 0.*np.sin(2*self.phi))))*buf)
                #second step
                
                #buf = np.fft.ifft(np.exp(-dt *(1+1j*(self.Dint + dOm_curr)*2/self.kappa )) *buf)
                buf = np.fft.ifft(np.exp(-dt *(1+1j*(self.Dint + dOm_curr)*2/self.kappa )) *buf + f0*self.N_points/(-1-1j*(self.Dint + dOm_curr)*2/self.kappa)*(np.exp(dt*(-1-1j*(self.Dint + dOm_curr)*2/self.kappa)) -1.))
                
                t+=dt
            sol[it,:] = np.fft.fft(buf)/len(buf)
            #sol[it,:] = buf
            
        if out_param == 'map':
            if Normalized_Units == False :
                return sol/np.sqrt(2*self.g0/self.kappa)
            else:
                detuning/=self.kappa/2
                return sol
        elif out_param == 'fin_res':
            return sol[-1, :]/np.sqrt(2*self.g0/self.kappa)
        else:
            print ('wrong parameter') 
            
    def Propagate_SplitStepCLIB(self, simulation_parameters, Pump, Seed=[0], dt=5e-4, HardSeed=False):
        #start_time = time.time()
        T = simulation_parameters['slow_time']
        out_param = simulation_parameters['output']
        detuning = simulation_parameters['detuning_array']
        eps = simulation_parameters['noise_level']
        J =  self.J_EO
        #dt = simulation_parameters['time_step']#in photon lifetimes
        
        pump = Pump*np.sqrt(1./(hbar*self.w0))
        
        if HardSeed == False:
            seed = self.seed_level(Pump, detuning[0])*np.sqrt(2*self.g0/self.kappa)
        else:
            seed = Seed*np.sqrt(2*self.g0/self.kappa)/self.N_points
        ### renormalization
        T_rn = (self.kappa/2)*T
        f0 = pump*np.sqrt(8*self.g0*self.kappa_ex/self.kappa**3)
        j = J/self.kappa*2
        print(r'Normalized pump power $f_0^2$ = ' + str(np.round(max(abs(f0)**2), 2)))
        print(r'Normalized detuning $\zeta_0$ = [' + str(detuning[0]*2/self.kappa) + ',' +str(detuning[-1]*2/self.kappa)+ ']')
        
        #noise_const = self.noise(eps) # set the noise level
        #nn = len(detuning)
        
        t_st = float(T_rn)/len(detuning)
        #dt=1e-4 #t_ph
        
        sol = np.ndarray(shape=(len(detuning), self.N_points), dtype='complex') # define an array to store the data
        sol[0,:] = (seed)
        
        #%% crtypes defyning
        if self.D2_mod == 0:
            LLE_core = ctypes.CDLL(PyCORe_directory+'/lib/lib_lle_core.so')
        else:
            double_p=ctypes.POINTER(ctypes.c_double)
            LLE_core = ctypes.CDLL(PyCORe_directory+'/lib/lib_lle_core_faraday.so')
            In_D2_mod = np.array(2/self.kappa*self.D2_mod*self.mu**2,dtype=ctypes.c_double)
            In_D2_mod_p = In_D2_mod.ctypes.data_as(double_p)
            
            In_FSR = ctypes.c_double(self.FSR)
            In_kappa = ctypes.c_double(self.kappa)
        LLE_core.PropagateSS.restype = ctypes.c_void_p
        #%% defining the ctypes variables
        
        A = np.fft.ifft(seed)*(len(seed))
        #A = seed
        In_val_RE = np.array(np.real(A),dtype=ctypes.c_double)
        In_val_IM = np.array(np.imag(A),dtype=ctypes.c_double)
        In_phi = np.array(self.phi,dtype=ctypes.c_double)
        In_Nphi = ctypes.c_int(self.N_points)
        In_f_RE = np.array(np.real(f0),dtype=ctypes.c_double)
        In_f_IM = np.array(np.imag(f0),dtype=ctypes.c_double)
        In_J = ctypes.c_double(j)
        
        In_det = np.array(2/self.kappa*detuning,dtype=ctypes.c_double)
        In_Ndet = ctypes.c_int(len(detuning))
        In_Dint = np.array(self.Dint*2/self.kappa,dtype=ctypes.c_double)
        In_Tmax = ctypes.c_double(t_st)
        In_Nt = ctypes.c_int(int(t_st/dt)+1)
        In_dt = ctypes.c_double(dt)
        In_noise_amp = ctypes.c_double(eps)
        
        In_res_RE = np.zeros(len(detuning)*self.N_points,dtype=ctypes.c_double)
        In_res_IM = np.zeros(len(detuning)*self.N_points,dtype=ctypes.c_double)
        
        double_p=ctypes.POINTER(ctypes.c_double)
        In_val_RE_p = In_val_RE.ctypes.data_as(double_p)
        In_val_IM_p = In_val_IM.ctypes.data_as(double_p)
        In_phi_p = In_phi.ctypes.data_as(double_p)
        In_det_p = In_det.ctypes.data_as(double_p)
        In_Dint_p = In_Dint.ctypes.data_as(double_p)
        In_f_RE_p = In_f_RE.ctypes.data_as(double_p)
        In_f_IM_p = In_f_IM.ctypes.data_as(double_p)
        
        In_res_RE_p = In_res_RE.ctypes.data_as(double_p)
        In_res_IM_p = In_res_IM.ctypes.data_as(double_p)
        #%%running simulations
        if self.D2_mod ==0:
            LLE_core.PropagateSS(In_val_RE_p, In_val_IM_p, In_f_RE_p, In_f_IM_p, In_det_p, In_J, In_phi_p, In_Dint_p, In_Ndet, In_Nt, In_dt, In_Nphi, In_noise_amp, In_res_RE_p, In_res_IM_p)
        else:
            LLE_core.PropagateSS(In_val_RE_p, In_val_IM_p, In_f_RE_p, In_f_IM_p, In_det_p, In_J, In_phi_p, In_Dint_p, In_D2_mod_p, In_FSR, In_kappa, In_Ndet, In_Nt, In_dt, In_Nphi, In_noise_amp, In_res_RE_p, In_res_IM_p)
        ind_modes = np.arange(self.N_points)
        for ii in range(0,len(detuning)):
            sol[ii,ind_modes] = (In_res_RE[ii*self.N_points+ind_modes] + 1j*In_res_IM[ii*self.N_points+ind_modes])*self.N_points
        #sol = np.reshape(In_res_RE,[len(detuning),self.N_points]) + 1j*np.reshape(In_res_IM,[len(detuning),self.N_points])
                    
        if out_param == 'map':
            return sol/np.sqrt(2*self.g0/self.kappa)
        elif out_param == 'fin_res':
            return sol[-1, :]/np.sqrt(2*self.g0/self.kappa)
        else:
            print ('wrong parameter')
        
    def Propagate_SAMCLIB(self, simulation_parameters, Pump, Seed=[0], dt=5e-4,HardSeed=False):
        #start_time = time.time()
        T = simulation_parameters['slow_time']
        out_param = simulation_parameters['output']
        detuning = simulation_parameters['detuning_array']
        eps = simulation_parameters['noise_level']
        J =  self.J_EO
        abtol = simulation_parameters['absolute_tolerance']
        reltol = simulation_parameters['relative_tolerance']
        #dt = simulation_parameters['time_step']#in photon lifetimes
        
        pump = Pump*np.sqrt(1./(hbar*self.w0))
        
        if HardSeed == False:
            seed = self.seed_level(Pump, detuning[0])*np.sqrt(2*self.g0/self.kappa)
        else:
            seed = Seed*np.sqrt(2*self.g0/self.kappa)
        ### renormalization
        T_rn = (self.kappa/2)*T
        f0 = np.fft.ifft(pump*np.sqrt(8*self.g0*self.kappa_ex/self.kappa**3))*self.N_points
        j = J/self.kappa*2
        print('f0^2 = ' + str(np.round(max(abs(f0)**2), 2)))
        print('xi [' + str(detuning[0]*2/self.kappa) + ',' +str(detuning[-1]*2/self.kappa)+ ']')
        
        #noise_const = self.noise(eps) # set the noise level
        #nn = len(detuning)
        
        t_st = float(T_rn)/len(detuning)
        #dt=1e-4 #t_ph
        
        sol = np.ndarray(shape=(len(detuning), self.N_points), dtype='complex') # define an array to store the data
        sol[0,:] = (seed)/self.N_points
        
        #%% crtypes defyning
        LLE_core = ctypes.CDLL(PyCORe_directory+'/lib/lib_lle_core.so')
        if self.n2t==0:
            LLE_core.PropagateSAM.restype = ctypes.c_void_p
        else:
            LLE_core.PropagateThermalSAM.restype = ctypes.c_void_p
        
        #%% defining the ctypes variables
        
        A = np.fft.ifft(seed)#*self.N_points
        
        #plt.plot(abs(A))
        
        In_val_RE = np.array(np.real(A),dtype=ctypes.c_double)
        In_val_IM = np.array(np.imag(A),dtype=ctypes.c_double)
        In_phi = np.array(self.phi,dtype=ctypes.c_double)
        In_Nphi = ctypes.c_int(self.N_points)
        In_f_RE = np.array(np.real(f0),dtype=ctypes.c_double)
        In_f_IM = np.array(np.imag(f0),dtype=ctypes.c_double)
        In_atol = ctypes.c_double(abtol)
        In_rtol = ctypes.c_double(reltol)
        In_J = ctypes.c_double(j)
        In_det = np.array(2/self.kappa*detuning,dtype=ctypes.c_double)
        In_Ndet = ctypes.c_int(len(detuning))
        In_Dint = np.array(self.Dint*2/self.kappa,dtype=ctypes.c_double)
        In_Tmax = ctypes.c_double(t_st)
        In_Nt = ctypes.c_int(int(t_st/dt)+1)
        In_dt = ctypes.c_double(dt)
        In_noise_amp = ctypes.c_double(eps)
        
        
            
        if self.n2t!=0:
            In_kappa = ctypes.c_double(self.kappa_0+self.kappa_ex)
            In_t_th = ctypes.c_double(self.t_th)
            In_n2 = ctypes.c_double(self.n2)
            In_n2t = ctypes.c_double(self.n2t)
            
            
            
        In_res_RE = np.zeros(len(detuning)*self.N_points,dtype=ctypes.c_double)
        In_res_IM = np.zeros(len(detuning)*self.N_points,dtype=ctypes.c_double)
        
        double_p=ctypes.POINTER(ctypes.c_double)
        In_val_RE_p = In_val_RE.ctypes.data_as(double_p)
        In_val_IM_p = In_val_IM.ctypes.data_as(double_p)
        In_phi_p = In_phi.ctypes.data_as(double_p)
        In_det_p = In_det.ctypes.data_as(double_p)
        In_Dint_p = In_Dint.ctypes.data_as(double_p)
        In_f_RE_p = In_f_RE.ctypes.data_as(double_p)
        In_f_IM_p = In_f_IM.ctypes.data_as(double_p)
        
        In_res_RE_p = In_res_RE.ctypes.data_as(double_p)
        In_res_IM_p = In_res_IM.ctypes.data_as(double_p)
        #%%running simulations
        if self.n2t==0:
            LLE_core.PropagateSAM(In_val_RE_p, In_val_IM_p, In_f_RE_p, In_f_IM_p, In_det_p, In_J, In_phi_p, In_Dint_p, In_Ndet, In_Nt, In_dt, In_atol, In_rtol, In_Nphi, In_noise_amp, In_res_RE_p, In_res_IM_p)
        else:
            LLE_core.PropagateThermalSAM(In_val_RE_p, In_val_IM_p, In_f_RE_p, In_f_IM_p, In_det_p, In_J, In_t_th, In_kappa, In_n2, In_n2t, In_phi_p, In_Dint_p, In_Ndet, In_Nt, In_dt, In_atol, In_rtol, In_Nphi, In_noise_amp, In_res_RE_p, In_res_IM_p)
        ind_modes = np.arange(self.N_points)
        for ii in range(0,len(detuning)):
            sol[ii,ind_modes] = np.fft.fft(In_res_RE[ii*self.N_points+ind_modes] + 1j*In_res_IM[ii*self.N_points+ind_modes])
            
        #sol = np.reshape(In_res_RE,[len(detuning),self.N_points]) + 1j*np.reshape(In_res_IM,[len(detuning),self.N_points])
                    
        if out_param == 'map':
            return sol/np.sqrt(2*self.g0/self.kappa)
        elif out_param == 'fin_res':
            return sol[-1, :]/np.sqrt(2*self.g0/self.kappa)
        else:
            print ('wrong parameter')
            
   
    def Propagate_PseudoSpectralSAMCLIB(self, simulation_parameters, Pump, Seed=[0], dt=5e-4,HardSeed=False,lib='NR'):
        #start_time = time.time()
        T = simulation_parameters['slow_time']
        out_param = simulation_parameters['output']
        detuning = simulation_parameters['detuning_array']
        eps = simulation_parameters['noise_level']
        J =  self.J_EO
        abtol = simulation_parameters['absolute_tolerance']
        reltol = simulation_parameters['relative_tolerance']
        #dt = simulation_parameters['time_step']#in photon lifetimes
        
        pump = Pump*np.sqrt(1./(hbar*self.w0))
        
        if HardSeed == False:
            seed = self.seed_level(Pump, detuning[0])*np.sqrt(2*self.g0/self.kappa)
        else:
            seed = Seed*np.sqrt(2*self.g0/self.kappa)
        ### renormalization
        T_rn = (self.kappa/2)*T
        f0 = np.fft.ifft(pump*np.sqrt(8*self.g0*self.kappa_ex/self.kappa**3))*self.N_points
        j = J/self.kappa*2
        print(r'Normalized pump power $f_0^2$ = ' + str(np.round(max(abs(f0)**2), 2)))
        print(r'Normalized detuning $\zeta_0$ = [' + str(detuning[0]*2/self.kappa) + ',' +str(detuning[-1]*2/self.kappa)+ ']')
       
        #noise_const = self.noise(eps) # set the noise level
        #nn = len(detuning)
        
        t_st = float(T_rn)/len(detuning)
        #dt=1e-4 #t_ph
        
        sol = np.ndarray(shape=(len(detuning), self.N_points), dtype='complex') # define an array to store the data
        sol[0,:] = (seed)/self.N_points
        
        #%% ctypes defyning
        
        if lib=='NR':
            LLE_core = ctypes.CDLL(PyCORe_directory+'/lib/lib_lle_core.so')
        else:
            LLE_core = ctypes.CDLL(PyCORe_directory+'/lib/lib_boost_lle_core.so')
            
        LLE_core.Propagate_PseudoSpectralSAM.restype = ctypes.c_void_p
        if self.tau_r !=0:
            LLE_core.Propagate_PseudoSpectralSAM_Raman.restype = ctypes.c_void_p
        #%% defining the ctypes variables
        
        A = np.fft.ifft(seed)#*self.N_points
        
        #plt.plot(abs(A))
        
        In_val_RE = np.array(np.real(A),dtype=ctypes.c_double)
        In_val_IM = np.array(np.imag(A),dtype=ctypes.c_double)
        In_phi = np.array(self.phi,dtype=ctypes.c_double)
        In_tau_r_mu = np.array(self.mu*self.tau_r*self.FSR*2*np.pi,dtype=ctypes.c_double)
        In_Nphi = ctypes.c_int(self.N_points)
        In_f_RE = np.array(np.real(f0),dtype=ctypes.c_double)
        In_f_IM = np.array(np.imag(f0),dtype=ctypes.c_double)
        In_atol = ctypes.c_double(abtol)
        In_rtol = ctypes.c_double(reltol)
        In_J = ctypes.c_double(j)
        In_det = np.array(2/self.kappa*detuning,dtype=ctypes.c_double)
        In_Ndet = ctypes.c_int(len(detuning))
        In_Dint = np.array(self.Dint*2/self.kappa,dtype=ctypes.c_double)
        In_Tmax = ctypes.c_double(t_st)
        In_Nt = ctypes.c_int(int(t_st/dt)+1)
        In_dt = ctypes.c_double(dt)
        In_noise_amp = ctypes.c_double(eps)
        
        
            
        if self.n2t!=0:
            In_kappa = ctypes.c_double(self.kappa_0+self.kappa_ex)
            In_t_th = ctypes.c_double(self.t_th)
            In_n2 = ctypes.c_double(self.n2)
            In_n2t = ctypes.c_double(self.n2t)
            
            
            
        In_res_RE = np.zeros(len(detuning)*self.N_points,dtype=ctypes.c_double)
        In_res_IM = np.zeros(len(detuning)*self.N_points,dtype=ctypes.c_double)
        
        double_p=ctypes.POINTER(ctypes.c_double)
        In_val_RE_p = In_val_RE.ctypes.data_as(double_p)
        In_val_IM_p = In_val_IM.ctypes.data_as(double_p)
        In_phi_p = In_phi.ctypes.data_as(double_p)
        In_tau_r_mu_p = In_tau_r_mu.ctypes.data_as(double_p)
        In_det_p = In_det.ctypes.data_as(double_p)
        In_Dint_p = In_Dint.ctypes.data_as(double_p)
        In_f_RE_p = In_f_RE.ctypes.data_as(double_p)
        In_f_IM_p = In_f_IM.ctypes.data_as(double_p)
        
        In_res_RE_p = In_res_RE.ctypes.data_as(double_p)
        In_res_IM_p = In_res_IM.ctypes.data_as(double_p)
        #%%running simulations
        if self.n2t==0:
            if self.tau_r == 0:
                LLE_core.Propagate_PseudoSpectralSAM(In_val_RE_p, In_val_IM_p, In_f_RE_p, In_f_IM_p, In_det_p, In_J, In_phi_p, In_Dint_p, In_Ndet, In_Nt, In_dt, In_atol, In_rtol, In_Nphi, In_noise_amp, In_res_RE_p, In_res_IM_p)
            else:
                LLE_core.Propagate_PseudoSpectralSAM_Raman(In_val_RE_p, In_val_IM_p, In_f_RE_p, In_f_IM_p, In_det_p, In_tau_r_mu_p, In_Dint_p, In_Ndet, In_Nt, In_dt, In_atol, In_rtol, In_Nphi, In_noise_amp, In_res_RE_p, In_res_IM_p)
        else:
            #LLE_core.PropagateThermalSAM(In_val_RE_p, In_val_IM_p, In_f_RE_p, In_f_IM_p, In_det_p, In_J, In_t_th, In_kappa, In_n2, In_n2t, In_phi_p, In_Dint_p, In_Ndet, In_Nt, In_dt, In_atol, In_rtol, In_Nphi, In_noise_amp, In_res_RE_p, In_res_IM_p)
            pass
        
        ind_modes = np.arange(self.N_points)
        for ii in range(0,len(detuning)):
            sol[ii,ind_modes] = np.fft.fft(In_res_RE[ii*self.N_points+ind_modes] + 1j*In_res_IM[ii*self.N_points+ind_modes])
            
        #sol = np.reshape(In_res_RE,[len(detuning),self.N_points]) + 1j*np.reshape(In_res_IM,[len(detuning),self.N_points])
                    
        if out_param == 'map':
            return sol/np.sqrt(2*self.g0/self.kappa)
        elif out_param == 'fin_res':
            return sol[-1, :]/np.sqrt(2*self.g0/self.kappa)
        else:
            print ('wrong parameter')
            
#%%            
              
 
    #%%

    def seed_level (self, pump, detuning, Normalized_Units=False):
        if Normalized_Units == False:
            f_norm = pump*np.sqrt(1./(hbar*self.w0))*np.sqrt(8*self.g0*self.kappa_ex/self.kappa**3)
            detuning_norm  = detuning*2/self.kappa
            stat_roots = np.roots([1, -2*detuning_norm, (detuning_norm**2+1), -abs(f_norm[0])**2])
            ind_roots = [np.imag(ii)==0 for ii in stat_roots]
            res_seed = np.zeros_like(f_norm)
            res_seed[0] = abs(np.min(stat_roots[ind_roots]))**.5/np.sqrt(2*self.g0/self.kappa)
        else:
            f_norm = pump
            detuning_norm  = detuning
            stat_roots = np.roots([1, -2*detuning_norm, (detuning_norm**2+1), -abs(f_norm[0])**2])
            ind_roots = [np.imag(ii)==0 for ii in stat_roots]
            res_seed = np.zeros_like(f_norm)
            res_seed[0] = abs(np.min(stat_roots[ind_roots]))**.5
        return res_seed
    
    def seed_soliton(self, pump, detuning):
        fast_t = np.linspace(-pi,pi,len(pump))*np.sqrt(self.kappa/2/self.D2)
        f_norm = abs(pump[0]*np.sqrt(1./(hbar*self.w0))*np.sqrt(8*self.g0*self.kappa_ex/self.kappa**3))
        detuning_norm  = detuning*2/self.kappa
        stat_roots = np.roots([1, -2*detuning_norm, (detuning_norm**2+1), -abs(f_norm)**2])
        
        ind_roots = [np.imag(ii)==0 for ii in stat_roots]
        B = np.sqrt(2*detuning_norm)
        print(detuning_norm)
        print(f_norm)
        level = np.min(np.abs(stat_roots[ind_roots]))
        print(level)
        return np.fft.fft(level**.5*np.exp(1j*np.arctan((detuning_norm-level)/f_norm)) + B*np.exp(1j*np.arccos(2*B/np.pi/f_norm))*np.cosh(B*fast_t)**-1)/np.sqrt(2*self.g0/self.kappa)/len(pump)
        
        
    def NeverStopSAM (self, T_step, detuning_0=-1, Pump_P=2., nmax=1000, abtol=1e-10, reltol=1e-9, out_param='fin_res'):
        self.Pump = self.Pump/abs(self.Pump)
        def deriv_1(dt, field_in):
        # computes the first-order derivative of field_in
            field_fft = np.fft.fft(field_in)
            omega = 2.*np.pi*np.fft.fftfreq(len(field_in),dt)
            out_field = np.fft.ifft(-1j*omega*field_fft)
            return out_field
        
        def deriv_2(dt, field_in):
        # computes the second-order derivative of field_in
            field_fft = np.fft.fft(field_in)
            omega = 2.*np.pi*np.fft.fftfreq(len(field_in),dt)
            field_fft *= -omega**2
            out_field = np.fft.ifft(field_fft)
            return out_field 
        
        def disp(field_in,Dint_in):
        # computes the dispersion term in Fourier space
            field_fft = np.fft.fft(field_in)
            out_field = np.fft.ifft(Dint_in*field_fft)     
            return out_field

        ### define the rhs function
        def LLE_1d(Z, A):
            # for nomalized
            if np.size(self.Dint)==1 and self.Dint == 1:
                 dAdt2 = deriv_2(self.TimeStep, A)
                 dAdT =  1j*dAdt2/2 + 1j*self.gamma*self.L/self.Tr*np.abs(A)**2*A - (self.kappa/2+1j*dOm_curr)*A + np.sqrt(self.kappa/2/self.Tr)*self.Pump*Pump_P**.5
            elif np.size(self.Dint)==1 and self.Dint == -1:
                 dAdt2 = deriv_2(self.TimeStep, A)
                 dAdT =  -1j*dAdt2/2 + 1j*self.gamma*self.L/self.Tr*np.abs(A)**2*A - (self.kappa/2+1j*dOm_curr)*A + np.sqrt(self.kappa/2/self.Tr)*self.Pump*Pump_P**.5
            else:  
                # with out raman
                Disp_int = disp(A,self.Dint)
                if self.Traman==0:
                    dAdT =  -1j*Disp_int + 1j*self.gamma*self.L/self.Tr*np.abs(A)**2*A - (self.kappa/2+1j*dOm_curr)*A + np.sqrt(self.kappa/2/self.Tr)*self.Pump*Pump_P**.5
                else:
                    # with raman
                    dAAdt = deriv_1(self.TimeStep,abs(A)**2)
                    dAdT =  -1j*Disp_int + 1j*self.gamma*self.L/self.Tr*np.abs(A)**2*A - (self.kappa/2+1j*dOm_curr)*A -1j*self.gamma*self.Traman*dAAdt*A + np.sqrt(self.kappa/2/self.Tr)*self.Pump*Pump_P**.5
            return dAdT
        
        r = complex_ode(LLE_1d).set_integrator('dopri5', atol=abtol, rtol=reltol,nsteps=nmax)# set the solver
        r.set_initial_value(self.seed, 0)# seed the cavity
        
        
        img = mpimg.imread('phase_space.png')
        xx = np.linspace(-1,5,np.size(img,axis=1))
        yy = np.linspace(11,0,np.size(img,axis=0))
        XX,YY = np.meshgrid(xx,yy)
        
        
        fig = plt.figure(figsize=(11,7))        
        plt.subplots_adjust(top=0.95,bottom=0.1,left=0.06,right=0.986,hspace=0.2,wspace=0.16)

        ax1 = plt.subplot(221)
        ax1.pcolormesh(XX,YY,img[:,:,1])
        plt.xlabel('Detuning')
        plt.ylabel('f^2')
        plt.title('Choose the region')
        plt.xlim(min(xx),max(xx))
        dot = plt.plot(detuning_0, Pump_P,'rx')
        
        
        ax2 = plt.subplot(222)
        line, = plt.plot(abs(self.seed)**2)
        plt.ylim(0,1.1)
        plt.ylabel('$|\Psi|^2$')
        
        ax3 = plt.subplot(224)
        line2, = plt.semilogy(self.mu, np.abs(np.fft.fft(self.seed))**2)
        plt.ylabel('PSD')
        plt.xlabel('mode number')
        ### widjets
        axcolor = 'lightgoldenrodyellow'

        resetax = plt.axes([0.4, 0.025, 0.1, 0.04])
        button = Button(resetax, 'Stop', color=axcolor, hovercolor='0.975')
        
        axboxf = plt.axes([0.1, 0.35, 0.1, 0.075])
        text_box_f = TextBox(axboxf, 'f^2', initial=str(Pump_P))
        
        axboxd = plt.axes([0.1, 0.25, 0.1, 0.075])
        text_box_d = TextBox(axboxd, 'Detuning', initial=str(detuning_0))
        
        Run = True
        def setup(event): 
            global Run
            Run = False   
        button.on_clicked(setup)
        
        def onclick(event): 
            if event.inaxes == ax1:
                ix, iy = event.xdata, event.ydata     
                text_box_d.set_val(np.round(ix,4))
                text_box_f.set_val(np.round(iy,4))
                ax1.plot([ix],[iy],'rx')
   

        fig.canvas.mpl_connect('button_press_event', onclick)
        
        while Run:
            dOm_curr = float(text_box_d.text) # get the detuning value
            Pump_P = float(text_box_f.text)
            Field = r.integrate(r.t+T_step)
            F_mod_sq = np.abs(Field)**2
            F_sp = np.abs(np.fft.fft(Field))**2
            line.set_ydata(F_mod_sq)
            line2.set_ydata(F_sp)
            ax2.set_ylim(0, max(F_mod_sq))
            ax3.set_ylim(min(F_sp),max(F_sp))
            plt.pause(1e-10)
    
        
    def Jacobian(self,zeta_0,A,D1):
        
        N = self.N_points
        d2 = self.D2/self.kappa
        dphi = abs(self.phi[1]-self.phi[0])
        index_1 = np.arange(0,N)
        index_2 = np.arange(N,2*N)
        Jacob = np.zeros([2*N+1,2*N+1],dtype=complex)
        
        Jacob[:-1,:-1] += self.LinMatrix(zeta_0,dense=False)
        Jacob[index_1,index_1] += + 2*1j*abs(A[index_1])**2 
        Jacob[index_2,index_2] +=  - 2*1j*abs(A[index_1])**2 
        
        Jacob[index_1,index_2] += 1j*A[index_1]*A[index_1]
        Jacob[index_2,index_1] += -1j*np.conj(A[index_1])*np.conj(A[index_1])
        
        Jacob[index_1,-1] = -(self.D1A(A[index_1]))
        Jacob[index_2,-1] = np.conj(Jacob[index_1,-1])
        
        Jacob[-1,index_1] = np.real((self.FirstDmat[np.argmax(np.real(A[index_1])),:]))
        Jacob[-1,index_2] = np.real((self.FirstDmat[np.argmax(np.real(A[index_1])),:]))
        
        
        #Jacob_sparse = dia_matrix(Jacob)
        #return Jacob_sparse
        return Jacob
    
    def JacobianForLinAnalysis(self,zeta_0,A):
        
        N = self.N_points
        d2 = self.D2/self.kappa
        dphi = abs(self.phi[1]-self.phi[0])
        index_1 = np.arange(0,N)
        index_2 = np.arange(N,2*N)
        Jacob = np.zeros([2*N,2*N],dtype=complex)
        
        Jacob[:,:] += self.LinMatrix(zeta_0,dense=False)
        Jacob[index_1,index_1] += + 2*1j*abs(A[index_1])**2 
        Jacob[index_2,index_2] +=  - 2*1j*abs(A[index_1])**2 
        
        Jacob[index_1,index_2] += 1j*A[index_1]*A[index_1]
        Jacob[index_2,index_1] += -1j*np.conj(A[index_1])*np.conj(A[index_1])
        
        #Jacob[index_1,-1] = -(self.D1A(A[index_1]))
        #Jacob[index_2,-1] = np.conj(Jacob[index_1,-1])
        
        #Jacob[-1,index_1] = np.real((self.FirstDmat[np.argmax(np.real(A[index_1])),:]))
        #Jacob[-1,index_2] = np.real((self.FirstDmat[np.argmax(np.real(A[index_1])),:]))
        
        
        #Jacob_sparse = dia_matrix(Jacob)
        #return Jacob_sparse
        return Jacob
    
    def FirstDerivativeMatrix(self):
        D = np.zeros([self.N_points,self.N_points],dtype=complex)
        index = np.arange(0,self.N_points)
        D_fourier = np.zeros([self.N_points,self.N_points],dtype=complex)
        D_fourier[index,index] = 1j*self.mu
            
        Fourier_matrix = dft(self.N_points)
        D = np.dot(np.dot(Fourier_matrix,D_fourier),np.conj(Fourier_matrix.T)/self.N_points)
        
        return D
        
        
    def D1A(self,A):
        D = np.zeros([self.N_points,self.N_points],dtype=complex)
        index = np.arange(0,self.N_points)
        D_fourier = np.zeros([self.N_points,self.N_points],dtype=complex)
        A_spectrum = np.fft.fft(A)
        D_fourier[index,index] = 1j*self.mu
        #A_spectrum = np.dot(D_fourier,A_spectrum)
        Fourier_matrix = dft(self.N_points)
        D = np.dot(np.dot(Fourier_matrix,D_fourier),np.conj(Fourier_matrix.T)/self.N_points)
        res = np.dot(D,A)
        #res = np.dot(-1j*self.mu,A_spectrum)#/self.N_points
        #res = np.fft.ifft(A_spectrum)
        
        return res
    
    def DispersionMatrix(self,D1=0,order=0):
        D = np.zeros([self.N_points,self.N_points],dtype=complex)
        index = np.arange(0,self.N_points)
        d2 = self.D2/self.kappa
        #dphi = abs(self.phi[1]-self.phi[0])
        
        if order==0:
            D_fourier = np.zeros([self.N_points,self.N_points],dtype=complex)
            D_fourier[index,index] = -1j*(self.Dint+D1*self.mu)*2/self.kappa
            
            Fourier_matrix = dft(self.N_points)
            D = np.dot(np.dot(Fourier_matrix,D_fourier),np.conj(Fourier_matrix.T)/self.N_points)
        
        if order == 2:
            D[index[:-1],index[1:]] = 1j*d2/dphi**2
            D[0,self.N_points-1] =  1j*d2/dphi**2
            D += D.T
            D[index,index]= -2*1j*d2/dphi**2
        if order == 4:
            D[index[:-2],index[2:]] = -1/12*1j*d2/dphi**2
            
            
            D[index[:-1],index[1:]] = 4/3*1j*d2/dphi**2
            
            
            D += D.T
            
            D[0,self.N_points-2] =  -1/12*1j*d2/dphi**2 
            D[self.N_points-1,1] =  -1/12*1j*d2/dphi**2
            
            D[0,self.N_points-1] =  4/3*1j*d2/dphi**2
            D[self.N_points-1,0] =  4/3*1j*d2/dphi**2
            
            D[1,self.N_points-1] =  -1/12*1j*d2/dphi**2 
            D[self.N_points-2,0] =  -1/12*1j*d2/dphi**2
            
            D[index,index]= -5/2*1j*d2/dphi**2
            
        if order == 6:
            D[index[:-3],index[3:]] = 1./90*1j*d2/dphi**2
            
            
            D[index[:-2],index[2:]] = -3./20*1j*d2/dphi**2
            
            
            D[index[:-1],index[1:]] = 3./2*1j*d2/dphi**2
            
            
            D += D.T
            
            D[0,self.N_points-3] =  1./90*1j*d2/dphi**2
            D[self.N_points-1,2] =  1./90*1j*d2/dphi**2
            
            D[0,self.N_points-2] = -3./20*1j*d2/dphi**2
            D[self.N_points-1,1] =  -3./20*1j*d2/dphi**2
            
            D[0,self.N_points-1] =   3./2*1j*d2/dphi**2
            D[self.N_points-1,0] =   3./2*1j*d2/dphi**2
            
            D[1,self.N_points-2] =  1/90*1j*d2/dphi**2 
            D[self.N_points-2,1] =  1/90*1j*d2/dphi**2
            
            D[1,self.N_points-1] = -3./20*1j*d2/dphi**2
            D[self.N_points-2,0] =  -3./20*1j*d2/dphi**2
            
            D[2,self.N_points-1] =  1/90*1j*d2/dphi**2 
            D[self.N_points-3,0] =  1/90*1j*d2/dphi**2
            
            D[index,index]= -49./18*1j*d2/dphi**2
            
        return D
    
    def LinMatrix(self,zeta_0,dense=True):
        self.FirstDmat=self.FirstDerivativeMatrix()
        D = np.zeros([2*self.N_points,2*self.N_points],dtype=complex)
        d2 = self.D2/self.kappa
        dphi = abs(self.phi[1]-self.phi[0])
        index_1 = np.arange(0,self.N_points)
        index_2 = np.arange(self.N_points,2*self.N_points)
      
        D[:self.N_points,:self.N_points] = self.D
        D[self.N_points:,self.N_points:] = np.conj(D[:self.N_points,:self.N_points])
        D[index_1,index_1]+=-(1+ 1j*zeta_0)
        D[index_2,index_2]+=-(1- 1j*zeta_0)
        
        if dense==True:
            D_sparse = dia_matrix(D)
            return D_sparse
        else:
            return D
        
    
    def NewtonRaphson(self,A_input,dOm, Pump,D1=0,HardSeed = True, tol=1e-5,max_iter=50):
        self.D = self.DispersionMatrix(D1=D1,order=0)
        FirstDerivativeMatrix=self.FirstDerivativeMatrix()
        A_guess = np.fft.ifft(A_input)
        
        d2 = self.D2/self.kappa
        zeta_0 = dOm*2/self.kappa
        dphi = abs(self.phi[1]-self.phi[0])
        pump = Pump*np.sqrt(1./(hbar*self.w0))
        
        Aprev = np.zeros(2*self.N_points+1,dtype=complex)
        
        
        f0 = pump*np.sqrt(8*self.g0*self.kappa_ex/self.kappa**3)
        
        
        index_1 = np.arange(0,self.N_points)
        index_2 = np.arange(self.N_points,2*self.N_points)
        
        f0_direct = np.zeros(Aprev.size-1,dtype=complex)
        f0_direct[index_1] = np.fft.ifft(f0)*self.N_points
        
        f0_direct[index_2] = np.conj(f0_direct[index_1])
        
       
        if HardSeed == False:
            A_guess = A_guess+ f0_direct/(1+1j*zeta_0)
            Aprev[:self.N_points] = A_guess
        else:
            Aprev[:self.N_points] = A_guess*np.sqrt(2*self.g0/self.kappa)
        
        Aprev[index_2] = np.conj(Aprev[:self.N_points])
        Aprev[-1] = D1*2/self.kappa
        
        Ak = np.zeros(Aprev.size,dtype=complex)
        
        

        buf= np.zeros(Aprev.size,dtype=complex)
        buf_prev= np.zeros(Aprev.size,dtype=complex)
        
        M_lin0 = self.LinMatrix(zeta_0)
        
        D1_res=D1*2/self.kappa
       
        print('f0^2 = ' + str(np.round(max(abs(f0_direct)**2), 2)))
        print('xi = ' + str(zeta_0) )
        
        diff = self.N_points
        counter =0
        diff_array=[]
        
        while diff>tol:
            
            
            self.D = self.DispersionMatrix(D1=self.kappa/2*D1_res,order=0)
            J = self.Jacobian(zeta_0, Aprev[index_1],D1=D1_res*self.kappa/2)
            buf[index_1] =  1j*abs(Aprev[index_1])**2*Aprev[index_1]         
            buf[index_2] = np.conj(buf[index_1])
            #buf[index_2] =  -1j*abs(Aprev[index_2])**2*Aprev[index_2]      
            #buf0= buf+  M_lin0.dot(Aprev)+ f0_direct
            buf[:-1] += (self.LinMatrix(zeta_0)).dot(Aprev[:-1]) + f0_direct
            buf[-1]=np.real(self.D1A(np.real(Aprev[index_1]))[np.argmax(np.real(Aprev[index_1]))])
            
            
            Ak = Aprev - np.linalg.solve(J,buf)
            
            D1_res= np.real(Ak[-1])
            
            
            diff = np.sqrt(abs((Ak-Aprev).dot(np.conj(Ak-Aprev))/(Ak.dot(np.conj(Ak)))))
            #print(diff, abs((Ak[-1]-Aprev[-1])/D1_res))
            diff_array += [diff]
            Aprev[:] = Ak[:]
            buf_prev[:]=buf[:]
            Aprev[index_2] = np.conj(Aprev[index_1])
            counter +=1
            
            #plt.scatter(counter,diff,c='k')
            if counter>max_iter:
                print("Did not coverge in " + str(max_iter)+ " iterations, relative error is " + str(diff))
                res = np.zeros(self.N_points,dtype=complex)
                res = Ak[index_1]
                v = self.kappa/2*D1_res
                return np.fft.fft(res)/np.sqrt(2*self.g0/self.kappa), v, diff_array
                break
        print("Converged in " + str(counter) + " iterations, relative error is " + str(diff))
        res = np.zeros(self.N_points,dtype=complex)
        res = Ak[index_1]
        v = self.kappa/2*D1_res
        print('D1_res ', D1_res)
        return np.fft.fft(res)/np.sqrt(2*self.g0/self.kappa), v,diff_array
    
    def NewtonRaphsonFixedD1(self,A_input,dOm, Pump,HardSeed = True, tol=1e-5,max_iter=50):
        self.D = self.DispersionMatrix(D1=0,order=0)
        
        A_guess = np.fft.ifft(A_input)
        
        d2 = self.D2/self.kappa
        zeta_0 = dOm*2/self.kappa
        dphi = abs(self.phi[1]-self.phi[0])
        pump = Pump*np.sqrt(1./(hbar*self.w0))
        
        Aprev = np.zeros(2*self.N_points,dtype=complex)
        
        
        f0 = pump*np.sqrt(8*self.g0*self.kappa_ex/self.kappa**3)
        
        
        index_1 = np.arange(0,self.N_points)
        index_2 = np.arange(self.N_points,2*self.N_points)
        
        f0_direct = np.zeros(Aprev.size,dtype=complex)
        f0_direct[index_1] = np.fft.ifft(f0)*self.N_points
        
        f0_direct[index_2] = np.conj(f0_direct[index_1])
        
       
        if HardSeed == False:
            A_guess = A_guess+ f0_direct/(1+1j*zeta_0)
            Aprev[:self.N_points] = A_guess
        else:
            Aprev[:self.N_points] = A_guess*np.sqrt(2*self.g0/self.kappa)
        
        Aprev[index_2] = np.conj(Aprev[:self.N_points])
        
        
        Ak = np.zeros(Aprev.size,dtype=complex)
        
        

        buf= np.zeros(Aprev.size,dtype=complex)
        buf_prev= np.zeros(Aprev.size,dtype=complex)
        
        M_lin0 = self.LinMatrix(zeta_0)
        
       
       
        print('f0^2 = ' + str(np.round(max(abs(f0_direct)**2), 2)))
        print('xi = ' + str(zeta_0) )
        
        diff = self.N_points
        counter =0
        diff_array=[]
        
        while diff>tol:
            
            
            #self.D = self.DispersionMatrix(D1=self.kappa/2*D1_res,order=0)
            J = self.JacobianForLinAnalysis(zeta_0, Aprev[index_1])
            buf[index_1] =  1j*abs(Aprev[index_1])**2*Aprev[index_1]         
            buf[index_2] = np.conj(buf[index_1])
            #buf[index_2] =  -1j*abs(Aprev[index_2])**2*Aprev[index_2]      
            #buf0= buf+  M_lin0.dot(Aprev)+ f0_direct
            buf[:] += (self.LinMatrix(zeta_0)).dot(Aprev[:]) + f0_direct
            
            
            
            Ak = Aprev - np.linalg.solve(J,buf)
            
            
            
            
            diff = np.sqrt(abs((Ak-Aprev).dot(np.conj(Ak-Aprev))/(Ak.dot(np.conj(Ak)))))
            #print(diff, abs((Ak[-1]-Aprev[-1])/D1_res))
            diff_array += [diff]
            Aprev[:] = Ak[:]
            buf_prev[:]=buf[:]
            Aprev[index_2] = np.conj(Aprev[index_1])
            counter +=1
            
            #plt.scatter(counter,diff,c='k')
            if counter>max_iter:
                print("Did not coverge in " + str(max_iter)+ " iterations, relative error is " + str(diff))
                res = np.zeros(self.N_points,dtype=complex)
                res = Ak[index_1]
            
                return np.fft.fft(res)/np.sqrt(2*self.g0/self.kappa), diff_array
                break
        print("Converged in " + str(counter) + " iterations, relative error is " + str(diff))
        res = np.zeros(self.N_points,dtype=complex)
        res = Ak[index_1]
        
        
        return np.fft.fft(res)/np.sqrt(2*self.g0/self.kappa),diff_array

        
    def LinearStability(self,solution,dOm,v=0,plot_eigvals=True):
        self.D = self.DispersionMatrix(D1=v,order=0)
        A=np.fft.ifft(solution)
        
        d2 = self.D2/self.kappa
        d1 = v*2/self.kappa
        zeta_0 = dOm*2/self.kappa
        dphi = abs(self.phi[1]-self.phi[0])
        field = np.zeros_like(A)
        field = A*np.sqrt(2*self.g0/self.kappa)
        
        
        index_1 = np.arange(0,self.N_points)
        index_2 = np.arange(self.N_points,2*self.N_points)
        
       
        #Full_Matrix=self.Jacobian(zeta_0,field,D1=d1).todense()
        #Full_Matrix=self.Jacobian(zeta_0,field,D1=d1)
        Full_Matrix=self.JacobianForLinAnalysis(zeta_0,field)
        
        
        eig_vals,eig_vec = np.linalg.eig(Full_Matrix)
        
        eigen_vectors = np.zeros([self.N_points,2*self.N_points],dtype=complex)
        if plot_eigvals==True:
            plt.scatter(np.real(eig_vals),np.imag(eig_vals))
            plt.xlabel('Real part')
            plt.ylabel('Imaginary part')
            
        for jj in range(2*self.N_points):
            eigen_vectors[:,jj]=(eig_vec[:self.N_points,jj]).T
            eigen_vectors[:,jj]=np.fft.fft(eigen_vectors[:,jj])
        
        return eig_vals[:-1]*self.kappa/2, eigen_vectors/np.sqrt(2*self.g0/self.kappa)
    def printProgressBar (self, iteration, total, prefix = '', suffix = '', time = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
        """
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            length      - Optional  : character length of bar (Int)
            fill        - Optional  : bar fill character (Str)
            printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
        """
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print('\r%s |%s| %s%% %s %s' % (prefix, bar, percent, suffix, time), end = printEnd)
        # Print New Line on Complete
        if iteration == total: 
                print()