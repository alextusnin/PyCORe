#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is the chains of resonator class for PyCORe
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

from Resonator import Resonator

class_path = os.path.abspath(__file__)

# Get the directory containing the script
PyCORe_directory = os.path.dirname(class_path)

class CROW(Resonator):#all idenical resonators

        def __init__(self):
        #Physical parameters initialization
            self.n0 = 0
            self.n2 = 0
            self.FSR = 0
            self.w0 = 0
            self.width = 0
            self.height = 0
            self.kappa_0 = 0
            self.Dint = np.array([0])
            
            self.n2t=0
            self.t_th=0
            
            self.Tr = 0
            self.Aeff = 0
            self.Leff = 0
            self.Veff = 0
            self.g0 = 0
            self.gamma = 0
            self.J = np.array([0])
            
            
                
            self.Bus_J = np.array([0])
            self.Bus_Phase = np.array([0])
            self.Snake_coupling=False       
            
            self.Delta = np.array([0])
            self.N_CROW = 0
            self.Delta_D1 = np.zeros(self.N_CROW)
            self.D2 = np.zeros(self.N_CROW)
            self.D3 = np.zeros(self.N_CROW)
            self.kappa_ex =np.array([0])
            self.kappa = self.kappa_0 + self.kappa_ex
            self.N_points = 0
            self.mu = np.array([0])
            self.phi = np.array([0])
            
            self.D2 = np.array([0])
            self.D3 = np.array([0])
            self.D = np.array([0])
            self.M_lin = np.array([0])

            
            
            #self.M_lin = np.array([0])

        def Init_From_Dict(self, resonator_parameters):
        #Physical parameters initialization
            self.n0 = resonator_parameters['n0']
            self.n2 = resonator_parameters['n2']
            self.FSR = resonator_parameters['FSR']
            self.w0 = resonator_parameters['w0']
            self.width = resonator_parameters['width']
            self.height = resonator_parameters['height']
            self.kappa_0 = resonator_parameters['kappa_0']
            self.Dint = resonator_parameters['Dint']
            
            
            self.Tr = 1/self.FSR #round trip time
            self.Aeff = self.width*self.height 
            self.Leff = c/self.n0*self.Tr 
            self.Veff = self.Aeff*self.Leff 
            self.g0 = hbar*self.w0**2*c*self.n2/self.n0**2/self.Veff
            self.gamma = self.n2*self.w0/c/self.Aeff
            self.J = np.array(resonator_parameters['Inter-resonator_coupling'])
            
            #try:
            if 'Snake bus coupling' in resonator_parameters.keys():
                self.Bus_J = np.array(resonator_parameters['Snake bus coupling'])
                self.Bus_Phase = np.array(resonator_parameters['Snake bus phases'])
                self.Snake_coupling= True
            else:
                self.Bus_J = np.array([[0],[0]])
                self.Bus_Phase = np.array([0])
                self.Snake_coupling=False
            if 'T thermal' in resonator_parameters.keys():
                self.n2t = resonator_parameters['n2 thermal']
                self.t_th=resonator_parameters['T thermal']
            
            self.Delta = np.array(resonator_parameters['Resonator detunings'])
            self.N_CROW = len(self.Dint[0,:])
            self.D2 = np.zeros(self.N_CROW)
            self.D3 = np.zeros(self.N_CROW)
            self.kappa_ex = resonator_parameters['kappa_ex']# V
            self.kappa = self.kappa_0 + self.kappa_ex
            self.N_points = len(self.Dint[:,0])
            self.mu = np.fft.fftshift(np.arange(-self.N_points/2, self.N_points/2))
            
            self.phi = np.linspace(0,2*np.pi,self.N_points)
            if 'Delta D1' in resonator_parameters.keys():
                self.Delta_D1 = resonator_parameters['Delta D1']
            else:
                self.Delta_D1 = np.zeros(self.N_CROW)
            def func(x, a, b, c, d):
                    return a + x*b + c*x**2/2 + d*x**3/6
            for ii in range(0,self.N_CROW):
                self.Dint[:,ii] = np.fft.ifftshift(self.Dint[:,ii])
                
                popt, pcov = curve_fit(func, self.mu, self.Dint[:,ii])
                self.D2[ii] = popt[2]
                self.D3[ii] = popt[3]
            
            ind_phase_modes = np.arange(0,(self.N_CROW-1)*self.N_points)
            ind_phase_modes = ind_phase_modes%self.N_points
            M_lin = diags(-(self.kappa.T.reshape(self.kappa.size)/self.kappa_0+1j*self.Dint.T.reshape(self.Dint.size)*2/self.kappa_0),0) + 1j*diags(self.J[:,:self.N_CROW-1].T.reshape(self.J[:,:self.N_CROW-1].size)*2/self.kappa_0 *np.exp(-1j*ind_phase_modes*np.pi),self.N_points) + 1j*diags(self.J[:,:self.N_CROW-1].T.reshape(self.J[:,:self.N_CROW-1].size)*2/self.kappa_0 *np.exp(1j*ind_phase_modes*np.pi),-self.N_points)
            if self.J[0,:].size == self.N_CROW:
                M_lin+= 1j*diags(self.J[:,self.N_CROW-1].T.reshape(self.J[:,self.N_CROW-1].size)*2/self.kappa_0 *np.exp(-1j*ind_phase_modes[:self.N_points]*np.pi),(self.N_CROW-1)*self.N_points)
                M_lin+= 1j*diags(self.J[:,self.N_CROW-1].T.reshape(self.J[:,self.N_CROW-1].size)*2/self.kappa_0 *np.exp(1j*ind_phase_modes[:self.N_points]*np.pi),-(self.N_CROW-1)*self.N_points)
            
            #self.M_lin = M_lin
            #self.M_lin = M_lin.todense()

            
           
            
        def seed_level (self, pump, detuning):
            
            f_norm = pump*np.sqrt(1./(hbar*self.w0))*np.sqrt(8*self.g0*self.kappa_ex/self.kappa_0**3)#we pump the first ring
            detuning_norm  = detuning*2/self.kappa_0
            
            #redo
            LinearM = np.eye(self.N_points*self.N_CROW,dtype = complex)
            

            ind_modes = np.arange(self.N_points)
            
            for ii in range(0,self.N_CROW-1):
                LinearM[ind_modes+ii*self.N_points,ind_modes+(ii+1)*self.N_points] = 1j*self.J.T.reshape(self.J.size)[ii*self.N_points +ind_modes]*2/self.kappa_0
            LinearM += LinearM.T
            
            
            indM = np.arange(self.N_points*self.N_CROW)

            LinearM[indM,indM] = -(self.kappa.T.reshape(self.kappa.size)[indM]/self.kappa_0 +1j*self.Delta.T.reshape(self.Delta.size)[indM]/self.kappa_0+ 1j*detuning_norm)
            
            
            
            res_seed = np.zeros_like(f_norm.reshape(f_norm.size))
            res_seed = np.linalg.solve(LinearM,f_norm.T.reshape(f_norm.size))
            res_seed*= 1/np.sqrt(2*self.g0/self.kappa_0)
            #res_seed.reshape((self.N_points,self.N_CROW))
            
            return res_seed
        def noise(self, a):
#        return a*np.exp(1j*np.random.uniform(-1,1,self.N_points)*np.pi)
            return a*(np.random.uniform(-1,1,self.N_points*self.N_CROW)+ 1j*np.random.uniform(-1,1,self.N_points*self.N_CROW))
        
        def Linear_analysis(self,plot_dint=True,plot_evec=True):
            M = np.zeros((self.N_CROW,self.N_CROW),dtype='complex')
            ev_arr = np.array([],dtype='complex')
            for ii in range(self.N_points):
                for jj in range(self.N_CROW):
                    M[jj,jj] = 1*self.Dint[ii,jj]+self.mu[ii]*self.Delta_D1[jj] + self.Delta[ii,jj]
                    if jj<self.N_CROW-1:
                        M[jj,jj+1] = self.J[0,jj]
                        M[jj+1,jj] = self.J[0,jj]
                        
                    if self.J[0,:].size==self.N_CROW:
                        M[0,self.N_CROW-1] = self.J[0,self.N_CROW-1]
                        M[self.N_CROW-1,0] = self.J[0,self.N_CROW-1]
                    ev,a = eig(M)
                if self.mu[ii]==0:
                    evec_r = np.real(a.reshape(self.N_CROW**2))
                ev_arr = np.append(ev_arr,ev.T)
            if plot_dint:
                plt.figure()
                for kk in range(self.N_CROW):
                    plt.plot(self.mu,np.real(ev_arr[kk::self.N_CROW]),'k.')
                    plt.xlim(self.mu.min(),self.mu.max())
                    plt.xlabel('Mode number')
                    plt.ylabel('Hybridized D$_{int}$')
                    plt.grid('on')
            if plot_evec:
                fig, ax = plt.subplots()
                patches = []
                for ii in range(self.N_CROW):
                    for jj in range(self.N_CROW):
                        wedge = Wedge((ii*1., jj*1.), .47, 0, 360, width=0.1)
                        patches.append(wedge)
                colors = evec_r
                p = PatchCollection(patches, cmap=cm.seismic,alpha=1)
                p.set_array(np.array(colors))
                ax.add_collection(p)
                fig.colorbar(p, ax=ax)
#                plt.title('J$_0$='+str(self.J/2/np.pi/1e9)+' GHz'+'; Factor='+str(fact)+'; N CROW=' +str(N_crow))
                plt.ylim(-0.5,self.N_CROW*1.-0.5)
                plt.xlim(-0.5,self.N_CROW*1.-0.5)
            return ev_arr
        
        def Propagate_SplitStep(self, simulation_parameters, Pump, Seed=[0], dt=1e-4):
            start_time = time.time()
            T = simulation_parameters['slow_time']
            out_param = simulation_parameters['output']
            detuning = simulation_parameters['detuning_array']
            eps = simulation_parameters['noise_level']
            #dt = simulation_parameters['time_step']#in photon lifetimes
            
            pump = Pump*np.sqrt(1./(hbar*self.w0))
            if Seed[0] == 0:
                seed = self.seed_level(Pump, detuning[0])*np.sqrt(2*self.g0/self.kappa_0)
            else:
                seed = Seed*np.sqrt(2*self.g0/self.kappa_0)
            ### renormalization
            T_rn = (self.kappa_0/2)*T
            f0 = pump*np.sqrt(8*self.g0*np.max(self.kappa_ex)/self.kappa_0**3)
            
            print('f0^2 = ' + str(np.round(np.max(abs(f0)**2), 2)))
            print('xi [' + str(detuning[0]*2/self.kappa_0) + ',' +str(detuning[-1]*2/self.kappa_0)+ '] (normalized on ' r'$kappa_0/2)$')
            noise_const = self.noise(eps) # set the noise level
            nn = len(detuning)
            
            t_st = float(T_rn)/len(detuning)
            #dt=1e-4 #t_ph
            
            sol = np.ndarray(shape=(len(detuning), self.N_points, self.N_CROW), dtype='complex') # define an array to store the data
            
            ind_modes = np.arange(self.N_points)
            ind_res = np.arange(self.N_CROW)
            for ii in range(self.N_CROW):
                sol[0,ind_modes,ii] = seed[ii*self.N_points+ind_modes]
           
            self.printProgressBar(0, nn, prefix = 'Progress:', suffix = 'Complete', length = 50)
            f0 = np.fft.ifft(f0,axis=0)*self.N_points
            for it in range(1,len(detuning)):
                noise_const = self.noise(eps)
                sol[it-1,:,:] += noise_const.reshape((self.N_points,self.N_CROW))
                self.printProgressBar(it + 1, nn, prefix = 'Progress:', suffix = 'Complete,', time='elapsed time = ' + '{:04.1f}'.format(time.time() - start_time) + ' s', length = 50)
                dOm_curr = detuning[it] # detuning value
                t=0
                buf  =  sol[it-1,:,:]
                
                
                buf = np.fft.ifft(buf,axis=0)*self.N_points
               
                while t<t_st:
                    for ii in range(self.N_CROW):
                        #First step
                        buf[:,ii] = np.fft.fft(np.exp(dt*(1j*abs(buf[:,ii])**2 +f0[:,ii]/buf[:,ii]))*buf[:,ii])
                        #second step
                    
                    #buf_vec = np.dot( expm(dt*(self.M_lin -1j*dOm_curr*2/self.kappa_0 *np.eye(self.M_lin[:,0].size))),buf.T.reshape(buf.size) )
                    
                    
                    #buf_vec = expm(csc_matrix(dt*(self.M_lin -1j*dOm_curr*2/self.kappa_0* eye(self.N_points*self.N_CROW) ))).dot(buf.T.reshape(buf.size))
                    #buf_vec = expm((dt*(self.M_lin -1j*dOm_curr*2/self.kappa_0* eye(self.N_points*self.N_CROW) )).todense()).dot(buf.T.reshape(buf.size))
                    #buf_vec = expm(dt*(self.M_lin -1j*dOm_curr*2/self.kappa_0* eye(self.N_points*self.N_CROW) )).dot(buf.T.reshape(buf.size))
                  
                    for ii in range(self.N_CROW):
                        buf[ind_modes,ii] = np.fft.ifft(buf_vec[ii*self.N_points+ind_modes])
                    
                    t+=dt
                sol[it,:,:] = np.fft.fft(buf,axis=0)/len(buf)
                #sol[it,:] = buf
                
            if out_param == 'map':
                return sol/np.sqrt(2*self.g0/self.kappa_0)
            elif out_param == 'fin_res':
                return sol[-1, :]/np.sqrt(2*self.g0/self.kappa_0)
            else:
                print ('wrong parameter')
            
        def Propagate_SAM(self, simulation_parameters, Pump, Seed=[0]):
            start_time = time.time()
            
            T = simulation_parameters['slow_time']
            abtol = simulation_parameters['absolute_tolerance']
            reltol = simulation_parameters['relative_tolerance']
            out_param = simulation_parameters['output']
            nmax = simulation_parameters['max_internal_steps']
            detuning = simulation_parameters['detuning_array']
            eps = simulation_parameters['noise_level']
            #dt = simulation_parameters['time_step']#in photon lifetimes
            
            pump = Pump*np.sqrt(1./(hbar*self.w0))
            if Seed[0,0] == 0:
                seed = self.seed_level(Pump, detuning[0])*np.sqrt(2*self.g0/self.kappa_0)
            else:
                seed = Seed.T.reshape(Seed.size)*np.sqrt(2*self.g0/self.kappa_0)
            ### renormalization
            T_rn = (self.kappa_0/2)*T
            f0 = pump*np.sqrt(8*self.g0*np.max(self.kappa_ex)/self.kappa_0**3)
            
            print('f0^2 = ' + str(np.round(np.max(abs(f0)**2), 2)))
            print('xi [' + str(detuning[0]*2/self.kappa_0) + ',' +str(detuning[-1]*2/self.kappa_0)+ '] (normalized on ' r'$kappa_0/2)$')
            noise_const = self.noise(eps) # set the noise level
            nn = len(detuning)
            
            t_st = float(T_rn)/len(detuning)
            #dt=1e-4 #t_ph
            
            sol = np.ndarray(shape=(len(detuning), self.N_points, self.N_CROW), dtype='complex') # define an array to store the data
            ind_modes = np.arange(self.N_points)
            ind_res = np.arange(self.N_CROW)
            for ii in range(self.N_CROW):
                sol[0,ind_modes,ii] = seed[ii*self.N_points+ind_modes]
           
            self.printProgressBar(0, nn, prefix = 'Progress:', suffix = 'Complete', length = 50)
            
            #def RHS(Time, A):
            #    A = A - noise_const#self.noise(eps)
            #    A_dir = np.zeros(A.size,dtype=complex)
              
            #    for ii in range(self.N_CROW):
            #        A_dir[ii*self.N_points+ind_modes] = np.fft.ifft(A[ii*self.N_points+ind_modes])## in the direct space
            #    A_dir*=self.N_points
            #    dAdT =  (self.M_lin -1j*dOm_curr*2/self.kappa_0* np.eye(self.N_points*self.N_CROW)).dot(A) + f0.reshape(f0.size) 
            #    for ii in range(self.N_CROW):
            #        dAdT[0,ii*self.N_points+ind_modes]+=1j*np.fft.fft(A_dir[ii*self.N_points+ind_modes]*np.abs(A_dir[ii*self.N_points+ind_modes])**2)/self.N_points
            #    return dAdT
            def RHS(Time, A):
                A = A - noise_const#self.noise(eps)
                A_dir = np.zeros(A.size,dtype=complex)
                dAdT = np.zeros(A.size,dtype=complex)
              
                for ii in range(self.N_CROW):
                    A_dir[ii*self.N_points+ind_modes] = np.fft.ifft(A[ii*self.N_points+ind_modes])## in the direct space
                A_dir*=self.N_points
                dAdT =  (-self.kappa.T.reshape(self.kappa.size)/2-1j*self.Dint.T.reshape(self.Dint.size) -1j*dOm_curr)*A*2/self.kappa_0 + f0.reshape(f0.size) 
                dAdT[0*self.N_points+ind_modes] += 1j*self.J[:,0]*2/self.kappa_0 *np.exp(-1j*self.mu*np.pi)*A[1*self.N_points+ind_modes]+1j*np.fft.fft(A_dir[0*self.N_points+ind_modes]*np.abs(A_dir[0*self.N_points+ind_modes])**2)/self.N_points
                dAdT[(self.N_CROW-1)*self.N_points+ind_modes] += 1j*self.J[:,self.N_CROW-2]*2/self.kappa_0 *np.exp(1j*self.mu*np.pi)*A[((self.N_CROW-2))*self.N_points+ind_modes]+1j*np.fft.fft(A_dir[(self.N_CROW-1)*self.N_points+ind_modes]*np.abs(A_dir[(self.N_CROW-1)*self.N_points+ind_modes])**2)/self.N_points
                for ii in range(1,self.N_CROW-1):
                    dAdT[ii*self.N_points+ind_modes]+= 1j*self.J[:,ii]*2/self.kappa_0 *np.exp(-1j*self.mu*np.pi)*A[(ii+1)*self.N_points+ind_modes] + 1j*self.J[:,ii-1]*2/self.kappa_0 *np.exp(1j*self.mu*np.pi)*A[(ii-1)*self.N_points+ind_modes] +  1j*np.fft.fft(A_dir[ii*self.N_points+ind_modes]*np.abs(A_dir[ii*self.N_points+ind_modes])**2)/self.N_points
                return dAdT
            r = complex_ode(RHS).set_integrator('dop853', atol=abtol, rtol=reltol,nsteps=nmax)# set the solver
            #r = ode(RHS).set_integrator('zvode', atol=abtol, rtol=reltol,nsteps=nmax)# set the solver
            
            r.set_initial_value(seed, 0)# seed the cavity
            
            for it in range(1,len(detuning)):
                self.printProgressBar(it + 1, nn, prefix = 'Progress:', suffix = 'Complete,', time='elapsed time = ' + '{:04.1f}'.format(time.time() - start_time) + ' s', length = 50)
                dOm_curr = detuning[it] # detuning value
                res = r.integrate(r.t+t_st)
                for ii in range(self.N_CROW):
                    sol[it,ind_modes,ii] = res[ii*self.N_points+ind_modes]
                
                
            if out_param == 'map':
                return sol/np.sqrt(2*self.g0/self.kappa_0)
            elif out_param == 'fin_res':
                return sol[-1, :]/np.sqrt(2*self.g0/self.kappa_0)
            else:
                print ('wrong parameter')
                
        def Propagate_SAMCLIB(self, simulation_parameters, Pump, BC, Seed=[0], dt=5e-4,HardSeed=False):
            
            
            T = simulation_parameters['slow_time']
            abtol = simulation_parameters['absolute_tolerance']
            reltol = simulation_parameters['relative_tolerance']
            out_param = simulation_parameters['output']
            nmax = simulation_parameters['max_internal_steps']
            detuning = simulation_parameters['detuning_array']
            eps = simulation_parameters['noise_level']
            
            pump = Pump*np.sqrt(1./(hbar*self.w0))
            
            
            if HardSeed == False:
                
                seed = self.seed_level(Pump, detuning[0])*np.sqrt(2*self.g0/self.kappa_0)
                
            else:
                seed = Seed.T.reshape(Seed.size)*np.sqrt(2*self.g0/self.kappa_0)
                seed/=self.N_points
            
            ### renormalization
            T_rn = (self.kappa_0/2)*T
            f0 = np.fft.ifft(pump*np.sqrt(8*self.g0*self.kappa_ex/self.kappa_0**3),axis=0)*self.N_points
            
            print('f0^2 = ' + str(np.round((abs(f0[0,:])**2), 2)))
            print('xi [' + str(detuning[0]*2/self.kappa_0) + ',' +str(detuning[-1]*2/self.kappa_0)+ '] (normalized on ' r'$kappa_0/2)$')
            noise_const = self.noise(eps) # set the noise level
            nn = len(detuning)
            
            t_st = float(T_rn)/len(detuning)
                
            sol = np.ndarray(shape=(len(detuning), self.N_points, self.N_CROW), dtype='complex') # define an array to store the data
            
            ind_modes = np.arange(self.N_points)
            ind_res = np.arange(self.N_CROW)
            j = np.zeros(self.J[0,:].size)
            delta = np.zeros(self.Delta[0,:].size)
            kappa = np.zeros(self.N_CROW)
            
            
            for ii in range(self.J[0,:].size):
                j[ii] = self.J[0,ii]
            for ii in range(self.N_CROW):
                sol[0,ind_modes,ii] = seed[ii*self.N_points+ind_modes]
                kappa[ii] = self.kappa[0,ii]
                delta[ii] = self.Delta[0,ii]
            
            #if self.Snake_coupling==True:
            bus_j = np.zeros(self.Bus_J[0,:].size)
            bus_phase = np.zeros(self.Bus_Phase[:].size)
            for ii in range(self.Bus_J[0,:].size):
                bus_j[ii] = self.Bus_J[0,ii]
                
            for ii in range(self.Bus_Phase[:].size):
                bus_phase[ii] = self.Bus_Phase[ii]
            
            f0 =(f0.T.reshape(f0.size))
            #%% crtypes definition
            
            if self.J[0,:].size == self.N_CROW:
                BC='PERIODIC'
            elif self.J[0,:].size == self.N_CROW-1:
                BC='OPEN'
            else:
                sys.exit('Unkown type of CROW')
                    
            
            if BC=='OPEN':
                if self.Snake_coupling==False:
                    if abs(self.Delta_D1.max())==0:
                        CROW_core = ctypes.CDLL(PyCORe_directory+'/lib/lib_crow_core.so')
                    if abs(self.Delta_D1.max())>0:
                        CROW_core = ctypes.CDLL(PyCORe_directory+'/lib/lib_crow_core_different_FSR.so')
                else :
                    CROW_core = ctypes.CDLL(PyCORe_directory+'/lib/lib_snake_coupling_crow_core.so')    
            elif BC=='PERIODIC':
                if self.Snake_coupling==False:
                    CROW_core = ctypes.CDLL(PyCORe_directory+'/lib/lib_periodic_crow_core.so')
                else:
                    CROW_core = ctypes.CDLL(PyCORe_directory+'/lib/lib_snake_coupling_periodic_crow_core.so')    
            else:
                sys.exit('Solver has not been found')
            
            if self.n2t==0:
                CROW_core.PropagateSAM.restype = ctypes.c_void_p
            else:
                CROW_core.PropagateThermalSAM.restype = ctypes.c_void_p
                
            A = np.zeros([self.N_CROW*self.N_points],dtype=complex)
            for ii in range(self.N_CROW):    
                A[ii*self.N_points+ind_modes] = np.fft.ifft(seed[ii*self.N_points+ind_modes])*self.N_points
                
        
            In_val_RE = np.array(np.real(A),dtype=ctypes.c_double)
            In_val_IM = np.array(np.imag(A),dtype=ctypes.c_double)
            In_phi = np.array(self.phi,dtype=ctypes.c_double)
            In_Nphi = ctypes.c_int(self.N_points)
            In_Ncrow = ctypes.c_int(self.N_CROW)
            In_f_RE = np.array(np.real(f0 ),dtype=ctypes.c_double)
            In_f_IM = np.array(np.imag(f0 ),dtype=ctypes.c_double)
            In_atol = ctypes.c_double(abtol)
            In_rtol = ctypes.c_double(reltol)
            
            In_det = np.array(detuning,dtype=ctypes.c_double)
            In_Ndet = ctypes.c_int(len(detuning))
            In_D2 = np.array(self.D2,dtype=ctypes.c_double)
            
            if self.n2t!=0:
                In_t_th = ctypes.c_double(self.t_th)
                In_n2 = ctypes.c_double(self.n2)
                In_n2t = ctypes.c_double(self.n2t)
            
            In_kappa = np.array(kappa,dtype=ctypes.c_double)
            In_delta = np.array(delta,dtype=ctypes.c_double)
            In_kappa_0 = ctypes.c_double(self.kappa_0)
            In_J = np.array(j,dtype=ctypes.c_double)
            
            #if self.Snake_coupling==True:
            In_bus_J = np.array(bus_j,dtype=ctypes.c_double)
            In_bus_phase = np.array(bus_phase,dtype=ctypes.c_double)
            
            In_Tmax = ctypes.c_double(t_st)
            In_Nt = ctypes.c_int(int(t_st/dt)+1)
            In_dt = ctypes.c_double(dt)
            In_noise_amp = ctypes.c_double(eps)
            
            In_res_RE = np.zeros(len(detuning)*self.N_points*self.N_CROW,dtype=ctypes.c_double)
            In_res_IM = np.zeros(len(detuning)*self.N_points*self.N_CROW,dtype=ctypes.c_double)
            
            double_p=ctypes.POINTER(ctypes.c_double)
            
            if self.Delta_D1.size==self.N_CROW:
                In_delta_D1 = np.array(self.Delta_D1,dtype=ctypes.c_double)
                In_delta_D1_p = In_delta_D1.ctypes.data_as(double_p)
            
            In_val_RE_p = In_val_RE.ctypes.data_as(double_p)
            In_val_IM_p = In_val_IM.ctypes.data_as(double_p)
            In_phi_p = In_phi.ctypes.data_as(double_p)
            In_det_p = In_det.ctypes.data_as(double_p)
            In_D2_p = In_D2.ctypes.data_as(double_p)
            
            In_kappa_p = In_kappa.ctypes.data_as(double_p)
            In_delta_p = In_delta.ctypes.data_as(double_p)
            In_J_p = In_J.ctypes.data_as(double_p)
            
            #if self.Snake_coupling==True:
            In_bus_j_p = In_bus_J.ctypes.data_as(double_p)
            In_bus_phase_p = In_bus_phase.ctypes.data_as(double_p)
            
            In_f_RE_p = In_f_RE.ctypes.data_as(double_p)
            In_f_IM_p = In_f_IM.ctypes.data_as(double_p)
            
            In_res_RE_p = In_res_RE.ctypes.data_as(double_p)
            In_res_IM_p = In_res_IM.ctypes.data_as(double_p)
            
            
            
                
            if self.Snake_coupling==False:
                
                if self.n2t==0:
                    if abs(self.Delta_D1.max())==0:
                        CROW_core.PropagateSAM(In_val_RE_p, In_val_IM_p, In_f_RE_p, In_f_IM_p, In_det_p, In_kappa_p, In_kappa_0, In_delta_p, In_J_p, In_phi_p, In_D2_p, In_Ndet, In_Nt, In_dt, In_atol, In_rtol, In_Nphi, In_Ncrow, In_noise_amp, In_res_RE_p, In_res_IM_p)
                    else:
                        CROW_core.PropagateSAM(In_val_RE_p, In_val_IM_p, In_f_RE_p, In_f_IM_p, In_det_p, In_kappa_p, In_kappa_0, In_delta_p, In_delta_D1_p, In_J_p, In_phi_p, In_D2_p, In_Ndet, In_Nt, In_dt, In_atol, In_rtol, In_Nphi, In_Ncrow, In_noise_amp, In_res_RE_p, In_res_IM_p)
                else:
                    CROW_core.PropagateThermalSAM(In_val_RE_p, In_val_IM_p, In_f_RE_p, In_f_IM_p, In_det_p, In_kappa_p, In_kappa_0, In_t_th, In_n2, In_n2t, In_delta_p, In_J_p, In_phi_p, In_D2_p, In_Ndet, In_Nt, In_dt, In_atol, In_rtol, In_Nphi, In_Ncrow, In_noise_amp, In_res_RE_p, In_res_IM_p)
            else:
                CROW_core.PropagateSAM(In_val_RE_p, In_val_IM_p, In_f_RE_p, In_f_IM_p, In_det_p, In_kappa_p, In_kappa_0, In_delta_p, In_J_p, In_bus_j_p, In_bus_phase_p, In_phi_p, In_D2_p, In_Ndet, In_Nt, In_dt, In_atol, In_rtol, In_Nphi, In_Ncrow, In_noise_amp, In_res_RE_p, In_res_IM_p)
            
            ind_modes = np.arange(self.N_points)
            for ii in range(0,len(detuning)):
                for jj in range(self.N_CROW):
                    sol[ii,ind_modes,jj] = np.fft.fft(In_res_RE[ii*self.N_points*self.N_CROW + jj*self.N_points+ind_modes] + 1j*In_res_IM[ii*self.N_points*self.N_CROW+ jj*self.N_points+ind_modes])#/np.sqrt(self.N_points)
                
            #sol = np.reshape(In_res_RE,[len(detuning),self.N_points]) + 1j*np.reshape(In_res_IM,[len(detuning),self.N_points])
                        
            if out_param == 'map':
                return sol/np.sqrt(2*self.g0/self.kappa_0)
            elif out_param == 'fin_res':
                return sol[-1, :]/np.sqrt(2*self.g0/self.kappa_0)
            else:
                print ('wrong parameter')
         
        def Propagate_PSEUDO_SPECTRAL_SAMCLIB(self, simulation_parameters, Pump, BC, Seed=[0], dt=5e-4,HardSeed=False, lib='NR'):
            
            
            T = simulation_parameters['slow_time']
            abtol = simulation_parameters['absolute_tolerance']
            reltol = simulation_parameters['relative_tolerance']
            out_param = simulation_parameters['output']
            nmax = simulation_parameters['max_internal_steps']
            detuning = simulation_parameters['detuning_array']
            eps = simulation_parameters['noise_level']
            
            pump = Pump*np.sqrt(1./(hbar*self.w0))
            
            
            if HardSeed == False:
                
                seed = self.seed_level(Pump, detuning[0])*np.sqrt(2*self.g0/self.kappa_0)
                
            else:
                seed = Seed.T.reshape(Seed.size)*np.sqrt(2*self.g0/self.kappa_0)
                seed/=self.N_points
            
            ### renormalization
            T_rn = (self.kappa_0/2)*T
            f0 = np.fft.ifft(pump*np.sqrt(8*self.g0*self.kappa_ex/self.kappa_0**3),axis=0)*self.N_points
            
            print('f0^2 = ' + str(np.round((abs(f0[0,:])**2), 2)))
            print('xi [' + str(detuning[0]*2/self.kappa_0) + ',' +str(detuning[-1]*2/self.kappa_0)+ '] (normalized on ' r'$kappa_0/2)$')
            noise_const = self.noise(eps) # set the noise level
            nn = len(detuning)
            
            t_st = float(T_rn)/len(detuning)
                
            sol = np.ndarray(shape=(len(detuning), self.N_points, self.N_CROW), dtype='complex') # define an array to store the data
            
            ind_modes = np.arange(self.N_points)
            ind_res = np.arange(self.N_CROW)
            j = np.zeros(self.J[0,:].size)
            delta = np.zeros(self.Delta[0,:].size)
            kappa = np.zeros(self.N_CROW)
            
            
            for ii in range(self.J[0,:].size):
                j[ii] = self.J[0,ii]
            for ii in range(self.N_CROW):
                sol[0,ind_modes,ii] = seed[ii*self.N_points+ind_modes]
                kappa[ii] = self.kappa[0,ii]
                delta[ii] = self.Delta[0,ii]
            
            #if self.Snake_coupling==True:
            bus_j = np.zeros(self.Bus_J[0,:].size)
            bus_phase = np.zeros(self.Bus_Phase[:].size)
            for ii in range(self.Bus_J[0,:].size):
                bus_j[ii] = self.Bus_J[0,ii]
                
            for ii in range(self.Bus_Phase[:].size):
                bus_phase[ii] = self.Bus_Phase[ii]
            
            f0 =(f0.T.reshape(f0.size))
            #%% crtypes definition
            
            if self.J[0,:].size == self.N_CROW:
                BC='PERIODIC'
            elif self.J[0,:].size == self.N_CROW-1:
                BC='OPEN'
            else:
                sys.exit('Unkown type of CROW')
                    
            
            if BC=='OPEN':
                if lib=='NR':
                    CROW_core = ctypes.CDLL(PyCORe_directory+'/lib/lib_crow_core.so')   
                elif lib=='boost':
                    CROW_core = ctypes.CDLL(PyCORe_directory+'/lib/lib_boost_crow_core.so')   
                else:
                    sys.exit('Solver has not been found')
            
            elif BC=='PERIODIC':
                CROW_core = ctypes.CDLL(PyCORe_directory+'/lib/lib_periodic_crow_core.so')
            else:
                sys.exit('Solver has not been found')
            print(BC)
            CROW_core.Propagate_PseudoSpectralSAM.restype = ctypes.c_void_p
            #if self.n2t==0:
            #    
            #else:
            #    pass
                #CROW_core.PropagateThermalSAM.restype = ctypes.c_void_p
                
            A = np.zeros([self.N_CROW*self.N_points],dtype=complex)
            for ii in range(self.N_CROW):    
                A[ii*self.N_points+ind_modes] = np.fft.ifft(seed[ii*self.N_points+ind_modes])*self.N_points
                
        
            In_val_RE = np.array(np.real(A),dtype=ctypes.c_double)
            In_val_IM = np.array(np.imag(A),dtype=ctypes.c_double)
            In_phi = np.array(self.phi,dtype=ctypes.c_double)
            In_Nphi = ctypes.c_int(self.N_points)
            In_Ncrow = ctypes.c_int(self.N_CROW)
            In_f_RE = np.array(np.real(f0 ),dtype=ctypes.c_double)
            In_f_IM = np.array(np.imag(f0 ),dtype=ctypes.c_double)
            In_atol = ctypes.c_double(abtol)
            In_rtol = ctypes.c_double(reltol)
            
            In_det = np.array(detuning,dtype=ctypes.c_double)
            In_Ndet = ctypes.c_int(len(detuning))
            In_Dint = np.array(self.Dint.T.reshape(self.Dint.size),dtype=ctypes.c_double)
            
            if self.n2t!=0:
                In_t_th = ctypes.c_double(self.t_th)
                In_n2 = ctypes.c_double(self.n2)
                In_n2t = ctypes.c_double(self.n2t)
            
            In_kappa = np.array(kappa,dtype=ctypes.c_double)
            In_delta = np.array(delta,dtype=ctypes.c_double)
            In_kappa_0 = ctypes.c_double(self.kappa_0)
            In_J = np.array(j,dtype=ctypes.c_double)
            
            #if self.Snake_coupling==True:
            In_bus_J = np.array(bus_j,dtype=ctypes.c_double)
            In_bus_phase = np.array(bus_phase,dtype=ctypes.c_double)
            
            In_Tmax = ctypes.c_double(t_st)
            In_Nt = ctypes.c_int(int(t_st/dt)+1)
            In_dt = ctypes.c_double(dt)
            In_noise_amp = ctypes.c_double(eps)
            
            In_res_RE = np.zeros(len(detuning)*self.N_points*self.N_CROW,dtype=ctypes.c_double)
            In_res_IM = np.zeros(len(detuning)*self.N_points*self.N_CROW,dtype=ctypes.c_double)
            
            double_p=ctypes.POINTER(ctypes.c_double)
            
            if self.Delta_D1.size==self.N_CROW:
                In_delta_D1 = np.array(self.Delta_D1,dtype=ctypes.c_double)
                In_delta_D1_p = In_delta_D1.ctypes.data_as(double_p)
            
            In_val_RE_p = In_val_RE.ctypes.data_as(double_p)
            In_val_IM_p = In_val_IM.ctypes.data_as(double_p)
            In_phi_p = In_phi.ctypes.data_as(double_p)
            In_det_p = In_det.ctypes.data_as(double_p)
            In_Dint_p = In_Dint.ctypes.data_as(double_p)
            
            In_kappa_p = In_kappa.ctypes.data_as(double_p)
            In_delta_p = In_delta.ctypes.data_as(double_p)
            In_J_p = In_J.ctypes.data_as(double_p)
            
            #if self.Snake_coupling==True:
            In_bus_j_p = In_bus_J.ctypes.data_as(double_p)
            In_bus_phase_p = In_bus_phase.ctypes.data_as(double_p)
            
            In_f_RE_p = In_f_RE.ctypes.data_as(double_p)
            In_f_IM_p = In_f_IM.ctypes.data_as(double_p)
            
            In_res_RE_p = In_res_RE.ctypes.data_as(double_p)
            In_res_IM_p = In_res_IM.ctypes.data_as(double_p)
            
   
                
            if self.n2t==0:
                CROW_core.Propagate_PseudoSpectralSAM(In_val_RE_p, In_val_IM_p, In_f_RE_p, In_f_IM_p, In_det_p, In_kappa_p, In_kappa_0, In_delta_p, In_J_p, In_phi_p, In_Dint_p, In_Ndet, In_Nt, In_dt, In_atol, In_rtol, In_Nphi, In_Ncrow, In_noise_amp, In_res_RE_p, In_res_IM_p)                
                
            else:
                CROW_core.Propagate_PseudoSpectralThermalSAM(In_val_RE_p, In_val_IM_p, In_f_RE_p, In_f_IM_p, In_det_p, In_kappa_p, In_kappa_0, In_t_th, In_n2, In_n2t, In_delta_p, In_J_p, In_phi_p, In_Dint_p, In_Ndet, In_Nt, In_dt, In_atol, In_rtol, In_Nphi, In_Ncrow, In_noise_amp, In_res_RE_p, In_res_IM_p)                
            #if self.n2t==0:
            #    if abs(self.Delta_D1.max())==0:
            #        CROW_core.PropagateSAM(In_val_RE_p, In_val_IM_p, In_f_RE_p, In_f_IM_p, In_det_p, In_kappa_p, In_kappa_0, In_delta_p, In_J_p, In_phi_p, In_D2_p, In_Ndet, In_Nt, In_dt, In_atol, In_rtol, In_Nphi, In_Ncrow, In_noise_amp, In_res_RE_p, In_res_IM_p)
            #    else:
            #        CROW_core.PropagateSAM(In_val_RE_p, In_val_IM_p, In_f_RE_p, In_f_IM_p, In_det_p, In_kappa_p, In_kappa_0, In_delta_p, In_delta_D1_p, In_J_p, In_phi_p, In_D2_p, In_Ndet, In_Nt, In_dt, In_atol, In_rtol, In_Nphi, In_Ncrow, In_noise_amp, In_res_RE_p, In_res_IM_p)
            #else:
            #    CROW_core.PropagateThermalSAM(In_val_RE_p, In_val_IM_p, In_f_RE_p, In_f_IM_p, In_det_p, In_kappa_p, In_kappa_0, In_t_th, In_n2, In_n2t, In_delta_p, In_J_p, In_phi_p, In_D2_p, In_Ndet, In_Nt, In_dt, In_atol, In_rtol, In_Nphi, In_Ncrow, In_noise_amp, In_res_RE_p, In_res_IM_p)
            
            ind_modes = np.arange(self.N_points)
            for ii in range(0,len(detuning)):
                for jj in range(self.N_CROW):
                    sol[ii,ind_modes,jj] = np.fft.fft(In_res_RE[ii*self.N_points*self.N_CROW + jj*self.N_points+ind_modes] + 1j*In_res_IM[ii*self.N_points*self.N_CROW+ jj*self.N_points+ind_modes])#/np.sqrt(self.N_points)
                
            #sol = np.reshape(In_res_RE,[len(detuning),self.N_points]) + 1j*np.reshape(In_res_IM,[len(detuning),self.N_points])
                        
            if out_param == 'map':
                return sol/np.sqrt(2*self.g0/self.kappa_0)
            elif out_param == 'fin_res':
                return sol[-1, :]/np.sqrt(2*self.g0/self.kappa_0)
            else:
                print ('wrong parameter')
            
        def Jacobian(self,j,d2,dphi,delta, kappa, zeta_0,A):
            N_m = self.N_points
            N_res = self.N_CROW
            ind_m = np.arange(self.N_points)
            ind_res = np.arange(self.N_CROW)
            
            J = np.zeros([2*N_m*N_res,2*N_m*N_res],dtype=complex)
            
            
            
            for jj in ind_res:
                            
                J[jj*N_m+ind_m,jj*N_m+ind_m] =0.5*(-(kappa[jj]+ 1j*(zeta_0+delta[jj]))  - 2*1j*d2[jj]/dphi**2 +2*1j*abs(A[jj*N_m+ind_m])**2 )
                J[(jj+N_res)*N_m+ind_m,(jj+N_res)*N_m+ind_m] = 0.5*(-(kappa[jj]- 1j*(zeta_0+delta[jj]))  + 2*1j*d2[jj]/dphi**2 -2*1j*abs(A[jj*N_m+ind_m])**2)
                
                J[jj*N_m+ind_m[:-1],jj*N_m+ind_m[1:]] = 1j*d2[jj]/dphi**2
                J[jj*N_m+0,jj*N_m+N_m-1] =  1j*d2[jj]/dphi**2
                
                J[(jj+N_res)*N_m+ind_m[:-1],(jj+N_res)*N_m+ind_m[1:]] = -1j*d2[jj]/dphi**2
                J[(jj+N_res)*N_m+0,(jj+N_res)*N_m+N_m-1] =  -1j*d2[jj]/dphi**2
           
            for jj in ind_res[:-1]:
                 J[jj*N_m+ind_m[:],(jj+1)*N_m+ind_m[:]] = 1j*j[jj]
                 J[(jj+N_res)*N_m+ind_m[:],((jj+N_res)+1)*N_m+ind_m[:]] = -1j*j[jj]
                
                
            if d2.size==j.size:
                jj=d2.size-1
                J[jj*N_m+ind_m[:],(0)*N_m+ind_m[:]] = 1j*j[jj]
                J[(jj+N_res)*N_m+ind_m[:],(0+N_res)*N_m+ind_m[:]] = -1j*j[jj]
                #D[jj*N_m*2+N_m+ind_m[:],(0)*N_m*2+N_m+ind_m[:]] = -1j*j[jj]
            J += J.T     
            for jj in ind_res:
                 J[jj*N_m+ind_m,(jj+N_res)*N_m+ind_m] = 1j*A[jj*N_m+ind_m]*A[jj*N_m+ind_m]
                 J[(jj+N_res)*N_m+ind_m,(jj)*N_m+ind_m] = -1j*np.conj(A[jj*N_m+ind_m])*np.conj(A[jj*N_m+ind_m])
                        
            #J+=np.conj(J.T)
            #for jj in ind_res:
                            
            #    J[jj*N_m+ind_m,jj*N_m+ind_m] = 2*1j*abs(A[jj*N_m+ind_m])**2
            #    J[(jj+N_res)*N_m+ind_m,(jj+N_res)*N_m+ind_m] = -2*1j*abs(A[jj*N_m+ind_m])**2
            
            #J += M_lin
            Jacob_sparse = dia_matrix(J)
            return Jacob_sparse
        def JacobianMatrix(self,zeta_0,A,order=0):
            if self.M_lin[:].size==1:
                self.M_lin = self.LinearMatrix(zeta_0,order)
                
            N_m = self.N_points
            N_res = self.N_CROW
            ind_m = np.arange(self.N_points)
            ind_res = np.arange(self.N_CROW)
            
            J = np.zeros([2*N_m*N_res,2*N_m*N_res],dtype=complex)
            for jj in ind_res:
                J[jj*N_m+ind_m,jj*N_m+ind_m] += +2*1j*abs(A[jj*N_m+ind_m])**2 
                J[(jj+N_res)*N_m+ind_m,(jj+N_res)*N_m+ind_m] += -2*1j*abs(A[jj*N_m+ind_m])**2
                J[jj*N_m+ind_m,(jj+N_res)*N_m+ind_m] = 1j*A[jj*N_m+ind_m]*A[jj*N_m+ind_m]
                J[(jj+N_res)*N_m+ind_m,(jj)*N_m+ind_m] = -1j*np.conj(A[jj*N_m+ind_m])*np.conj(A[jj*N_m+ind_m])
            J+= self.M_lin
            
            
            if order == 0:
                Jacob_sparse =(J) 
            else:
                Jacob_sparse = dia_matrix(J)    
                
            return Jacob_sparse
        def DispersionMatrix(self,order=0):
            
            
            D = np.zeros([self.N_points*self.N_CROW,self.N_points*self.N_CROW],dtype=complex)
           
            ind_m = np.arange(self.N_points)
            ind_res = np.arange(self.N_CROW)
            d2 = self.D2/self.kappa_0
            dphi = abs(self.phi[1]-self.phi[0])
            if order == 0:
                Fourier_matrix = dft(self.N_points)
                D_fourier = np.zeros([self.N_points,self.N_points],dtype=complex)
               
                for jj in ind_res:      
                   D_fourier[ind_m,ind_m] =-(self.kappa[:,jj]/2+1j*self.Delta[:,jj])*2/self.kappa_0 -1j*self.Dint[:,jj]*2/self.kappa_0
                   D[jj*self.N_points+ind_m[0]:jj*self.N_points+ind_m[-1]+1,jj*self.N_points+ind_m[0]:jj*self.N_points+ind_m[-1]+1] = np.dot(np.dot(Fourier_matrix,D_fourier),np.conj(Fourier_matrix.T)/self.N_points)
            if order == 2:
                N_m = self.N_points
                N_res = self.N_CROW
                for jj in ind_res:
                    D[jj*N_m+ind_m[:-1],jj*N_m+ind_m[1:]] = 1j*d2[jj]/dphi**2
                    D[jj*N_m+0,jj*N_m+N_m-1] =  1j*d2[jj]/dphi**2
                    
                    #D[(jj+N_res)*N_m+ind_m[:-1],(jj+N_res)*N_m+ind_m[1:]] = -1j*d2[jj]/dphi**2
                    #D[(jj+N_res)*N_m+0,(jj+N_res)*N_m+N_m-1] =  -1j*d2[jj]/dphi**2
                    
                    
                    D[jj*N_m+ind_m,jj*N_m+ind_m] = 0.5*(-(self.kappa[:,jj]/2+1j*self.Delta[:,jj])*2/self.kappa_0  - 2*1j*d2[jj]/dphi**2)
                D+=D.T
           
            return D
        
        def LinearMatrix(self,zeta_0,order=0):
            
            N_m = self.N_points
            N_res = self.N_CROW
            ind_m = np.arange(self.N_points)
            ind_res = np.arange(self.N_CROW)
            if self.D[:].size==1:
                self.D = self.DispersionMatrix(order)
            M_lin = np.zeros([2*self.N_points*self.N_CROW,2*self.N_points*self.N_CROW],dtype=complex) 
            
            for jj in ind_res[:-1]:
                M_lin[jj*N_m+ind_m[:],(jj+1)*N_m+ind_m[:]] += 1j*self.J[0,jj]*2/self.kappa_0
                M_lin[(jj+N_res)*N_m+ind_m[:],((jj+N_res)+1)*N_m+ind_m[:]] += -1j*self.J[0,jj]*2/self.kappa_0
            if self.Dint[0,:].size==self.J[0,:].size:
                jj=self.Dint[0,:].size-1
                M_lin[jj*N_m+ind_m[:],(0)*N_m+ind_m[:]] += 1j*self.J[0,jj]*2/self.kappa_0
                M_lin[(jj+N_res)*N_m+ind_m[:],(0+N_res)*N_m+ind_m[:]] += -1j*self.J[0,jj]*2/self.kappa_0
                
            M_lin += M_lin.T
            
            
            M_lin[:self.N_points*self.N_CROW,:self.N_points*self.N_CROW] += self.D
            M_lin[self.N_points*self.N_CROW:,self.N_points*self.N_CROW:] += np.conj(self.D)
            
            for jj in ind_res:
                M_lin[jj*N_m+ind_m,jj*N_m+ind_m] += -1j*(zeta_0) 
                M_lin[(jj+N_res)*N_m+ind_m,(jj+N_res)*N_m+ind_m] += 1j*(zeta_0) 
            return M_lin
        def LinMatrix(self,j,d2,dphi,delta, kappa, zeta_0):
            
            N_m = self.N_points
            N_res = self.N_CROW
            ind_m = np.arange(self.N_points)
            ind_res = np.arange(self.N_CROW)
            
            
            D = np.zeros([2*N_m*N_res,2*N_m*N_res],dtype=complex)
            
            for jj in ind_res:
                D[jj*N_m+ind_m[:-1],jj*N_m+ind_m[1:]] = 1j*d2[jj]/dphi**2
                D[jj*N_m+0,jj*N_m+N_m-1] =  1j*d2[jj]/dphi**2
                
                D[(jj+N_res)*N_m+ind_m[:-1],(jj+N_res)*N_m+ind_m[1:]] = -1j*d2[jj]/dphi**2
                D[(jj+N_res)*N_m+0,(jj+N_res)*N_m+N_m-1] =  -1j*d2[jj]/dphi**2
                
                
                D[jj*N_m+ind_m,jj*N_m+ind_m] = 0.5*(-(kappa[jj]+ 1j*(zeta_0+delta[jj]))  - 2*1j*d2[jj]/dphi**2)
                D[(jj+N_res)*N_m+ind_m,(jj+N_res)*N_m+ind_m] = 0.5*(-(kappa[jj]- 1j*(zeta_0+delta[jj]))  + 2*1j*d2[jj]/dphi**2)
                
                
            for jj in ind_res[:-1]:
                D[jj*N_m+ind_m[:],(jj+1)*N_m+ind_m[:]] = 1j*j[jj]
                D[(jj+N_res)*N_m+ind_m[:],((jj+N_res)+1)*N_m+ind_m[:]] = -1j*j[jj]
                
                
            if d2.size==j.size:
                jj=d2.size-1
                D[jj*N_m+ind_m[:],(0)*N_m+ind_m[:]] = 1j*j[jj]
                D[(jj+N_res)*N_m+ind_m[:],(0+N_res)*N_m+ind_m[:]] = -1j*j[jj]
                #D[jj*N_m*2+N_m+ind_m[:],(0)*N_m*2+N_m+ind_m[:]] = -1j*j[jj]
            D += D.T
            
            #D[:N_m*N_res]=np.conj(D[N_m*N_res:])
            
            return D
        
        def MatFormLLE(self,A,f0):
            index_1 = np.arange(0,self.N_points*self.N_CROW)
            index_2 = np.arange(self.N_points*self.N_CROW,2*self.N_points*self.N_CROW)
            result = np.zeros(2*self.N_points*self.N_CROW,dtype=complex)
            
            result[index_1] =  1j*abs(A[index_1])**2*A[index_1]         
            #buf[index_2] = np.conj(buf[index_1])
            result[index_2] =  -1j*abs(A[index_2])**2*A[index_2]         
            result += (self.M_lin).dot(A) + f0
            
            return result
            
        def NewtonRaphsonDirectSpace(self,Seed_sol,dOm, Pump, HardSeed = True, tol=1e-5,max_iter=50,order=0,learning_rate=1e-1):
            A_guess = np.fft.ifft(Seed_sol,axis=0)
            result = np.zeros_like(A_guess,dtype=complex)
            zeta_0 = dOm*2/self.kappa_0
            N_m = self.N_points
            N_res = self.N_CROW
            ind_modes = np.arange(self.N_points)
            ind_res = np.arange(self.N_CROW)
            pump = Pump*np.sqrt(1./(hbar*self.w0))
            f0_direct = np.fft.ifft(pump*np.sqrt(8*self.g0*np.max(self.kappa_ex)/self.kappa_0**3),axis=0)*self.N_points
            index_1 = np.arange(0,N_m*N_res)
            index_2 = np.arange(N_m*N_res,2*N_m*N_res)
            
            f0_direct =(f0_direct.T.reshape(f0_direct.size))
            self.D = self.DispersionMatrix(order=order)
            self.M_lin = self.LinearMatrix(zeta_0)
            
            
            Aprev = np.zeros(2*N_m*N_res,dtype=complex)
            if HardSeed == False:
                A_guess = A_guess.T.reshape(A_guess.size)+ solve(Mlin[:N_m*N_res,:N_m*N_res],-f0_direct)
                Aprev[index_1] = A_guess
            else:
                Aprev[index_1] = A_guess.T.reshape(A_guess.size)*np.sqrt(2*self.g0/self.kappa_0)
            
            
            Aprev[index_2] = np.conj(Aprev[index_1])
            
            Ak = np.zeros(Aprev.size,dtype=complex)
            
            
            
            f0 = np.zeros(Aprev.size,dtype=complex)
            f0[index_1] = f0_direct
            
            f0[index_2] = np.conj(f0[index_1])
    
            buf= np.zeros(Aprev.size,dtype=complex)
            J=self.JacobianMatrix(zeta_0, Aprev[index_1],order)
            print('f0^2 = ' + str(np.round(max(abs(f0_direct)**2), 2)))
            print('xi = ' + str(zeta_0) )
            
            diff = self.N_points
            rel_diff = self.N_points
            counter =0
            diff_array=[]
            rel_diff_array=[]
            isSparse = isspmatrix(J)
            #min_res = minimize(self.MatFormLLE, (2, 0), args=(f0),method='SLSQP')
            while diff>tol:
                J=self.JacobianMatrix(zeta_0, Aprev[index_1],order)
                #buf[index_1] =  1j*abs(Aprev[index_1])**2*Aprev[index_1]         
                
                #buf[index_2] =  -1j*abs(Aprev[index_2])**2*Aprev[index_2]         
                #buf += (self.M_lin).dot(Aprev) + f0
                buf = self.MatFormLLE(Aprev,f0)
                if isSparse==False:
                    Ak = Aprev - solve_dense(J,buf)*learning_rate
                else:
                    Ak = Aprev - solve_sparse(J,buf)*learning_rate
                
                
                rel_diff = np.sqrt(abs((Ak-Aprev).dot(np.conj(Ak-Aprev))/(Ak.dot(np.conj(Ak)))))
                rel_diff_array += [rel_diff]
                diff = np.sqrt(abs((Ak-Aprev).dot(np.conj(Ak-Aprev))))
                diff_array += [diff]

                
                Aprev = Ak
                Aprev[index_2] = np.conj(Aprev[index_1])
                counter +=1
                print(diff)
                #plt.scatter(counter,diff,c='k')
                if counter>max_iter:
                    print("Did not coverge in " + str(max_iter)+ " iterations, relative error is " + str(diff))
                    res = np.zeros(self.N_points,dtype=complex)
                    res = Ak[index_1]
                    
                    for jj in ind_res:
                        result[ind_modes,jj]= np.fft.fft(res[jj*N_m+ind_modes])
                    return result/np.sqrt(2*self.g0/self.kappa_0), diff_array
                    break
            print("Converged in " + str(counter) + " iterations, relative error is " + str(diff))
            
            res = np.zeros(self.N_points,dtype=complex)
            res = Ak[index_1]
            #res = men_res.x[:self.N_points]
            for jj in ind_res:
                result[ind_modes,jj]= np.fft.fft(res[jj*N_m+ind_modes])
            return result/np.sqrt(2*self.g0/self.kappa_0), diff_array    
            
        def NewtonRaphson(self,Seed_sol,dOm, Pump, HardSeed = True, tol=1e-5,max_iter=50, learning_rate=1e-1):
            A_guess = np.fft.ifft(Seed_sol,axis=0)
            result = np.zeros_like(A_guess,dtype=complex)
            N_m = self.N_points
            N_res = self.N_CROW
            ind_modes = np.arange(self.N_points)
            ind_res = np.arange(self.N_CROW)
            j = np.zeros(self.J[0,:].size)
            delta = np.zeros(self.Delta[0,:].size)
            kappa = np.zeros(self.N_CROW)
            d2 = np.zeros(self.N_CROW)
            
            for ii in range(self.J[0,:].size):
                j[ii] = self.J[0,ii]*2/self.kappa_0
            for ii in range(self.N_CROW):
                #A_guess[0,ind_modes,ii] = seed[ii*self.N_points+ind_modes]
                kappa[ii] = self.kappa[0,ii]/self.kappa_0
                delta[ii] = self.Delta[0,ii]*2/self.kappa_0
                d2[ii] = self.D2[ii]/self.kappa_0
                
            
            
            zeta_0 = dOm*2/self.kappa_0
            
            
            
            dphi = abs(self.phi[1]-self.phi[0])
            pump = Pump*np.sqrt(1./(hbar*self.w0))
            
            f0_direct = np.fft.ifft(pump*np.sqrt(8*self.g0*np.max(self.kappa_ex)/self.kappa_0**3),axis=0)*self.N_points
            
            index_1 = np.arange(0,N_m*N_res)
            index_2 = np.arange(N_m*N_res,2*N_m*N_res)
            
            f0_direct =(f0_direct.T.reshape(f0_direct.size))
            
            M_lin = self.LinMatrix(j, d2, dphi, delta, kappa, zeta_0)
            
            
            Aprev = np.zeros(2*N_m*N_res,dtype=complex)
            if HardSeed == False:
                A_guess = A_guess.T.reshape(A_guess.size)+ solve(Mlin[:N_m*N_res,:N_m*N_res],-f0_direct)
                Aprev[index_1] = A_guess
            else:
                Aprev[index_1] = A_guess.T.reshape(A_guess.size)*np.sqrt(2*self.g0/self.kappa_0)
            
            
            Aprev[index_2] = np.conj(Aprev[index_1])
            
            Ak = np.zeros(Aprev.size,dtype=complex)
            
            
            
            f0 = np.zeros(Aprev.size,dtype=complex)
            f0[index_1] = f0_direct
            
            f0[index_2] = np.conj(f0[index_1])
    
            buf= np.zeros(Aprev.size,dtype=complex)
            J=self.Jacobian(j, d2, dphi, delta, kappa, zeta_0, Aprev[index_1])
            print('f0^2 = ' + str(np.round(max(abs(f0_direct)**2), 2)))
            print('xi = ' + str(zeta_0) )
            
            diff = self.N_points
            rel_diff = self.N_points
            counter =0
            diff_array=[]
            rel_diff_array=[]
            
            while diff>tol:
                J=self.Jacobian(j, d2, dphi, delta, kappa, zeta_0, Aprev[index_1])
                
                buf[index_1] =  1j*abs(Aprev[index_1])**2*Aprev[index_1]         
                #buf[index_2] = np.conj(buf[index_1])
                buf[index_2] =  -1j*abs(Aprev[index_2])**2*Aprev[index_2]         
                buf += (M_lin).dot(Aprev) + f0
                
                #inv(M_lin)
                Ak = Aprev - solve_sparse(J,buf)*learning_rate
                
                
                diff = np.sqrt(abs((Ak-Aprev).dot(np.conj(Ak-Aprev))))
                diff_array += [diff]

                
                rel_diff = np.sqrt(abs((Ak-Aprev).dot(np.conj(Ak-Aprev))/(Ak.dot(np.conj(Ak)))))
                rel_diff_array += [rel_diff]

                
                Aprev = Ak
                Aprev[index_2] = np.conj(Aprev[index_1])
                counter +=1
                print(diff)
                #plt.scatter(counter,diff,c='k')
                if counter>max_iter:
                    print("Did not coverge in " + str(max_iter)+ " iterations, relative error is " + str(diff))
                    res = np.zeros(self.N_points,dtype=complex)
                    res = Ak[index_1]
                    
                    for jj in ind_res:
                        result[ind_modes,jj]= np.fft.fft(res[jj*N_m+ind_modes])
                    return result/np.sqrt(2*self.g0/self.kappa_0), diff_array
                    break
            print("Converged in " + str(counter) + " iterations, relative error is " + str(diff))
            res = np.zeros(self.N_points,dtype=complex)
            res = Ak[index_1]
            for jj in ind_res:
                result[ind_modes,jj]= np.fft.fft(res[jj*N_m+ind_modes])
            return result/np.sqrt(2*self.g0/self.kappa_0), diff_array    
        
        def LinearStability(self,Seed_sol,dOm,plot_eigvals=True,get_eigvecs=True,order=0, IsSparse=False, NumOfEigVals=10,which='LM'):
            
            A=np.fft.ifft(Seed_sol, axis=0)
            A = A.T.reshape(A.size)*np.sqrt(2*self.g0/self.kappa_0)
            
                      
            N_m = self.N_points
            N_res = self.N_CROW
            
            index_1 = np.arange(0,N_m*N_res)
            index_2 = np.arange(N_m*N_res,2*N_m*N_res)
            ind_modes = np.arange(self.N_points)
            ind_res = np.arange(self.N_CROW)
            
            A_vec = np.zeros(2*N_m*N_res,dtype=complex)
            A_vec[index_1]=A
            A_vec[index_2]=np.conj(A)
            
            
            
             
            
            
            
            zeta_0 = dOm*2/self.kappa_0
            dphi = abs(self.phi[1]-self.phi[0])
            
            
            
            
            index_1 = np.arange(0,self.N_points)
            index_2 = np.arange(self.N_points,2*self.N_points)
            
            self.D = self.DispersionMatrix(order=order)
            self.M_lin = self.LinearMatrix(zeta_0)
            
            if IsSparse==True:
                Full_Matrix=(self.JacobianMatrix(zeta_0,A_vec))
                
                eig_vals,eig_vec = scp_eigs(Full_Matrix,k=NumOfEigVals,which=which)
            else:    
                Full_Matrix=(self.JacobianMatrix(zeta_0,A_vec))
                if get_eigvecs==True:
                    eig_vals,eig_vec = np.linalg.eig(Full_Matrix)
                if get_eigvecs==False:
                    eig_vals = np.linalg.eigvals(Full_Matrix)
                    #scp_eigs(Full_Matrix, k=6, sigma=-0.01+0.001*1j,which='LR',return_eigenvectors=False)
            
            
            if plot_eigvals==True:
                plt.scatter(np.real(eig_vals),np.imag(eig_vals))
                plt.xlabel('Real part')
                plt.ylabel('Imaginary part')
            if get_eigvecs==True:
                eigen_vectors = np.zeros([self.N_points,self.N_CROW,2*self.N_points*self.N_CROW],dtype=complex)        
                for jj in range(2*self.N_points*self.N_CROW):
                    for ii in ind_res:
                        eigen_vectors[:,ii,jj]=(eig_vec[ii*N_m+ind_modes,jj]).T
                        eigen_vectors[:,ii,jj]=np.fft.fft(eigen_vectors[:,ii,jj])
                
            if get_eigvecs==True:    
                return eig_vals*self.kappa_0/2, eigen_vectors/np.sqrt(2*self.g0/self.kappa_0)
            if get_eigvecs==False:    
                return eig_vals*self.kappa_0/2