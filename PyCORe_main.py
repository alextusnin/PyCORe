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

#%%
from Resonator import Resonator
from SiL_Resonator import SiL_Resonator
from Chains import CROW


#%%

def Plot_Map(map_data, detuning, xlabel='index', units = '', colormap = 'cubehelix'):
    dOm = detuning[1]-detuning[0]
    dt=1
   
   
    Num_of_modes = map_data[0,:].size
    mu = np.arange(-Num_of_modes/2,Num_of_modes/2)
    def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
        '''
        Function to offset the "center" of a colormap. Useful for
        data with a negative min and positive max and you want the
        middle of the colormap's dynamic range to be at zero
    
        Input
        -----
          cmap : The matplotlib colormap to be altered
          start : Offset from lowest point in the colormap's range.
              Defaults to 0.0 (no lower ofset). Should be between
              0.0 and `midpoint`.
          midpoint : The new center of the colormap. Defaults to 
              0.5 (no shift). Should be between 0.0 and 1.0. In
              general, this should be  1 - vmax/(vmax + abs(vmin))
              For example if your data range from -15.0 to +5.0 and
              you want the center of the colormap at 0.0, `midpoint`
              should be set to  1 - 5/(5 + 15)) or 0.75
          stop : Offset from highets point in the colormap's range.
              Defaults to 1.0 (no upper ofset). Should be between
              `midpoint` and 1.0.
        '''
        cdict = {
            'red': [],
            'green': [],
            'blue': [],
            'alpha': []
        }
    
        # regular index to compute the colors
        reg_index = np.linspace(start, stop, 257)
    
        # shifted index to match the data
        shift_index = np.hstack([
            np.linspace(0.0, midpoint, 128, endpoint=False), 
            np.linspace(midpoint, 1.0, 129, endpoint=True)
        ])
    
        for ri, si in zip(reg_index, shift_index):
            r, g, b, a = cmap(ri)
    
            cdict['red'].append((si, r, r))
            cdict['green'].append((si, g, g))
            cdict['blue'].append((si, b, b))
            cdict['alpha'].append((si, a, a))
    
        newcmap = mcolors.LinearSegmentedColormap(name, cdict)
        plt.register_cmap(cmap=newcmap)
    
        return newcmap


    def onclick(event):
        
        ix, iy = event.xdata, event.ydata
        x = int(np.floor((ix-detuning.min())/dOm))
        max_val = (abs(map_data[x,:])**2).max()
        plt.suptitle('Chosen x axis value = %f'%np.round(ix,3) + ' '+units, fontsize=20)
        ax.lines.pop(0)
        ax.plot([ix,ix], [-np.pi, np.pi ],'r')

        ax2 = plt.subplot2grid((5, 1), (2, 0))            
        ax2.plot(phi, abs(map_data[x,:])**2/max_val, 'r')
        ax2.set_ylabel('Intracavity power [a.u.]')
        ax2.set_xlim(-np.pi,np.pi)
        ax2.set_ylim(0,1)        
        ax3 = plt.subplot2grid((5, 1), (3, 0))
        ax3.plot(phi, np.angle(map_data[x,:])/(np.pi),'b')
#        if max( np.unwrap(np.angle(map_data[x,:]))/(np.pi)) - min( np.unwrap(np.angle(map_data[x,:]))/(np.pi))<10:
#            ax3.plot(np.arange(0,dt*np.size(map_data,1),dt), np.unwrap(np.angle(map_data[x,:]))/(np.pi),'g')
        ax3.set_xlabel(r'$\varphi$')
        ax3.set_ylabel('Phase (rad)')
        ax3.set_xlim(-np.pi,np.pi)
        ax3.yaxis.set_major_locator(ticker.MultipleLocator(base=1.0))
        ax3.yaxis.set_major_formatter(ticker.FormatStrFormatter('%g $\pi$'))
        ax3.grid(True)
        
        ax4 = plt.subplot2grid((5, 1), (4, 0))            
        ax4.plot(mu,10*np.log10(abs(np.fft.fftshift(np.fft.fft(map_data[x,:])))**2/(abs(np.fft.fft(map_data[x,:]))**2).max()),'-o', color='black',markersize=3)
        ax4.set_ylabel('Spectrum, dB')
        ax4.set_xlim(mu.min(),mu.max())
        ax4.set_ylim(bottom=-150)
        
        ax4.set_xlim(mu.min()+10,mu.max()-10)
        #ax4.autoscale(enable=None, axis="x", tight=True)
        plt.show()
        f.canvas.draw()
        
    
    f = plt.figure()
    ax = plt.subplot2grid((5, 1), (0, 0), rowspan=2)
    #plt.suptitle('Choose the detuning', fontsize=20)
    f.set_size_inches(10,8)
    phi = np.linspace(-np.pi,np.pi,map_data[0,:].size)
#    orig_cmap = plt.get_cmap('viridis')
#    colormap = shiftedColorMap(orig_cmap, start=0., midpoint=.5, stop=1., name='shrunk')
    pc = ax.pcolormesh(detuning, phi, abs(np.transpose(map_data))**2, cmap=colormap)
    ax.plot([0, 0], [-np.pi, np.pi], 'r')
    ax.set_xlabel('Detuning')
    ax.set_ylabel(r'$\varphi$')
    ax.set_ylim(-np.pi, np.pi)
    ax.set_xlim(detuning.min(),detuning.max())
    ix=0
    
    x = int(((ix-detuning.min())/dOm))
    if (x<0) or (x>detuning.size):
        x = 0
    max_val = (abs(map_data[x,:])**2).max()
    plt.suptitle('Chosen x axis value = %f'%np.round(ix,3) + ' '+units, fontsize=20)
    ax.lines.pop(0)
    
    ax.plot([ix,ix], [-np.pi, np.pi ],'r')
    
    ax2 = plt.subplot2grid((5, 1), (2, 0))            
    ax2.plot(phi,abs(map_data[x,:])**2/max_val, 'r')
    ax2.set_ylabel('Intracavity power [a.u.]')
    ax2.set_xlim(-np.pi,np.pi)
    ax2.set_ylim(0,1)        
    ax3 = plt.subplot2grid((5, 1), (3, 0))
    ax3.plot(phi, np.angle(map_data[x,:])/(np.pi),'b')
#    if max( np.unwrap(np.angle(map_data[x,:]))/(np.pi)) - min( np.unwrap(np.angle(map_data[x,:]))/(np.pi))<10:
#        ax3.plot(np.arange(0,dt*np.size(map_data,1),dt), np.unwrap(np.angle(map_data[x,:]))/(np.pi),'g')
    ax3.set_xlabel(r'$\varphi$')
    ax3.set_ylabel('Phase (rad)')
    ax3.set_xlim(-np.pi,np.pi)
    ax3.yaxis.set_major_locator(ticker.MultipleLocator(base=1.0))
    ax3.yaxis.set_major_formatter(ticker.FormatStrFormatter('%g $\pi$'))
    ax3.grid(True)
    ax4 = plt.subplot2grid((5, 1), (4, 0))            
    ax4.plot(mu,10*np.log10(abs(np.fft.fftshift(np.fft.fft(map_data[x,:])))**2/(abs(np.fft.fft(map_data[x,:]))**2).max()), '-o',color='black',markersize=3)
    ax4.set_ylabel('Spectrum, dB')
    ax4.set_xlim(mu.min()+10,mu.max()-10)
    ax4.set_ylim(bottom=-150)     
    
    #ax4.autoscale(enable=None, axis="x", tight=True)
#    f.colorbar(pc)
    plt.subplots_adjust(left=0.07, bottom=0.07, right=0.95, top=0.93, wspace=None, hspace=0.4)
    f.canvas.mpl_connect('button_press_event', onclick)                


"""
here is a set of useful standard functions
"""
class FieldTheoryCROW:
    def __init__(self, resonator_parameters):
   #Physical parameters initialization
       
       
        self.J = resonator_parameters['Inter-resonator_coupling']
        self.N_CROW = resonator_parameters['N_res']
        self.N_theta = resonator_parameters['N_theta']
        self.D2 = resonator_parameters['D2']
        self.kappa_ex = resonator_parameters['kappa_ex']
        self.kappa_0 = resonator_parameters['kappa_0']
        self.kappa = self.kappa_0 + self.kappa_ex
        self.d2 = self.D2/self.kappa
        self.j = 2*self.J/self.kappa
        self.N_points = resonator_parameters['Number of modes']
        self.mu = np.fft.fftshift(np.arange(-self.N_points/2, self.N_points/2))
        self.phi = np.linspace(0,2*np.pi,self.N_points)
        self.theta =np.linspace(0,2*np.pi,self.N_theta)
        self.delta_theta = 2*np.pi/self.N_CROW
    
    def seed_level (self, pump, detuning):
        f_norm = pump
        detuning_norm  = detuning
        res_seed = np.zeros_like(f_norm)
        res_seed = f_norm/(1-1j*detuning_norm)
        return res_seed
        #res_seed[0] = abs(np.min(stat_roots[ind_roots]))**.5
        
    def noise(self, a):
        return a*(np.random.uniform(-1,1,self.N_points*self.N_theta)+ 1j*np.random.uniform(-1,1,self.N_points*self.N_theta))

    def Propagate_SAMCLIB(self, simulation_parameters, Pump, Seed=[0], dt=5e-4,HardSeed=False):
        
        
        T = simulation_parameters['slow_time']
        abtol = simulation_parameters['absolute_tolerance']
        reltol = simulation_parameters['relative_tolerance']
        out_param = simulation_parameters['output']
        nmax = simulation_parameters['max_internal_steps']
        detuning = simulation_parameters['detuning_array']
        eps = simulation_parameters['noise_level']
        
        
        if HardSeed == False:
            seed = self.seed_level(Pump, detuning[0])
        else:
            seed = Seed.T.reshape(Seed.size)
        ### renormalization
        T_rn = (self.kappa/2)*T
        #f0 = np.fft.ifft(Pump,axis=0)*self.N_points
        f0=Pump
        print('f0='+str(f0.max()))
        print('xi [' + str(detuning[0]) + ',' +str(detuning[-1])+ '] (normalized on ' r'$kappa/2)$')
        noise_const = self.noise(eps) # set the noise level
        nn = len(detuning)
        
        t_st = float(T_rn)/len(detuning)
            
        sol = np.ndarray(shape=(len(detuning), self.N_points, self.N_theta), dtype='complex') # define an array to store the data
        
        ind_modes = np.arange(self.N_points)
        ind_res = np.arange(self.N_CROW)
        j = self.j
        kappa = self.kappa
        seed = seed.T.reshape(seed.size)
        for ii in range(self.N_theta):
            sol[0,ind_modes,ii] = seed[ii*self.N_points+ind_modes]
        
        f0 =(f0.T.reshape(f0.size))
        #%% crtypes defyning
        CROW_core = ctypes.CDLL(os.path.abspath(__file__)[:-15]+'/lib/lib_2D_lle_core.so')
        #else:
        #    sys.exit('Solver has not been found')
        
        CROW_core.PropagateSAM.restype = ctypes.c_void_p
        
        A = np.zeros([self.N_theta*self.N_points],dtype=complex)
        for ii in range(self.N_theta):    
        #    A[ii*self.N_points+ind_modes] = np.fft.ifft( seed[ii*self.N_points+ind_modes])*self.N_points
            A[ii*self.N_points+ind_modes] =  seed[ii*self.N_points+ind_modes]
        
    
        In_val_RE = np.array(np.real(A),dtype=ctypes.c_double)
        In_val_IM = np.array(np.imag(A),dtype=ctypes.c_double)
        In_phi = np.array(self.phi,dtype=ctypes.c_double)
        In_Nphi = ctypes.c_int(self.N_points)
        In_Ncrow = ctypes.c_int(self.N_CROW)
        In_theta = np.array(self.theta,dtype=ctypes.c_double)
        In_Ntheta = ctypes.c_int(self.N_theta)
        In_f_RE = np.array(np.real(f0 ),dtype=ctypes.c_double)
        In_f_IM = np.array(np.imag(f0 ),dtype=ctypes.c_double)
        In_atol = ctypes.c_double(abtol)
        In_rtol = ctypes.c_double(reltol)
        In_d2 = ctypes.c_double(self.d2)
        In_j = ctypes.c_double(self.j)
        In_kappa = ctypes.c_double(self.kappa)
        In_delta_theta = ctypes.c_double(self.delta_theta)

        In_det = np.array(detuning,dtype=ctypes.c_double)
        In_Ndet = ctypes.c_int(len(detuning))
        
        In_Tmax = ctypes.c_double(t_st)
        In_Nt = ctypes.c_int(int(t_st/dt)+1)
        In_dt = ctypes.c_double(dt)
        In_noise_amp = ctypes.c_double(eps)
        
        In_res_RE = np.zeros(len(detuning)*self.N_points*self.N_theta,dtype=ctypes.c_double)
        In_res_IM = np.zeros(len(detuning)*self.N_points*self.N_theta,dtype=ctypes.c_double)
        
        double_p=ctypes.POINTER(ctypes.c_double)
        In_val_RE_p = In_val_RE.ctypes.data_as(double_p)
        In_val_IM_p = In_val_IM.ctypes.data_as(double_p)
        In_phi_p = In_phi.ctypes.data_as(double_p)
        In_theta_p = In_theta.ctypes.data_as(double_p)
        In_det_p = In_det.ctypes.data_as(double_p)
        
        In_f_RE_p = In_f_RE.ctypes.data_as(double_p)
        In_f_IM_p = In_f_IM.ctypes.data_as(double_p)
        
        In_res_RE_p = In_res_RE.ctypes.data_as(double_p)
        In_res_IM_p = In_res_IM.ctypes.data_as(double_p)
        
        CROW_core.PropagateSAM(In_val_RE_p, In_val_IM_p, In_f_RE_p, In_f_IM_p, In_det_p, In_phi_p, In_theta_p, In_delta_theta, In_d2, In_j, In_Ndet, In_Nt, In_dt, In_atol, In_rtol, In_Nphi, In_Ntheta, In_noise_amp, In_res_RE_p, In_res_IM_p)
            
        
        ind_modes = np.arange(self.N_points)
        for ii in range(0,len(detuning)):
            for jj in range(self.N_theta):
                sol[ii,ind_modes,jj] = (In_res_RE[ii*self.N_points*self.N_theta + jj*self.N_points+ind_modes] + 1j*In_res_IM[ii*self.N_points*self.N_theta+ jj*self.N_points+ind_modes])
            
        #sol = np.reshape(In_res_RE,[len(detuning),self.N_points]) + 1j*np.reshape(In_res_IM,[len(detuning),self.N_points])
                    
        if out_param == 'map':
            return sol
        elif out_param == 'fin_res':
            return sol[-1, :]
        else:
            print ('wrong parameter')



if __name__ == '__main__':
    print('PyCORe')
    
