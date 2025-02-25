'''
Created on 11 okt. 2021

@author: al8032pa
'''

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants
import os
from math import pi


def plotLines():
    c = scipy.constants.c
    filenames = ['C:/Users/al8032pa/Work Folders/Documents/Feko Simulations/Scripts/Data/Monostatic Stuff/uncloakedRI.txt',
                 'C:/Users/al8032pa/Work Folders/Documents/Feko Simulations/Scripts/Data/Monostatic Stuff/paperpts_GP_try8-c_Head-on_smallermeshRI.txt',
             #'C:/Users/al8032pa/Work Folders/Documents/Feko Simulations/Scripts/Data/Monostatic Stuff/GP_try8-cRI.txt',
             'C:/Users/al8032pa/Work Folders/Documents/Feko Simulations/Scripts/Data/Monostatic Stuff/shortuncloakedRI.txt',
             'C:/Users/al8032pa/Work Folders/Documents/Feko Simulations/Scripts/Data/Monostatic Stuff/GP_try7_optimum_optimum.Monos.FarField1.txt',
             'C:/Users/al8032pa/Work Folders/Documents/Feko Simulations/Scripts/Data/Monostatic Stuff/GP_try8-c_Head-on freqrangewithportRI.txt',
             'C:/Users/al8032pa/Work Folders/Documents/Feko Simulations/Scripts/Data/Monostatic Stuff/GP_try8-3c_alttofitmanufacturingactual_smallermesh_mono+notRI.txt',
             'C:/Users/al8032pa/Work Folders/Documents/Feko Simulations/Scripts/Data/Monostatic Stuff/GP_try8-3c_alttofitmanufacturingactual_mono_smallermesh_withportloadRI.txt',
             #'C:/Users/al8032pa/Work Folders/Documents/Feko Simulations/Scripts/Data/Monostatic Stuff/GP_try8-3c_alttofitmanufacturingactualRI.txt',
            ]
    uncloakedSim = np.transpose(np.loadtxt(filenames[0], delimiter = ',', dtype = complex, skiprows = 3))
    cloakedSim = np.transpose(np.loadtxt(filenames[1], delimiter = ',', dtype = complex, skiprows = 3))
    suncloakedSim = np.transpose(np.loadtxt(filenames[2], delimiter = ',', dtype = complex, skiprows = 3))
    lcloakedSim = np.transpose(np.loadtxt(filenames[3], delimiter = ',', dtype = complex, skiprows = 3))
    cloakedSimwport = np.transpose(np.loadtxt(filenames[4], delimiter = ',', dtype = complex, skiprows = 3))
    cloakedSimAlt = np.transpose(np.loadtxt(filenames[5], delimiter = ',', dtype = complex, skiprows = 3))
    cloakedSimAltLoad = np.transpose(np.loadtxt(filenames[6], delimiter = ',', dtype = complex, skiprows = 3))
    
    plt.plot(cloakedSimAlt[0]/1e9, 10*np.log10(np.abs((1/c*cloakedSimAlt[0]*(2*pi)*cloakedSimAlt[1]/cloakedSimAlt[0]*2*c)**2)/12.566370614359), color = 'tab:purple', linestyle = '--', label = 'Prototype Cloaked')
    plt.plot(cloakedSim[0]/1e9, 10*np.log10(np.abs((1/c*cloakedSim[0]*(2*pi)*cloakedSim[1]/cloakedSim[0]*2*c)**2)/12.566370614359), color = 'tab:gray', linestyle = ':', label = 'First Final Design sim.', linewidth = 2.1)
    plt.plot(uncloakedSim[0]/1e9, 10*np.log10(np.abs((1/c*uncloakedSim[0]*(2*pi)*uncloakedSim[1]/uncloakedSim[0]*2*c)**2)/12.566370614359), color = 'tab:red', linestyle = '--', label = 'Uncloaked')
    #plt.plot(cloakedSimAltLoad[0]/1e9, 10*np.log10(np.abs((1/c*cloakedSimAltLoad[0]*(2*pi)*cloakedSimAltLoad[1]/cloakedSimAltLoad[0]*2*c)**2)/12.566370614359), label = 'Short Cloaked Simulation - Alt Dims+load', color = 'tab:purple', linestyle = ':', linewidth = 2.1)
    plt.plot(suncloakedSim[0]/1e9, 10*np.log10(np.abs((1/c*suncloakedSim[0]*(2*pi)*suncloakedSim[1]/uncloakedSim[0]*2*c)**2)/12.566370614359), color = 'tab:blue', linestyle = '--', label = 'Short Uncloaked')
    plt.plot(lcloakedSim[0]/1e9, 10*np.log10(np.abs((1/c*lcloakedSim[0]*(2*pi)*lcloakedSim[1]/lcloakedSim[0]*2*c)**2)/12.566370614359), color = 'tab:orange', linestyle = '--')#, label = 'Long Cloaked Simulation'
    #plt.plot(cloakedSimwport[0]/1e9, 10*np.log10(np.abs((1/c*cloakedSimwport[0]*(2*pi)*cloakedSimwport[1]/cloakedSimwport[0]*2*c)**2)/12.566370614359), label = 'Short Cloaked Simulation with 50 $\Omega$ port load', color = 'tab:green', linestyle = '--')
    
if __name__ == '__main__':
    pass
    #plotLines()
    #plt.show()