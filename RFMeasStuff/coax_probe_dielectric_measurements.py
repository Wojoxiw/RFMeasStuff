#stuff is here
from scipy.constants import c, pi, elementary_charge, k
import scipy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import skrf as rf
import miepython
from scipy.interpolate import interp1d
from scipy.special import spherical_jn, spherical_yn
import scipy
import os
import skrf as rf

folder = 'C:/Users/al8032pa/Work Folders/Documents/antenna measurements/Coaxial Probe Dielectric Measurement/' # folder holding the data folders
colors = ('tab:blue','tab:orange','tab:red','tab:purple','tab:green','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan')
e = elementary_charge
c0 = c

def timeGate(fs, s11_f, t=0e-9, w=5e-9, showPlots=True, name=''): ## time-gates the s11, to select the reflection peak corresponding to reflections from the material. t in ns
    s11_time = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(s11_f)))
    ts = np.fft.ifftshift(np.fft.fftfreq(len(s11_time), fs[1]-fs[0]))
    window = np.cos((ts-t)/w)
    window[np.argmin(np.abs(ts-t+w/2)):np.argmin(np.abs(ts-t-w/2))] = 1 # in centre of window
    window[0:np.argmin(np.abs(ts-t+w))] = 0 # outside window
    window[np.argmin(np.abs(ts-t-w)):-1] = 0 # outside window
    s11_time_windowed = s11_time*window+1e-16
    s11 = np.fft.ifftshift(np.fft.fft(np.fft.fftshift(s11_time_windowed)))
    if(showPlots):
        plt.plot(fs, 10*np.log10(np.abs(s11_f)))
        plt.title(name+' S11, freq')
        plt.show()
        plt.plot(fs, np.angle(s11_f))
        plt.title(name+' S11 angle, freq')
        plt.show()
        plt.plot(ts, 10*np.log10(np.abs(s11_time)))
        plt.plot(ts, window)
        plt.plot(ts, 10*np.log10(np.abs(s11_time_windowed)))
        plt.title(name+' S11, time')
        plt.show()
        plt.plot(fs, 10*np.log10(abs(s11)))
        plt.title(name+' S11 gated, freq')
        plt.show()
    
    return s11

def debye_eps(freqs, compound, beta=1): # based on (9)/(10) in 'Microwave Dielectric Measurements of Erythrocyte Suspensions' by J-Z. Bao, et al.
    
    #===========================================================================
    # if(compound=='water'):
    #     delta_eps = 72.56; tau = 7.37e-12; eps_inf = 4.55
    # elif(compound=='ethanol'):
    #     delta_eps = 20.57; tau = 143.18e-12; eps_inf = 4.68
    # return eps_inf + delta_eps/(1+1j*2*pi*freqs*tau)**beta
    #===========================================================================

    ## based on J. BARTHEL, K. BACHHUBER, R. BUCHNER and H. HETZENAUER DIELECTRIC SPECTRA OF SOME COMMON WATER AND LOWER ALCOHOLS, CHEMICAL PHYSICS LETTERS, Volume 165, number 4
    369-373, 1990
    if(compound=='water'):
        eps1=77.97; tau1=8.32e-12; eps2=6.18; tau2=1.02e-12; eps3=0; tau3=0; eps_inf=4.49
    elif(compound=='ethanol'):
        eps1=24.32; tau1=163e-12; eps2=4.49; tau2=8.97e-12; eps3=3.82; tau3=1.81e-12; eps_inf=2.69
    elif(compound=='1propanol'):
        eps1=20.43; tau1=329e-12; eps2=3.74; tau2=15.1e-12; eps3=3.20; tau3=2.40e-12; eps_inf=2.44
    elif(compound=='2propanol'):
        eps1=19.40; tau1=359e-12; eps2=3.47; tau2=14.5e-12; eps3=3.04; tau3=1.96e-12; eps_inf=2.42
    elif(compound=='air'):
        return freqs*0+1
            
    return eps_inf + (eps1-eps2)/(1+2j*pi*freqs*tau1) + (eps2-eps3)/(1 + 2j*pi*freqs*tau2) + (eps3-eps_inf)/(1+2j*pi*freqs*tau3)


def method(s11, s11_short, s11_open, s11_l, eps_l): # based on the appendix of 'Microwave Dielectric Measurements of Erythrocyte Suspensions' by J-Z. Bao, et al. This attempt has not produced good results.
    ## 
    A3 = s11_short
    A2 = ( -s11_open*s11_l + A3*(s11_l - eps_l*s11_open) - eps_l*s11_l*s11_open )/( s11_open-s11_l )
    A2 = ( eps_l*s11_l + eps_l*A3 + s11_l - A3*s11_l/s11_open )/( s11_l/s11_open - 1 )
    A1 = (-s11_open + A2 + A3)/s11_open
    
    eps = (A2 - A1*s11)/(s11 - A3)
    return eps

def SolveSecondOrderEquation(a, b):
    """
    Solve the second order equation z**2 + a*z + b = 0 under the
    constraint that z should be in the lower half plane and a
    continuous function.
    """
    c = (a/2.)**2 - b
    z = -a/2. + np.sqrt(-1j*c)*np.exp(1j*pi/4)
    return(z)

def method2(fs, s11, s11_water, s11_open, s11_short, s11_known, knownName): # based on Christos' master's thesis/the code he used
    """
    Perform the calibration using four standards: open, short, water,
    ethanol, assumed to correspond to the first four reflection
    coefficients. Apply the calibration to the remaining measurements
    and return the result.
    """
    
    # Then compute parameters for the calibration
    Delta = (s11_known - s11_water)*(s11_short - s11_open)/(s11_known - s11_open)/(s11_water - s11_short)
    
    #===========================================================================
    # plt.plot(fs, np.real(Delta))
    # plt.plot(fs, np.imag(Delta))
    # plt.show()
    #===========================================================================
    
    eps_air = np.ones(np.size(fs), dtype=complex)
    eps_water = debye_eps(fs, 'water')
    eps_known = debye_eps(fs, knownName)
    xi = ((1 + Delta)*eps_known - eps_water - Delta*eps_air)/(eps_water**2 + Delta*eps_air**2 - (1 + Delta)*eps_known**2)
    y_water = eps_water + xi*eps_water**2
    y_air = eps_air + xi*eps_air**2
    
    n = 6
    Delta_sample = (s11 - s11_water)*(s11_short - s11_open)/(s11 - s11_open)/(s11_water - s11_short)
    y_sample = (y_water + Delta_sample*y_air)/(1 + Delta_sample)
    eps = SolveSecondOrderEquation(1/xi, -y_sample/xi)

    return eps

def method2Simple(fs, s11_test, s11_water, s11_open, s11_short, s11_known, knownName):
    """
    As above, but based on Christos' 'CalibrateSimple'.
    """
    Zc = 50.

    Z = lambda s11: Zc*(1 + s11)/(1 - s11)
    eps_water = debye_eps(fs, 'water')
    eps_known = debye_eps(fs, knownName)
    
    # Then compute parameters for the calibration
    B_over_D = Z(s11_short)
    A_over_C = Z(s11_open)
    D_over_C_over_Z2_water = (Z(s11_water) - A_over_C)/(B_over_D - Z(s11_water))
    D_over_C_over_Z2_ethanol = (Z(s11_known)-A_over_C)/(B_over_D-Z(s11_known))

    # Finally perform the calibration
    epsilon1 = eps_water/(D_over_C_over_Z2_water*(B_over_D - Z(s11_test))/(Z(s11_test) - A_over_C))
    epsilon2 = eps_known/(D_over_C_over_Z2_ethanol*(B_over_D - Z(s11_test))/(Z(s11_test) - A_over_C))

    return epsilon1, epsilon2

if __name__ == '__main__':
    begin=True # so I can de-indent
plt.rc('axes', titlesize=27)     # fontsize of the axes title
plt.rc('axes', labelsize=27)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=18)    # fontsize of the tick labels
plt.rc('ytick', labelsize=18)    # fontsize of the tick labels
plt.rc('legend', fontsize=12)    # legend fontsize
plt.rc('figure', titlesize=30)  # fontsize of the figure title


fileLoc = folder+'coaxanalysis (previous work by Christos)/data/'

sname = 'christos_short2'
short = np.loadtxt(fileLoc+sname+'/S11 data/'+sname+'_S11_1', skiprows=8, delimiter=',')
short = [short[:, 0], short[:, 1] + 1j*short[:, 2]]

sname = 'christos_open2'
openm = np.loadtxt(fileLoc+sname+'/S11 data/'+sname+'_S11_1', skiprows=8, delimiter=',')
openm = [openm[:, 0], openm[:, 1] + 1j*openm[:, 2]]

sname = 'christos_open3'
openm2 = np.loadtxt(fileLoc+sname+'/S11 data/'+sname+'_S11_1', skiprows=8, delimiter=',')
openm2 = [openm2[:, 0], openm2[:, 1] + 1j*openm2[:, 2]]

sname = 'christos_ethanol2'
ethanol = np.loadtxt(fileLoc+sname+'/S11 data/'+sname+'_S11_1', skiprows=8, delimiter=',')
ethanol = [ethanol[:, 0], ethanol[:, 1] + 1j*ethanol[:, 2]]

sname = 'christos_water2'
water = np.loadtxt(fileLoc+sname+'/S11 data/'+sname+'_S11_1', skiprows=8, delimiter=',')
water = [water[:, 0], water[:, 1] + 1j*water[:, 2]]

sname = 'christos_isopropylalkohol2'
isopropanol = np.loadtxt(fileLoc+sname+'/S11 data/'+sname+'_S11_1', skiprows=8, delimiter=',')
isopropanol = [isopropanol[:, 0], isopropanol[:, 1] + 1j*isopropanol[:, 2]]

sname = 'christos_solid_woodA'
wood = np.loadtxt(fileLoc+sname+'/S11 data/'+sname+'_S11_1', skiprows=8, delimiter=',')
wood = [wood[:, 0], wood[:, 1] + 1j*wood[:, 2]]

#===============================================================================
# water[1] = timeGate(water[0], water[1])
# short[1] = timeGate(short[0], short[1])
# openm[1] = timeGate(short[0], openm[1])
# ethanol[1] = timeGate(short[0], ethanol[1])
# isopropanol[1] = timeGate(short[0], isopropanol[1])
#===============================================================================

#===============================================================================
# #eps = method(isopropanol[1], short[1], openm[1], ethanol[1], debye_eps(water[0], 'ethanol'))
# eps = method2(water[0], isopropanol[1], water[1], openm[1], short[1], ethanol[1], 'ethanol')
# #eps = method2Simple(water[0], isopropanol[1], water[1], openm[1], short[1], ethanol[1], 'ethanol')[1]
#    
# fig = plt.figure()
# ax1 = plt.subplot(1, 1, 1)
# ref='1propanol'
#    
# ax1.plot(water[0]/1e9, np.real(eps), label=ref+', calculated', color='tab:green', linewidth=2.5)
# ax1.plot(water[0]/1e9, -np.imag(eps), linestyle='--', color='tab:green', linewidth=2.5)
# fs = np.linspace(0, 22e9, 5000)
# ax1.plot(fs/1e9, np.real(debye_eps(fs, ref)), label = ref+', ref.', color='tab:blue', linewidth=2.5)
# ax1.plot(fs/1e9, -np.imag(debye_eps(fs, ref)), linestyle='--', color='tab:blue', linewidth=2.5)
#===============================================================================


#fileLoc = folder+'coax probe measurements/'
#fileLoc = folder+'coax probe measurements day 2/calibrated/'
#fileLoc = folder+'coax probe measurements day 2/uncalibrated/'
#fileLoc = folder+'coax probe measurements day 2 b/'
fileLoc = folder+'coax probe measurements day 3/short cable/'
#fileLoc = folder+'coax probe measurements day 3/long cable/'
#fileLoc = folder+'coax probe measurements day 3/long cable2/'
##measurements taken with the R&S VNA that goes up to 13.6 GHz, 1000 pts, 1kHz IF BW, -10dBm power
freqs = np.transpose(np.loadtxt(fileLoc+'open1.s1p', skiprows=5))[0]

meas = {} # 'load in' all the files
for root, dirs, files in os.walk(fileLoc):
    for file in files:
        if(file.endswith('.s1p')):
            name = file[0:-4]
            print(f'Loading in {name}')
            dat = np.transpose(np.loadtxt(fileLoc+file, skiprows=5))
            s11 = 10**(dat[1]/20)*np.exp(1j*dat[2]*pi/180)
            #s11 = rf.Network(fileLoc+file).s11.s.flatten()
            if(True): ## time-gate the data
                s11 = timeGate(freqs, s11, showPlots=False, name=name)
            meas = meas | {name: s11} ## the name



ref = '' ## to plot reference curves
eps = method2(freqs, meas['2propanol1'], meas['water1'], meas['open1'], meas['short1'], meas['ethanol1'], 'ethanol')
#eps = method2Simple(freqs, meas['ethanol1'], meas['water2'], meas['open2'], meas['short2'], meas['2propanol2'], '2propanol')[0]

fig = plt.figure()
ax1 = plt.subplot(1, 1, 1)
   

if(ref != ''):
    ax1.plot(freqs/1e9, np.real(eps), label=ref+', calculated', color='tab:green', linewidth=2.5)
    ax1.plot(freqs/1e9, -np.imag(eps), linestyle='--', color='tab:green', linewidth=2.5)
    fs = np.linspace(0, 22e9, 5000)
    ax1.plot(fs/1e9, np.real(debye_eps(fs, ref)), label = ref+', ref.', color='tab:blue', linewidth=2.5)
    ax1.plot(fs/1e9, -np.imag(debye_eps(fs, ref)), linestyle='--', color='tab:blue', linewidth=2.5)
    plt.xlim(0, 22)
else:
    i=-1
    for sample in ['heavycyl1', 'lightcyl1', 'medcyl1', 'johan1_1', 'johan3_1', 'plexiglass1']:#['heavycyl1', 'lightcyl3', 'medcyl3', 'johan1_2', 'johan3_1']:#['heavycyl', 'lightcyl', 'medcyl', 'johan1', 'johan3']:
        i+=1
        eps = method2(freqs, meas[sample], meas['water1'], meas['open1'], meas['short1'], meas['ethanol1'], 'ethanol')
        ax1.plot(freqs/1e9, np.real(eps), label=sample+', measured', color=colors[i%len(colors)], linewidth=2.5)
        ax1.plot(freqs/1e9, -np.imag(eps), linestyle='--', color=colors[i%len(colors)], linewidth=2.5)
    plt.xlim(0, 14)

first_legend = ax1.legend(framealpha=0.5, ncol=1, loc = 'upper left')
##second legend to distinguish between dashed and regular lines (real and imaginary parts)
handleds = []
line_dashed = mlines.Line2D([], [], color='black', linestyle='solid', linewidth=2.5, label=r'Real$(\epsilon_r)$') ##fake lines to create second legend elements
handleds.append(line_dashed)
line_solid = mlines.Line2D([], [], color='black', linestyle='--', linewidth=2.5, label=r'Imag$(\epsilon_r)$') ##fake lines to create second legend elements
handleds.append(line_solid)
    
second_legend = ax1.legend(handles=handleds, loc='upper right', framealpha=0.5)
ax1.add_artist(first_legend)
ax1.add_artist(second_legend)

plt.ylabel(r'$\epsilon_r$')
plt.xlabel('Frequency [GHz]')
plt.title('Permittivities')

plt.ylim(-0.5, 20)

plt.tight_layout()
plt.grid()
plt.show()
