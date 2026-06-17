from pylab import *

# Set up defaults for plotting
from matplotlib import rc
rc('font', family='serif', size=20)
rc('text', usetex=True)
rc('lines', linewidth=2)
rc('legend', fancybox=True, shadow=True, fontsize=16, loc='best')
rc('axes', grid=True)

c0 = 299792458.
mu0 = 4*pi*1e-7
eps0 = 1/c0**2/mu0
eta0 = sqrt(mu0/eps0)

datadir = 'C:/Users/al8032pa/Work Folders/Documents/antenna measurements/Coaxial Probe Dielectric Measurement/coaxanalysis (previous work by Christos)/data/'

names_short = ['christos_short1',
               'christos_short2',
               'christos_short3',
               'christos_short4']
names_open = ['christos_open',
              'christos_open2',
              'christos_open3']
names_water = ['christos_water1',
               'christos_water2',
               'christos_water3']
names_ethanol = ['christos_ethanol1',
                 'christos_ethanol2',
                 'christos_ethanol3']
names_propyl = ['christos_isopropylalkohol1',
                'christos_isopropylalkohol2',
                'christos_isopropylalkohol3']
names_saline = ['christos_saline1',
                'christos_saline2',
                'christos_saline3']


# Sequences of calibration and samples for evaluation
n_open = 0     # Index of open standard
n_short = 1    # Index of short standard
n_water = 2    # Index of water standard
n_ethanol = 3  # Index of ethanol standard
names_naked = ['christos_open',
               'christos_short1',
               'christos_water1',
               'christos_ethanol1',
               'christos_open2',
               'christos_open3',
               'christos_water2',
               'christos_water3',
               'christos_ethanol2',
               'christos_ethanol3',
               'christos_saline1',
               'christos_saline2',
               'christos_saline3',
               'christos_isopropylalkohol1',
               'christos_isopropylalkohol2',
               'christos_isopropylalkohol3',
               'christos_foamB',
               'christos_rubber']
names_plastic = ['christos_solid_open',
                 'christos_solid_short',
                 'christos_solid_water',
                 'christos_solid_ethanol',
                 'christos_solid_foamA',
                 'christos_solid_foamB',
                 'christos_solid_woodA',
                 'christos_solid_metal']
names_plastic2 = ['christos_open',
                  'christos_short1',
                  'christos_water1',
                  'christos_ethanol1',
                  'christos_solid_foamA',
                  'christos_solid_foamB',
                  'christos_solid_woodA',
                  'christos_solid_metal']


def SolveSecondOrderEquation(a, b):
    """
    Solve the second order equation z**2 + a*z + b = 0 under the
    constraint that z should be in the lower half plane and a
    continuous function.
    """
    c = (a/2.)**2 - b
    z1 = -a/2. + sqrt(c)
    z2 = -a/2. - sqrt(c)
    z = -a/2. + sqrt(-1j*c)*exp(1j*pi/4)
    return(z)
    

def ReadData(filename):
    data = loadtxt(filename, skiprows=8, delimiter=',')
    f = data[:,0]
    s11 = data[:,1] + 1j*data[:,2]
    return(f, s11)

def PlotS11(names, titlestr=None):
    figure()
    for n, name in enumerate(names):
        filename = datadir + name + '/S11 data/' + name + '_S11_1'
        f, s11 = ReadData(filename)
        subplot(2,1,1)
        plot(f*1e-9, 20*log10(abs(s11)), label='{0}'.format(n))
        subplot(2,1,2)
        plot(f*1e-9, 180/pi*angle(s11))
    subplot(2,1,2)
    xlabel('Frequency (GHz)')
    subplot(2,1,1)
    legend(loc='best')
    if titlestr:
        title(titlestr)
    return()

def PlotAllParameters():
    PlotS11(names_short, titlestr='short')
    PlotS11(names_open, titlestr='open')
    PlotS11(names_water, titlestr='water')
    PlotS11(names_ethanol, titlestr='ethanol')
    PlotS11(names_propyl, titlestr='isopropylalkohol')
    PlotS11(names_propyl, titlestr='saline')
    PlotS11(names_plastic, titlestr='plastic')
    PlotS11(names_naked, titlestr='naked')
    show()


def PlotCalibratedParameters(name_sample, name_open=names_open[0], 
                             name_short=names_short[0]):
    name = name_open
    filename = datadir + name + '/S11 data/' + name + '_S11_1'
    f, s11_open = ReadData(filename)
    name = name_short
    filename = datadir + name + '/S11 data/' + name + '_S11_1'
    f, s11_short = ReadData(filename)
    name = name_sample
    filename = datadir + name + '/S11 data/' + name + '_S11_1'
    f, s11_sample = ReadData(filename)

    delay = (s11_open - s11_short)/2

    s11_cal = s11_sample/delay
    figure()
    subplot(2,1,1)
    plot(f*1e-9, 20*log10(abs(s11_cal)))
    subplot(2,1,2)
    plot(f*1e-9, 180/pi*unwrap(angle(s11_cal)))
    xlabel('Frequency (GHz)')
    show()

def PermittivityWater(f, T=298):
    """
    Permittivity data from 
    U. Kaatze, "Complex permittivity of water as a function of
    frequency and temperature," Journal of Chemical & Engineering
    Data, vol. 34, no. 4, pp. 371�374, 1989.
    """
    eps_s = 10**(1.94404 - 1.991e-3*(T - 273.15))
    eps_inf = 5.77 - 2.74e-2*(T - 273.15)
    tau = 3.745e-15*(1 + 7e-5*(T - 300.65)**2)*exp(2.2957e3/T)
    epsilon = eps_inf + (eps_s - eps_inf)/(1 + 2j*pi*f*tau)
    return(epsilon)

def PermittivityBarthel(f, substance='ethanol'):
    """
    Permittivity data from 
    J. BARTHEL, K. BACHHUBER, R. BUCHNER and H. HETZENAUER
    DIELECTRIC SPECTRA OF SOME COMMON WATER AND LOWER ALCOHOLS
    CHEMICAL PHYSICS LETTERS
    Volume 165, number 4
    369-373, 1990
    """
    data = {'water': [77.97, 8.32e-12, 6.18, 1.02e-12, 0, 0, 4.49],
            'methanol': [32.50, 51.5e-12, 5.91, 7.09e-12, 4.90, 1.12e-12, 2.79],
            'ethanol': [24.32, 163e-12, 4.49, 8.97e-12, 3.82, 1.81e-12, 2.69],
            '1-propanol': [20.43, 329e-12, 3.74, 15.1e-12, 3.20, 2.40e-12,2.44],
            '2-propanol': [19.40, 359e-12, 3.47, 14.5e-12, 3.04, 1.96e-12,2.42]}
    eps1, tau1, eps2, tau2, eps3, tau3, eps_inf = data[substance]
    s = 2j*pi*f
    if substance == 'water':
        epsilon = eps_inf + (eps1 - eps2)/(1 + s*tau1) \
            + (eps2 - eps_inf)/(1 + s*tau2) 
    else:
        epsilon = eps_inf + (eps1 - eps2)/(1 + s*tau1) \
            + (eps2 - eps3)/(1 + s*tau2) \
            + (eps3 - eps_inf)/(1 + s*tau3)
    return(epsilon)

def Calibrate(measurements):
    """
    Perform the calibration using four standards: open, short, water,
    ethanol, assumed to correspond to the first four reflection
    coefficients. Apply the calibration to the remaining measurements
    and return the result.
    """
    # First input all the data
    data = []
    for name in measurements:
        filename = datadir + name + '/S11 data/' + name + '_S11_1'
        f, s11 = ReadData(filename)
        data.append(s11)
    Gamma = array(data)
#    delay = (Gamma[0] - Gamma[1])/2    
#    for n in range(0, len(measurements)):
#        Gamma[n] = Gamma[n]/delay
    
    
    # Then compute parameters for the calibration
    Delta = (Gamma[n_ethanol] - Gamma[n_water])*(Gamma[n_short] - Gamma[n_open])/(Gamma[n_ethanol] - Gamma[n_open])/(Gamma[n_water] - Gamma[n_short])
    eps_air = ones(len(f), dtype=complex)
    eps_water = PermittivityWater(f)
    eps_ethanol = PermittivityBarthel(f)
    xi = ((1 + Delta)*eps_ethanol - eps_water - Delta*eps_air)/(eps_water**2 + Delta*eps_air**2 - (1 + Delta)*eps_ethanol**2)
    y_water = eps_water + xi*eps_water**2
    y_air = eps_air + xi*eps_air**2

    def plotcomplex(x, y, filename, label):
        figure()
        plot(x, real(y), label=r'$\mathrm{Re}(' + label + r')$')
        plot(x, imag(y), label=r'$\mathrm{Im}(' + label + r')$')
        grid(True)
        xlabel('Frequency (GHz)')
        legend(loc='best')
#        savefig(filename)
#        close()
        show()
#    plotcomplex(f*1e-9, Delta, 'Delta.pdf', r'\Delta')
#    plotcomplex(f*1e-9, xi, 'xi.pdf', r'\xi')
#    plotcomplex(f*1e-9, y_water, 'ywater.pdf', r'y_{\mathrm{water}}')
#    plotcomplex(f*1e-9, y_air, 'yair.pdf', r'y_{\mathrm{air}}')
#    exit()
    n = 6
    Delta_sample = (Gamma[n] - Gamma[n_water])*(Gamma[n_short] - Gamma[n_open])/(Gamma[n] - Gamma[n_open])/(Gamma[n_water] - Gamma[n_short])
    y_sample = (y_water + Delta_sample*y_air)/(1 + Delta_sample)
    epsilon = SolveSecondOrderEquation(1/xi, -y_sample/xi)
#    plotcomplex(f*1e-9, y_sample, 'water2.pdf', r'y_{\mathrm{water}}')
#    exit()

    # Finally perform the calibration
    permittivities = []
    for n in range(0, len(measurements)):
        Delta_sample = (Gamma[n] - Gamma[n_water])*(Gamma[n_short] - Gamma[n_open])/(Gamma[n] - Gamma[n_open])/(Gamma[n_water] - Gamma[n_short])
        y_sample = (y_water + Delta_sample*y_air)/(1 + Delta_sample)
        epsilon = SolveSecondOrderEquation(1/xi, -y_sample/xi)
        permittivities.append(epsilon)
    return(f, array(permittivities))

def CalibrateSimple(measurements):
    """
    Perform the calibration using four standards: open, short, water,
    ethanol, assumed to correspond to the first four reflection
    coefficients. Apply the calibration to the remaining measurements
    and return the result.
    """
    Zc = 50.

    # First input all the data
    data = []
    for name in measurements:
        filename = datadir + name + '/S11 data/' + name + '_S11_1'
        f, s11 = ReadData(filename)
        data.append(s11)
    Gamma = array(data)
    Z = Zc*(1 + Gamma)/(1 - Gamma)
    epsilon_water = PermittivityWater(f)
    epsilon_ethanol = PermittivityBarthel(f)
    
    # Then compute parameters for the calibration
    B_over_D = Z[n_short]
    A_over_C = Z[n_open]
    D_over_C_over_Z2_water = (Z[n_water] - A_over_C)/(B_over_D - Z[n_water])
    D_over_C_over_Z2_ethanol = (Z[n_ethanol]-A_over_C)/(B_over_D-Z[n_ethanol])

    # Finally perform the calibration
    permittivities = []
    for n in range(0, len(measurements)):
        epsilon1 = epsilon_water/(D_over_C_over_Z2_water*(B_over_D - Z[n])/(Z[n] - A_over_C))
        epsilon2 = epsilon_ethanol/(D_over_C_over_Z2_ethanol*(B_over_D - Z[n])/(Z[n] - A_over_C))

        permittivities.append(epsilon1)
    return(f, array(permittivities))

    

if __name__ == '__main__':
#    PlotAllParameters()
#    PlotCalibratedParameters(names_naked[-2])

    case = 'naked'
    if case == 'naked':
        measurements = names_naked
        dropchars = 9
        fileend = '_naked.pdf'
    elif case == 'plastic':
        measurements = names_plastic
        dropchars = 15
        fileend = '_plastic.pdf'
    else:
        measurements = names_plastic2
        dropchars = 15
        fileend = '_plastic2.pdf'

    f, permittivities = Calibrate(measurements)
    f, permittivities2 = CalibrateSimple(measurements)
    for n, epsilon in enumerate(permittivities):
        figure()
        plot(f*1e-9, real(epsilon), 'b-', label=r'$\mathrm{Re}(\epsilon)$')
        plot(f*1e-9, -imag(epsilon), 'r-', label=r'$-\mathrm{Im}(\epsilon)$')
#        plot(f*1e-9, real(permittivities2[n]), 'b--', label=r'$\mathrm{Re}(\epsilon_2)$')
#        plot(f*1e-9, -imag(permittivities2[n]), 'r--', label=r'$-\mathrm{Im}(\epsilon_2)$')
        xlabel('Frequency (GHz)')
        title(measurements[n][dropchars:])
        ymax = max(real(epsilon[f>1e9]))
        #ylim(-ymax*0.1, ymax*1.1)
        grid(True)
        legend(loc='best')
#        filename = 'results/' + measurements[n][dropchars:] + fileend
#        savefig(filename)

    figure()
    epsilon1 = permittivities[2] # Water1
    epsilon2 = permittivities[6] # Water2
    error = (epsilon1-epsilon2)/(epsilon1+epsilon2)*2
    plot(f*1e-9, abs(error), 'b-', label=r'$\mathrm{abs}((\epsilon_1-\epsilon_2)/(\epsilon_1+\epsilon_2)*2)$')
    legend(loc='best')
    show()
