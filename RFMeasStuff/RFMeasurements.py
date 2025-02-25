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
import plotMonostaticSim


e = elementary_charge
c0 = c

driftCorrPreTG = False


# Set up various spherical bessel derivatives etc, used for theoretical calculation of sphere stuff
spherical_jnp = lambda n, z: spherical_jn(n,z, derivative=True)
spherical_ynp = lambda n, z: spherical_yn(n,z, derivative=True)
spherical_hn = lambda n, z: spherical_jn(n, z) - 1j*spherical_yn(n, z)
spherical_hnp = lambda n, z: spherical_jnp(n, z) - 1j*spherical_ynp(n, z)
spherical_xjnp = lambda n, z: spherical_jn(n, z) + z*spherical_jnp(n, z)
spherical_xhnp = lambda n, z: spherical_hn(n, z) + z*spherical_hnp(n, z)

def plot(ps, showGate = None):
    plt.figure(figsize=(9,4.5))
    for p in ps:
        #plt.figure(figsize=(8,8))
        plt.subplot(121)
        p.plot_s_db()
        plt.title('Frequency Domain')
        
        plt.subplot(122)
        p.plot_s_time_db()
        if(showGate is not None):
            b = p.time_gate(showGate, return_all = True)
            b['gate'].plot_s_time_db
            #p['gate']
        plt.title('Time Domain')
        
        #=======================================================================
        # plt.subplot(223)
        # p.windowed().plot_s_db()
        # plt.title('Windowed Frequency Domain')
        # 
        # plt.subplot(224)
        # p.plot_s_db_time()
        # plt.title('Windowed Time Domain')
        #=======================================================================
        
        plt.tight_layout()
    plt.show()
    
def driftComp(interpPhi, interpM, data, rawData, tbM, tbFM, BGs, nBGs, monostatic): ##compensates for drift given the interpolating function, padded+gated S data, raw data and other stuff to find time of meas.
    ###first, find the time of the measurements
    t = -999999 ##'average time' of the measurement, used for the interpolation
    if(monostatic): ##can't change list into S21 or S11 in one line, so doing an if statement...
        for i in range(len(BGs)): ##check for time if it is a background
            if(rawData == BGs[i].s11.s.squeeze()).all():
                t = tbM*( i*2 )
        
        for i in range(len(nBGs)): ##check for time if it is not a background
            if(rawData == nBGs[i].s11.s.squeeze()).all():
                t = tbM*( i*2 - 1 ) ### i*2 - 1 because the interpolation was made with t=0 being the first background measurement
    else:
        for i in range(len(BGs)): ##check for time if it is a background
            if(rawData == BGs[i].s21.s.squeeze()).all():
                t = tbM*( i*2 )
        
        for i in range(len(nBGs)): ##check for time if it is not a background
            if(rawData == nBGs[i].s21.s.squeeze()).all():
                t = tbM*( i*2 - 1 )
                
                
    if(t==-999999): ##didn't find a match
        raise ValueError('Didn\'t find a match to data when trying to.')
    #print('interp at t='+str(t))
    
    ###once time is found, correct for drift using the interpolation
    returnData = np.zeros(len(data), dtype=complex)
    for i in range(len(returnData)):
        interpChanges = np.exp(-1j*interpPhi[i](t+i*tbFM))#/(1+interpM[i](t+i*tbFM)) ##entire frequency spectrum, different per time
        returnData[i] = data[i]*interpChanges
    
    return returnData #returnData ##the drift compensated data at time t
    
def plotBackgroundSubtractionStats(data, tgS21, tgS11, timePassed): ##ata is the set of backgrounds - tg is the time-gating -- timePassed is the time between first and last measurement, for calibration purposes
    freqRange = '6.6-15ghz'
    font = {'size'   : 20}
    plt.rc('font', **font)
    labelFontSize = 23
    titleFontSize = 25
    legendFontSize = 8
    figSize = (14,8)
    
    print('Time between consecutive measurements [s]: '+str(timePassed/((len(data)-1)*2)))
    
    colors = ('tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan', 'black')
    markers = ('o','.',',','<','^','h','^','+','d','|')
    fs = data[0][freqRange].frequency.f ## the frequency data
    data21 = [] ##list of S-21s, in order
    data11 = [] ##list of S-11s, in order
    subS21 = [] ##list of absolute values of array of subtractions at one 'distance', for each background. Different lengths means these can't be arrays...
    subS11 = [] ##list of absolute values of array of subtractions at one 'distance', for each background. Different lengths means these can't be arrays...
    phaseDsS21 = [] ##list of array of phase differences at one 'distance', for each background
    phaseDsS11 = [] ##list of array of phase differences at one 'distance', for each background
    magDsS21 = [] ##list of array of magnitude differences at one 'distance', for each background
    magDsS11 = [] ##list of array of magnitude differences at one 'distance', for each background
    
    for datum in data:
        if(driftCorrPreTG):
            data21.append(datum[freqRange].s21.s.squeeze()) ## non-time-dated freq-domain transmission
            data11.append(datum[freqRange].s11.s.squeeze())## non-time-dated freq-domain reflection
        else:
            data21.append(tplusgate((fs, datum[freqRange].s21.s.squeeze()), tgS21[0][1], tgS21[1], tgS21[2], tgS21[0][0])[2][0:len(fs)]) ## padded, time-gated freq-domain transmission
            data11.append(tplusgate((fs, datum[freqRange].s11.s.squeeze()), tgS11[0][1], tgS11[1], tgS11[2], tgS11[0][0])[2][0:len(fs)])## padded, time-gated freq-domain reflection
        
        
    sameMagData = [] ##compensating for magnitude changes - setting all equal to the first
    samePhaseData = [] ##compensating for phase changes - setting all equal to the first
    for h in range(len(data21)):
        sameMagData.append(data21[h]*( np.abs(data21[0])/np.abs(data21[h]) ))
        deltaPhi = np.angle(data21[h])-np.angle(data21[0])
        samePhaseData.append(data21[h]*( np.exp(-1j*deltaPhi) ))
    #data21 = sameMagData ## to compensate for magnitude
    #data21 = samePhaseData ## to compensate for phase
    
    for i in range(len(data)): ##iterates over each background, to get 'distance'
        if(i == 0):
            continue ##skip the 'zero-distance' one
        S21dist = []; S11dist = []; phaseS21dist = []; phaseS11dist = []; magS21dist = []; magS11dist = [] ##all subtractions at i 'distance'
        for j in range(len(data)): ##iterates over each background again, to get each one in relation with each other one
            k = j+i## subtract only k>j from j, to avoid duplicates
            if(k >= 0 and k <= (len(data)-1)): ##ensure that it is actually in the set
                ##simple subtractions
                S21dist.append(np.abs(data21[k]-data21[j]))
                S11dist.append(np.abs(data11[k]-data11[j]))
                #or phase differences between the measurements, can use the assumption of exponential phase dependence, or just take np.angle
                ### VNA has phase decreasing as frequency/distance increase, due to their convention, so:
                #phaseS21dist.append(np.imag((data21[j]-data21[k])/data21[j]))
                #phaseS11dist.append(np.imag((data11[j]-data11[k])/data11[j]))
                phaseS21dist.append(-np.unwrap(np.angle(data21[k])-np.angle(data21[j]))) ##needs a negative sign because of the VNA convention
                phaseS11dist.append(-np.unwrap(np.angle(data11[k])-np.angle(data11[j]))) ##needs a negative sign because of the VNA convention
                #or magnitude differences between the measurements
                magS21dist.append((np.abs(data21[k])-np.abs(data21[j]))/np.abs(data21[j]))
                magS11dist.append((np.abs(data11[k])-np.abs(data11[j]))/np.abs(data11[j]))
                ## these next two are just to plot after compensating for 1 change
                #magS21dist.append((np.abs(data21[k]-data21[j]))/np.abs(data21[j]))
                #magS11dist.append((np.abs(data21[k]-data21[j]))/np.abs(data11[j]))
                    
        subS21.append(np.array(S21dist)); subS11.append(np.array(S11dist))
        phaseDsS21.append(np.array(phaseS21dist)); phaseDsS11.append(np.array(phaseS11dist))
        magDsS21.append(np.array(magS21dist)); magDsS11.append(np.array(magS11dist))
        
    
    #===========================================================================
    # plt.plot(fs, 10*np.log10(overallmeanS21), marker = '.', label = 'Mean S21 Subtraction')
    # plt.plot(fs, 10*np.log10(overallmeanS11), marker = '.', label = 'Mean S21 Subtraction')
    # plt.legend()
    # plt.show()
    #===========================================================================
    
    ###here, cut off outside on either side because Fourier transform distorts that data
    cutOff = 0.1
    npoints = len(fs) #- number of pts we have
    ct1 = round(npoints*cutOff) ##first cut-off index
    ct2 = round(npoints*(1-cutOff))## second cut-off index
    
    
    ########
    ####Compensation stuff
    ########
    colors = ('tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan')
    tbB = timePassed/(len(data)-1)##time between background-measurements
    tMeas = 45 ##estimated single measurement sweep time
    tbFM = tMeas/len(fs) ##time between frequency-points in one measurement, estimated as: (meas. sweep time)/(# of pts per sweep)
    bGMTs = np.arange(len(data))*tbB ##times at which the background measurements begin (approx.)
    
    ##the interpolations, per frequency point
    
    #===========================================================================
    # tPoints = np.zeros(len(data)*len(fs))
    # for i in range(len(data)):
    #     for j in range(len(fs)):
    #         tPoints[i*len(data)+j] = tbB*i + tbFM*j
    #         
    # print(type(fs))
    # print(np.shape(-np.unwrap(np.angle(data21[:])-np.angle(data21[0]), axis = 1)))
    #===========================================================================
    
    interperPhiS21b = [] ## interp[500](Times)[:] = interpolation at times for frequency pt. 500
    interperPhiS11b = []
    interperMS21b = []
    interperMS11b = []
    for h in range(len(fs)): ##creating a list of an interpolater for each frequency
        interperPhiS21b.append(scipy.interpolate.CubicSpline(bGMTs+h*tbFM, (-np.unwrap(np.angle(np.transpose(data21)[h])-np.angle(np.transpose(data21)[h][0])))))
        interperPhiS11b.append(scipy.interpolate.CubicSpline(bGMTs+h*tbFM, (-np.unwrap(np.angle(np.transpose(data11)[h])-np.angle(np.transpose(data11)[h][0])))))
        interperMS21b.append(scipy.interpolate.CubicSpline(bGMTs+h*tbFM, (np.abs(np.transpose(data21)[h])-np.abs(np.transpose(data21)[h][0]))/np.abs(np.transpose(data21)[h][0])))
        interperMS11b.append(scipy.interpolate.CubicSpline(bGMTs+h*tbFM, (np.abs(np.transpose(data11)[h])-np.abs(np.transpose(data11)[h][0]))/np.abs(np.transpose(data11)[h][0])))
    
    ###these gives a different interpolation and are packaged differently (Because of time point differences?)
    interperPhiS21 = scipy.interpolate.CubicSpline(bGMTs, (-np.unwrap(np.angle(data21[:])-np.angle(data21[0]), axis = 1))) ## interp(Times)[:,500] = interpolation at times for frequency pt. 500
    interperPhiS11 = scipy.interpolate.CubicSpline(bGMTs, (-np.unwrap(np.angle(data11[:])-np.angle(data11[0]), axis = 1)))
    interperMS21 = scipy.interpolate.CubicSpline(bGMTs, (np.abs(data21[:])-np.abs(data21[0]))/np.abs(data21[0]))
    interperMS11 = scipy.interpolate.CubicSpline(bGMTs, (np.abs(data11[:])-np.abs(data11[0]))/np.abs(data11[0]))
    
    #Everything below here is old/used for plotting
    
 #==============================================================================
 #    ##plotting them all at once
 #    plt.figure(figsize=figSize)
 #    plt.ylabel(r'$\Delta \phi$ [radians]', fontsize = labelFontSize)
 #    plt.xlabel('Time [s]', fontsize = labelFontSize)
 #    plt.title(r'Phase Drift $S_{21}$ $\Delta \phi$ by Time', fontsize = titleFontSize)
 #    b = 0
 #    handleds = []
 #    for i in range(len(data)):
 #        times = (tbB*i+tbFM*np.arange(len(fs)))[ct1:ct2]
 #        plt.plot(times, (-np.unwrap((np.angle(data21[i])-np.angle(data21[b])))*(1))[ct1:ct2], marker = '.', color = colors[i%len(colors)], markersize=2)
 #    handleds.append(mlines.Line2D([], [], color='black', label='No Mean, $\phi^n-\phi^1$', lw=3)) ##fake lines to create legend elements
 # 
 #    for i in range(len(data)-2):
 #        plt.scatter(tbB*(i+1), np.mean(np.mean(phaseDsS21[i+1], axis = 0)[ct1:ct2], axis = 0), marker = 'o', color = colors[(i+1)%len(colors)],edgecolors= "black")
 #    handleds.append(mlines.Line2D([], [], marker='o', color='w', label='Mean of Frequency-Means', markerfacecolor='black', markersize=10)) ##fake lines to create legend elements
 #        
 #    for i in range(len(data)):
 #        plt.scatter(tbB*i, np.mean(-np.unwrap(np.angle(data21[i])-np.angle(data21[b]))[ct1:ct2]), marker = 's', color = colors[i%len(colors)],edgecolors= "black")
 #    handleds.append(mlines.Line2D([], [], marker='s', color='w', label='Frequency-Means of $\phi^n-\phi^1$', markerfacecolor='black', markersize=10)) ##fake lines to create legend elements
 #     
 #    #h=100
 #    #plt.plot(bGMTs+h*tbFM,(-np.unwrap(np.angle(np.transpose(data21)[h])-np.angle(np.transpose(data21)[h][0]))),marker = 'o', label = 'TESTHERE')
 #     
 #    newTs = np.arange(0,5500)
 #    plt.plot(newTs,interperPhiS21b[500](newTs), label = 'Cubic Spline Interpolation')
 #    plt.plot(newTs,interperPhiS21(newTs)[:,500], linestyle = ':', label = 'Cubic Spline Interpolation')
 #    handleds.append(mlines.Line2D([], [], color='b', label='Cubic Spline Interpolation')) ##fake lines to create legend elements, no idea how to avoid this for this line
 #        
 #    plt.tight_layout()
 #    plt.legend(handles=handleds,ncol = 2, fontsize = legendFontSize*2, framealpha=0.4)
 #    plt.show()
 #==============================================================================
    
    #===========================================================================
    # ###if we plot the mean changes
    # plt.figure(figsize=figSize)
    # plt.ylabel(r'Mean of $\Delta \phi$ [radians]', fontsize = labelFontSize)
    # plt.xlabel('Time [s]', fontsize = labelFontSize)
    # plt.title(r'Mean of Frequency-Mean Time-Gated $S_{21}$ $\Delta \phi$ at Times', fontsize = titleFontSize)
    # for i in range(len(data)-2):
    #     plt.plot(tbB*i, np.mean(np.mean(phaseDsS21[i], axis = 0)[ct1:ct2], axis = 0), marker = 'o')
    # plt.tight_layout()
    # plt.show()
    #===========================================================================
    
    #===========================================================================
    # ##calculate the changes from first measurement to last:
    # plt.figure(figsize=figSize)
    # plt.ylabel(r'$\Delta \phi$ [radians]', fontsize = labelFontSize)
    # plt.xlabel('Time [s]', fontsize = labelFontSize)
    # plt.title(r'Frequency-Mean Time-Gated $S_{21}$ $\Delta \phi$ at Times', fontsize = titleFontSize)
    # for i in range(len(data)):
    #     plt.plot(tbB*i, np.mean(np.unwrap(np.angle(data21[i])-np.angle(data21[0]))[ct1:ct2]), marker = 'o')
    # plt.tight_layout()
    # plt.show()
    #===========================================================================
    
    #===========================================================================
    # ##calculate the changes from first measurement to last:
    # plt.figure(figsize=figSize)
    # plt.ylabel(r'$\Delta \phi$ [radians]', fontsize = labelFontSize)
    # plt.xlabel('Frequency [Hz]', fontsize = labelFontSize)
    # plt.title(r'Time-Gated $S_{21}$ $\Delta \phi = \phi^n - \phi^1$, D = n-1', fontsize = titleFontSize)
    # for i in range(len(data)):
    #     plt.plot(fs[ct1:ct2], -np.unwrap(np.angle(data21[i])-np.angle(data21[0]))[ct1:ct2], marker = 'o', label = 'D='+str(i))
    # plt.plot(fs, interperPhiS21(3000), label = 'spline')
    # plt.legend(ncol = 2, fontsize = legendFontSize*1.2, framealpha=0.4)
    # plt.tight_layout()
    # plt.show()
    #===========================================================================
    
    #===========================================================================
    # ##calculate the changes from first data-point to the last:
    # plt.figure(figsize=figSize)
    # plt.ylabel(r'$\Delta \phi/f$ [radians/Hz]', fontsize = labelFontSize)
    # plt.xlabel('Time [s]', fontsize = labelFontSize)
    # plt.title(r'Time-Gated $S_{21}$ $\Delta \phi/f$ at Times', fontsize = titleFontSize)
    # for i in range(len(data)):
    #     times = (tbB*i+tbFM*np.arange(len(fs)))[ct1:ct2]
    #     plt.plot(times, (np.unwrap((np.angle(data21[i])-np.angle(data21[0])))*(1/fs))[ct1:ct2], marker = 'o')
    # plt.tight_layout()
    # plt.show()
    #===========================================================================
    
    ########
    ####Compensation stuff
    ########
    colors = ('tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan', 'black')
    
    #===========================================================================
    # ##can calculate delta x = delta phi*c/( 2*pi*f ) as a function of time
    # plt.figure(figsize=figSize)
    # plt.ylabel(r'Mean of $\Delta x$ [m]', fontsize = labelFontSize)
    # plt.xlabel('Frequency [Hz]', fontsize = labelFontSize)
    # plt.title(r'Mean Time-Gated $S_{21}$ $\Delta x$ at Distances', fontsize = titleFontSize)
    # for i in range(len(data)-2):
    #     plt.plot(fs[ct1:ct2], np.mean(phaseDsS21[i], axis = 0)[ct1:ct2]*c/(2*pi*fs[ct1:ct2]), marker = markers[i%len(markers)], label = 'd='+str(i+1), color = colors[i%len(colors)])
    # plt.legend(ncol = 2, fontsize = legendFontSize*1.3, framealpha=0.6)
    # plt.tight_layout()
    # plt.show()
    #===========================================================================
    
    #===========================================================================
    # plt.figure(figsize=figSize)
    # plt.ylabel(r'Mean of $\Delta \phi$ [radians]', fontsize = labelFontSize)
    # plt.xlabel('Frequency [Hz]', fontsize = labelFontSize)
    # plt.title(r'Mean Time-Gated $S_{21}$ $\Delta \phi$ at Distances', fontsize = titleFontSize)
    # for i in range(len(data)-2):
    #     plt.plot(fs[ct1:ct2], np.mean(phaseDsS21[i], axis = 0)[ct1:ct2], marker = markers[i%len(markers)], label = 'd='+str(i+1), color = colors[i%len(colors)])
    # plt.legend(ncol = 2, fontsize = legendFontSize*1.3, framealpha=0.4)
    # plt.tight_layout()
    # plt.show()
    #===========================================================================
    
    #===========================================================================
    # for i in (1,9,19):
    #     plt.figure(figsize=figSize)
    #     plt.ylabel(r'$\Delta \phi$ [radians]', fontsize = labelFontSize)
    #     plt.xlabel('Frequency [Hz]', fontsize = labelFontSize)
    #     plt.title(r'Time-Gated $S_{21}$ $\Delta \phi$ at Distance d='+str(i+1), fontsize = titleFontSize)
    #     for j in range(len(phaseDsS21[i])):
    #         plt.plot(fs[ct1:ct2], phaseDsS21[i][j][ct1:ct2], marker = markers[j%len(markers)], label = 'n='+str(j+1), color = colors[j%len(colors)])
    #     plt.legend(ncol = 2, fontsize = legendFontSize*1.2, framealpha=0.4)
    #     plt.tight_layout()
    #     plt.show()
    #===========================================================================
        
    #===========================================================================
    # plt.figure(figsize=figSize)
    # plt.ylabel(r'Mean of $\Delta M$', fontsize = labelFontSize)
    # plt.xlabel('Frequency [Hz]', fontsize = labelFontSize)
    # plt.title(r'Mean Time-Gated $S_{21}$ $\Delta M$ at Distances', fontsize = titleFontSize)
    # for i in range(len(data)-2):
    #     plt.plot(fs[ct1:ct2], np.mean(magDsS21[i], axis = 0)[ct1:ct2]*c/(2*pi*fs[ct1:ct2]), marker = markers[i%len(markers)], label = 'd='+str(i+1), color = colors[i%len(colors)])
    # plt.legend(ncol = 2, fontsize = legendFontSize, framealpha=0.4)
    # plt.tight_layout()
    # plt.show()
    #===========================================================================
    
    #===========================================================================
    # for i in (1,9,19):
    #     plt.figure(figsize=figSize)
    #     plt.ylabel(r'$\Delta M$', fontsize = labelFontSize)
    #     plt.xlabel('Frequency [Hz]', fontsize = labelFontSize)
    #     plt.title(r'Time-Gated $S_{21}$ $\Delta M$ at Distance d='+str(i+1), fontsize = titleFontSize)
    #     for j in range(len(magDsS21[i])):
    #         plt.plot(fs[ct1:ct2], magDsS21[i][j][ct1:ct2], marker = markers[j%len(markers)], label = 'n='+str(j+1), color = colors[j%len(colors)])
    #     plt.legend(ncol = 2, fontsize = legendFontSize, framealpha=0.4)
    #     plt.tight_layout()
    #     plt.show()
    #===========================================================================

    
    
    
    #===========================================================================
    # plt.figure(figsize=figSize)
    # plt.ylabel(r'Mean of $S_{21,n}-S_{21,n \pm d}$ [logscale]', fontsize = labelFontSize)
    # plt.xlabel('Frequency [Hz]', fontsize = labelFontSize)
    # plt.title(r'Mean + S-D Time-Gated $S_{21}$ Subtractions at Distances', fontsize = titleFontSize)
    # plt.yscale('log')
    # for i in (0,4,9,14,19):
    #     plt.errorbar(fs[ct1:ct2], np.mean(subS21[i], axis = 0)[ct1:ct2], yerr = np.std(subS21[i], axis = 0)[ct1:ct2], marker = markers[i%len(markers)], label = 'd='+str(i+1), color = colors[i%len(colors)], alpha = 0.3)
    # plt.legend(ncol = 2, fontsize = legendFontSize*1.8, framealpha=0.4)
    # plt.show()
    #===========================================================================
    
    
    #===========================================================================
    # plt.figure(figsize=figSize)
    # plt.ylabel(r'Mean of $S_{21,n}-S_{21,n \pm d}$ [dB]', fontsize = labelFontSize)
    # plt.xlabel('Frequency [Hz]', fontsize = labelFontSize)
    # plt.title(r'Mean Time-Gated $S_{21}$ Subtractions at Distances', fontsize = titleFontSize)
    # for i in range(len(data)-2):
    #     plt.plot(fs[ct1:ct2], 10*np.log10(np.mean(subS21[i], axis = 0))[ct1:ct2], marker = markers[i%len(markers)], label = 'd='+str(i+1), color = colors[i%len(colors)])
    # plt.legend(ncol = 2, fontsize = legendFontSize, framealpha=0.4)
    # plt.show()
    #===========================================================================
    
    #===========================================================================
    # plt.figure(figsize=figSize)
    # plt.title(r'Mean Time-Gated  $S_{11}$ Subtractions at Distances', fontsize = titleFontSize)
    # plt.ylabel(r'Mean of $S_{11,n}-S_{11,n \pm d}$ [dB]', fontsize = labelFontSize)
    # plt.xlabel('Frequency [Hz]', fontsize = labelFontSize)
    # for i in range(len(data)-2):
    #     plt.plot(fs[ct1:ct2], 10*np.log10(np.mean(subS11[i], axis = 0))[ct1:ct2], marker = markers[i%len(markers)], label = 'd='+str(i+1), color = colors[i%len(colors)])
    # plt.legend(ncol = 2, fontsize = legendFontSize, framealpha=0.4)
    # plt.show()
    #===========================================================================
    
    #===========================================================================
    # for i in (1,9,19):
    #     plt.figure(figsize=figSize)
    #     plt.ylabel(r'$S_{21,n}-S_{21,n \pm d}$ [dB]', fontsize = labelFontSize)
    #     plt.xlabel('Frequency [Hz]', fontsize = labelFontSize)
    #     plt.title(r'Time-Gated $S_{21}$ Subtractions at Distance d='+str(i+1), fontsize = titleFontSize)
    #     for j in range(len(subS21[i])):
    #         plt.plot(fs[ct1:ct2], 10*np.log10(subS21[i][j])[ct1:ct2], marker = markers[j%len(markers)], label = 'n='+str(j+1), color = colors[j%len(colors)])
    #     plt.legend(ncol = 2, fontsize = legendFontSize, framealpha=0.4)
    #     plt.show()
    #===========================================================================
    
        
        #=======================================================================
        # plt.title('S21 Subtractions at Distance'+str(i+1))
        # for j in range(len(subS21[i])):
        #     plt.plot(fs, 10*np.log10(subS21[i][j]), marker = '.', label = str(j))
        # plt.legend()
        # plt.show()
        # plt.clf()
        #=======================================================================
    #===========================================================================
    # freqRanges = ['6.6-15GHz','7-8GHz','8-9GHz','9-10GHz','10-11GHz','11-12GHz','12-13GHz','13-14GHz','14-15GHz'] ##frequency bins, if plotting the data this way. First bin is all the data.
    # fs = [] ##frequencies from each of these bins
    # nSbyFreqR = []
    # 
    # 
    # avgDiffBins = [] ##avgDiffs for each freq range
    # avgDiffsTG = [] ##averaging differences over the entire frequency range. time-gated
    # avgDiffsNTG = [] ##averaging differences over the entire frequency range. not time-gated
    # avgSDsTG = [] ##averaging standard deviations over the entire frequency range. time-gated
    # avgSDsNTG = [] ##averaging standard deviations over the entire frequency range. not time-gated
    # 
    # for b in range(len(freqRanges)): ##iterates over each frequency range
    #     nS = [] ##neighbour subtractions, [0] being nearest-neighbour [1] 1-removed, etc.
    #     fs.append(data[0][freqRanges[b]].frequency.f)
    #     for i in range(len(data)): ##iterates over each dataset
    #         nearingI = []
    #         for j in range(len(data)): ##iterates over each dataset again, to get each one in relation with each other one
    #             for k in [i+j,i-j]: ##check forward and backward
    #                 if(k > 0 and k < (len(data)-1)): ##check that it is actually in the set
    #                     datum = (data[i]-data[k])[freqRanges[b]]
    #                     nearingI.append(datum)
    #         nS.append(nearingI)
    #     nSbyFreqR.append(nS)
    #     
    #     for n in range(len(nS)):
    #         avgDiffsTG = [] ##averaging differences over the entire frequency range. time-gated
    #         avgDiffsNTG = [] ##averaging differences over the entire frequency range. not time-gated
    #         avgSDsTG = [] ##averaging standard deviations over the entire frequency range. time-gated
    #         avgSDsNTG = [] ##averaging standard deviations over the entire frequency range. not time-gated
    #         for j in nS[n]:
    #             avgDiffsNTG.append(np.mean(j.s21.s.squeeze()))
    #             avgDiffsTG.append(np.mean(tplusgate((fs[b], j.s21.s.squeeze()), tg[0][1], tg[1], tg[2], tg[0][0])[2][0:len(fs[b])]))
    #         avgDiffBins.append([avgDiffsTG, avgDiffsNTG, avgSDsTG, avgSDsNTG])
    #         for h in avgDiffsTG:
    #             plt.plot(10*np.log10(h), marker = '.')
    #         plt.legend()
    #         plt.show()
    #         plt.clf()
    #===========================================================================
    
    
    
    #===========================================================================
    # for item in nSbyFreqR[0][0]:
    #     plt.plot(item.frequency.f, 10*np.log10(item.s21.s.squeeze()))
    #     
    #     
    # plt.title(r'Raw $S_{12, \mathrm{ref}}$ vs $S_{12, \mathrm{ref}} - S_{12, \mathrm{other}}$')
    # #plt.title(r'Time-gated $S_{12, \mathrm{ref}}$ vs $S_{12, \mathrm{ref}} - S_{12, \mathrm{other}}$')
    # plt.legend()
    # plt.show()
    # plt.clf()
    #===========================================================================
    
    #===========================================================================
    # labelled = [] ##to avoid duplicate plot labels
    # for i in range(len(data)):
    #     labelled.append(False)
    #     for item in nSbyFreqR[0][i]:
    #         if(labelled[i]):
    #             label = None
    #         else:
    #             label = str(i+1)+'th nearest neighbour'
    #             labelled[i] = True
    #             
    #         fs = item.frequency.f
    #         if(tg == None):
    #             plt.plot(fs, 10*np.log10(item.s21.s.squeeze()), label = label, color = colors[i%len(colors)])
    #         else:
    #             plt.plot(fs, 10*np.log10(tplusgate((fs, item.s21.s.squeeze()), tg[0][1], tg[1], tg[2], tg[0][0])[2][0:len(fs)]), label = label, color = colors[i%len(colors)])
    # plt.legend()
    # plt.show()
    #===========================================================================
    
    ## interp(Times)[:,500] = interpolation at times for frequency pt. 500
    ##can do interpolation splines for each frequency
    return interperPhiS11b, interperMS11b, interperPhiS21b, interperMS21b, tbB/2, tbFM## [interpolation splines for phi, then for M, time between measurements [s], est. time per frequency point [s]]
    
def plotBackgroundSubtraction(data, names, tg = None): ##first data is the reference, tg is the pad type+number, then tr peak time, then tr peak width
    plt.figure(figsize=(8,8))
    plt.ylabel('dB Scale')
    
    if(tg == None):
        plt.title(r'Raw $S_{12, \mathrm{ref}}$ vs $S_{12, \mathrm{ref}} - S_{12, \mathrm{other}}$')
        fs = data[0].frequency.f
        plt.plot(fs, 10*np.log10(np.abs(data[0].s21.s.squeeze())), label = names[0])
        for i in range(len(data)-1):  
            plt.plot(fs, 10*np.log10(np.abs(data[0].s21.s.squeeze()-data[i+1].s21.s.squeeze())), label = names[i+1])
    else:
        plt.title(r'Time-gated $S_{12, \mathrm{ref}}$ vs $S_{12, \mathrm{ref}} - S_{12, \mathrm{other}}$')
        pad = tg[0]
        fs = data[0].frequency.f
        tgRef = tplusgate((fs, data[0].s21.s.squeeze()), pad[1], tg[1], tg[2], pad[0])[2][0:len(fs)]
        plt.plot(fs, 10*np.log10(np.abs(tgRef)), label = names[0])
        for i in range(len(data)-1):
            plt.plot(fs, 10*np.log10(np.abs(tgRef - tplusgate((fs, data[i+1].s21.s.squeeze()), pad[1], tg[1], tg[2], pad[0])[2][0:len(fs)])), label = names[i+1])
        
    plt.legend()
    plt.show()


def tplusgate(frequencyDomain, gateType = 'hamming', peakCentre = -1, peakWidth = -1, padding = None):
    freqs = frequencyDomain[0]
    Sfreq = frequencyDomain[1] ## S parameter in frequency domain
    if(padding != None):
        Sfreq = np.pad(Sfreq, (0, padding), 'constant') 
    if(peakCentre == -1 or peakWidth == -1):
        print('No gating peak/width specified.')
        return frequencyDomain
    
    Stime = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(Sfreq))) ## S parameter in time domain
    #times = np.zeros(len(Stime))
    #for i in range(len(times)):
    #    times[i] = (i-len(times)/2) * 285.713/(len(times))*2
     
    times = np.fft.ifftshift(np.fft.fftfreq(len(Stime), freqs[1]-freqs[0]))
        
    start = np.argmin(np.abs(times-(peakCentre-peakWidth/2)))
    end = np.argmin(np.abs(times-(peakCentre+peakWidth/2)))
    
    gate = np.zeros(len(times))
    if(gateType == 'rectangular'):
        gate[start:end] = 1
        
    elif(gateType == 'hamming'):
        hgat = np.hamming(end-start)  
        for i in range(len(hgat)):
            gate[start+i] = hgat[i]**1
    elif(gateType == 'tukey'):
        alpha = 0.1
        gate[start:end] = scipy.signal.windows.tukey(end-start, alpha)
    else:
        print('Error time gating: no valid gate selected')
        
    StimeGated = Stime*gate
    SfreqGated = np.fft.ifftshift(np.fft.fft(np.fft.fftshift(StimeGated)))[0:len(frequencyDomain[0])]
    return [freqs, Sfreq, SfreqGated, times, Stime, StimeGated]
    
    
def processOneData(data, emptData, trpeak, trpeakw, dAnt, pad, name, makePlots = 0, sphereCalData = None, monostatic = None, interpData = None):
    rt = dAnt/2
    rr = dAnt/2
    rd = dAnt
    f = data.frequency.f
    data = data.s.squeeze()
    emptData = emptData.s.squeeze()
    if(sphereCalData != None):
        sphereData = sphereCalData[0].s.squeeze()
        if(len(sphereCalData) == 3):
            sphereEmptData = sphereCalData[2].s.squeeze()

    if(driftCorrPreTG):
        #### interp data pre-tg
        if(interpData != None): ## cancel the drift - just do this for the time-gated and padded data
            if(monostatic):
                data = driftComp(interpData[0], interpData[1], data, data, interpData[4], interpData[5], interpData[6], interpData[7], monostatic)
                emptData = driftComp(interpData[0], interpData[1], emptData, emptData, interpData[4], interpData[5], interpData[6], interpData[7], monostatic)
            else:
                data= driftComp(interpData[2], interpData[3], data, data, interpData[4], interpData[5], interpData[6], interpData[7], monostatic)
                emptData = driftComp(interpData[2], interpData[3], emptData, emptData, interpData[4], interpData[5], interpData[6], interpData[7], monostatic)
                if(sphereCalData != None):
                    sphereData = driftComp(interpData[2], interpData[3], sphereData, sphereData, interpData[4], interpData[5], interpData[6], interpData[7], monostatic)
                    sphereEmptData = driftComp(interpData[2], interpData[3], sphereEmptData, sphereEmptData, interpData[4], interpData[5], interpData[6], interpData[7], monostatic)
            
    
    if(sphereCalData != None): ## do sphere data also
        sphereTPG = tplusgate((f, sphereData), pad[1], trpeak, trpeakw, pad[0]) ## gated and padded
        sphereT = tplusgate((f, sphereData), pad[1], trpeak, trpeakw) ## just gated
        if(len(sphereCalData) == 3):
            sphereEmptTPG = tplusgate((f, sphereEmptData), pad[1], trpeak, trpeakw, pad[0]) ## gated and padded
            sphereEmptT = tplusgate((f, sphereEmptData), pad[1], trpeak, trpeakw) ## just gated
            
    
    dataTPG = tplusgate((f, data), pad[1], trpeak, trpeakw, pad[0]) ## gated and padded
    dataT = tplusgate((f, data), pad[1], trpeak, trpeakw) ## just gated
    emptTPG = tplusgate((f, emptData), pad[1], trpeak, trpeakw, pad[0]) ## gated and padded
    emptT = tplusgate((f, emptData), pad[1], trpeak, trpeakw) ## just gated
    
    figsizex = 14
    figsizey = 8
    linewidth = 1.2
    font = {'size'   : 12}
    plt.rc('font', **font)
    
    if(makePlots > 1):
        fig = plt.figure( figsize=(figsizex, figsizey), dpi=80, facecolor='w', edgecolor='k')
        ax1 = fig.add_subplot(121)
        ax1.set_xlabel('Frequency [GHz]', fontsize = '19')
        ax1.set_ylabel(r'Transmission [dB]', fontsize = '19')
        plt.plot(dataT[0]*1e-9, 20*np.log10(np.abs(dataT[1])), label = r'$S_{21}$', linewidth=linewidth, marker = '.', markersize = 10)
        ax1.set_title(r"|$S_{21}$| by Frequency for "+name, fontsize = '22')
        ax1.grid()
        fig.tight_layout()
        ax1.legend(fontsize = 14)
        
        ax2 = fig.add_subplot(122)
        ax2.set_xlabel('Time [ns]', fontsize = '19')
        ax2.set_ylabel(r'Transmission [dB]', fontsize = '19')
        ax2.plot(dataT[3]*1e9,20*np.log10(np.abs(dataT[4])), label = r'$S_{21}$', linewidth=linewidth, marker = '.', markersize = 10)
        ax2.plot(dataT[3]*1e9,20*np.log10(np.abs(dataT[5])), label = r'$S_{21}$, GateType: '+pad[1], linewidth=linewidth, marker = '.', markersize = 10)  
        ax2.set_title(r"|$S_{21}$| by Time for "+name, fontsize = '22')
        ax2.grid()
        fig.tight_layout()
        ax2.legend(fontsize = 14)
        plt.show()
        
        
        fig = plt.figure( figsize=(figsizex, figsizey), dpi=80, facecolor='w', edgecolor='k')
        ax1 = fig.add_subplot(121)
        ax1.set_xlabel('Frequency [GHz]', fontsize = '19')
        ax1.set_ylabel(r'Transmission [dB]', fontsize = '19')
        plt.plot(dataTPG[0]*1e-9, 20*np.log10(np.abs(dataTPG[1])), label = r'$S_{21}$', linewidth=linewidth, marker = '.', markersize = 10)
        ax1.set_title(r"Padded |$S_{21}$| by Frequency for "+name, fontsize = '22')
        ax1.grid()
        fig.tight_layout()
        ax1.legend(fontsize = 14)
        
        ax2 = fig.add_subplot(122)
        ax2.set_xlabel('Time [ns]', fontsize = '19')
        ax2.set_ylabel(r'Transmission [dB]', fontsize = '19')
        ax2.plot(dataTPG[3]*1e9,20*np.log10(np.abs(dataTPG[4])), label = r'$S_{21}$', linewidth=linewidth, marker = '.', markersize = 10)
        ax2.plot(dataTPG[3]*1e9,20*np.log10(np.abs(dataTPG[5])), label = r'$S_{21}$, GateType: '+pad[1], linewidth=linewidth, marker = '.', markersize = 10)
        ax2.set_title(r"Padded |$S_{21}$| by Time for "+name, fontsize = '22')
        ax2.grid()
        fig.tight_layout()
        ax2.legend(fontsize = 14)
        plt.show()

    if(not driftCorrPreTG):
        #### interpData is ##S11 interpolations (phi then M), S21 interpolations (phi then M), time between measurements, time between frequency points, list of BGs, list of non BGs
        if(interpData != None): ## cancel the drift - just do this for the time-gated and padded data
            if(monostatic):
                dataTPG[2] = driftComp(interpData[0], interpData[1], dataTPG[2], data, interpData[4], interpData[5], interpData[6], interpData[7], monostatic)
                emptTPG[2] = driftComp(interpData[0], interpData[1], emptTPG[2], emptData, interpData[4], interpData[5], interpData[6], interpData[7], monostatic)
            else:
                dataTPG[2] = driftComp(interpData[2], interpData[3], dataTPG[2], data, interpData[4], interpData[5], interpData[6], interpData[7], monostatic)
                emptTPG[2] = driftComp(interpData[2], interpData[3], emptTPG[2], emptData, interpData[4], interpData[5], interpData[6], interpData[7], monostatic)
                if(sphereCalData != None):
                    sphereTPG[2] = driftComp(interpData[2], interpData[3], sphereTPG[2], sphereData, interpData[4], interpData[5], interpData[6], interpData[7], monostatic)
                    sphereEmptTPG[2] = driftComp(interpData[2], interpData[3], sphereEmptTPG[2], sphereEmptData, interpData[4], interpData[5], interpData[6], interpData[7], monostatic)
            
            
            
    ##calculate the return values
    if(sphereCalData != None): ##do sphere calibration
        if(len(sphereCalData) == 3):
            sigETG = -np.imag((dataT[2]-emptT[2])/(sphereT[2]-sphereEmptT[2]) * ForwardFarField(emptT[0], sphereCalData[1])/emptT[0]*2*c) ##time gated
            sigENTG = -np.imag((dataT[1]-emptT[1])/(sphereT[1]-sphereEmptT[1]) * ForwardFarField(emptT[0], sphereCalData[1])/emptT[0]*2*c) ##non processed
            sigETGP = -np.imag((dataTPG[2]-emptTPG[2])/(sphereTPG[2]-sphereEmptTPG[2]) * ForwardFarField(emptTPG[0], sphereCalData[1])/emptTPG[0]*2*c) ##padded and time gated
        else:
            sigETG = -np.imag((dataT[2]-emptT[2])/(sphereT[2]-emptT[2]) * ForwardFarField(emptT[0], sphereCalData[1])/emptT[0]*2*c) ##time gated
            sigENTG = -np.imag((dataT[1]-emptT[1])/(sphereT[1]-emptT[1]) * ForwardFarField(emptT[0], sphereCalData[1])/emptT[0]*2*c) ##non processed
            sigETGP = -np.imag((dataTPG[2]-emptTPG[2])/(sphereTPG[2]-emptTPG[2]) * ForwardFarField(emptTPG[0], sphereCalData[1])/emptTPG[0]*2*c) ##padded and time gated
    elif(monostatic): ##try to get monostatic here
        sigETG = dataT[2]-emptT[2] ##time gated
        sigENTG = dataT[1]-emptT[1] ##non processed
        sigETGP = dataTPG[2]-emptTPG[2] ##padded and time gated
    else:  ## use background calibration
        sigETG = -np.imag((dataT[2]-emptT[2])/emptT[2] * (2*rt*rr/rd))*c0/emptT[0] ##time gated
        sigENTG = -np.imag((dataT[1]-emptT[1])/emptT[1] * (2*rt*rr/rd))*c0/emptT[0] ##non processed
        sigETGP = -np.imag((dataTPG[2]-emptTPG[2])/emptTPG[2] * (2*rt*rr/rd))*c0/emptTPG[0] ##padded and time gated
        
    
    
    if(makePlots > 0):
        fig = plt.figure( figsize=(figsizex, figsizey), dpi=80, facecolor='w', edgecolor='k')
        ax1 = fig.add_subplot(111)
        ax1.set_xlabel('Frequency [GHz]', fontsize = '19')
        if(monostatic):
            ax1.set_ylabel(r'S - S$_{background}$', fontsize = '19')
            plt.plot(emptTPG[0], np.abs(sigETGP)**2, label = r'S - S$_{bg}$ for '+name+', padded + time-gated', linewidth=linewidth*1.5)
            plt.plot(emptT[0], np.abs(sigENTG)**2, label = r'S - S$_{bg}$ for '+name+', unprocessed', linewidth=linewidth*1.5)
            plt.plot(emptT[0], np.abs(sigETG)**2, label = r'S - S$_{bg}$ for '+name+', time-gated', linewidth=linewidth*1.5)
            ax1.set_title(r"S - S$_{bg}$ by Frequency for "+name, fontsize = '22')
        else:
            ax1.set_ylabel(r'$\sigma_E$ [arb. units]', fontsize = '19')
            plt.plot(emptTPG[0], sigETGP, label = '$\sigma_e$ for '+name+', padded + time-gated', linewidth=linewidth*1.5)
            plt.plot(emptT[0], sigENTG, label = '$\sigma_e$ for '+name+', unprocessed', linewidth=linewidth*1.5)
            plt.plot(emptT[0], sigETG, label = '$\sigma_e$ for '+name+', time-gated', linewidth=linewidth*1.5)
            ax1.set_title(r"$\sigma_E$ by Frequency for "+name, fontsize = '22')
        
        ax1.grid()
        plt.xlim(f[0],f[np.alen(f)-1])
        fig.tight_layout()
        ax1.legend(fontsize = 14)
        plt.show()
    
    return emptTPG[0], sigETGP

def ForwardFarField(fs, a):
    """Compute the forward scattered far field for a PEC sphere of radius a."""
    lambda0 = c0/fs
    x = 2*pi/lambda0*a
    
    F = np.zeros(x.shape, dtype=complex)
    N = int(x.max() + 4.05*(x.max())**(1/3) + 2) # Wiscombe truncation criterion
    for n in np.arange(1, N + 1):
        F = F - 1j*lambda0/(4*np.pi)*(2*n + 1)*(spherical_xjnp(n, x)/spherical_xhnp(n, x) + spherical_jn(n, x)/spherical_hn(n, x))
    return(F)

def BackwardScatteringCS(fs, a):
    """Compute the backward scattering cross-section for a PEC sphere of radius a."""
    lambda0 = c0/(fs+1) ##+1 so there are no zeros
    
    x = 2*pi/lambda0*a
    
    F = np.zeros(x.shape, dtype=complex)
    N = int(x.max() + 4.05*(x.max())**(1/3) + 2) # Wiscombe truncation criterion
    for n in np.arange(1, N + 1):
        F = F + (-1)**n*(2*n + 1)*(spherical_xjnp(n, x)/spherical_xhnp(n, x) - spherical_jn(n, x)/spherical_hn(n, x))
    F[fs == 0] = 0  ##since we are padding, remove 0 values
    RCS_back = lambda0**2/(4*pi)*np.abs(F)**2
    
    #===========================================================================
    # RCS_back = np.zeros(x.shape)
    # m = 0 - .01j##complex index of refraction for PEC?? In miepython.
    # for i in range(len(RCS_back)):
    #     if(fs[i] == 0):
    #         pass
    #     else:
    #         qext, qsca, qback, g = miepython.mie(m, x = 2*pi*(a)/lambda0[i])
    #         RCS_back[i] = qback*pi*a**2
    #===========================================================================
    return(RCS_back)
    
def theMainStuff(driftCompensating = False, toPlotComparisons = False):
    
    fileLoc = 'C:/Users/al8032pa/Work Folders/Documents/antenna measurements/'
    

    #===========================================================================
    # # S21s, ordered as taken -- using KEYSIGHT PNA-L Network Analyzer N5235B --first meas. taken with d_antennas ~4.7m, 0 dB power IF BW = 70Hz, 8-12GHz, 1601 pts. --9/1/2023
    # ### - noise when disconnected cables of about -105 dB +- 5 dB
    # ##horn antenna aperture is 20 x 15 cm
    # 
    # ###starting with background-background then calibration sphere measurements
    # emptyA1 = rf.Network(fileLoc+"09-1-2023/emptyA1.s2p")
    # emptyA2 = rf.Network(fileLoc+"09-1-2023/emptyA2.s2p")
    # emptyA3 = rf.Network(fileLoc+"09-1-2023/emptyA3.s2p")
    # emptyB1 = rf.Network(fileLoc+"09-1-2023/emptyB1.s2p")
    # emptyB2 = rf.Network(fileLoc+"09-1-2023/emptyB2.s2p")
    # emptyB3 = rf.Network(fileLoc+"09-1-2023/emptyB3.s2p")
    # emptyB4 = rf.Network(fileLoc+"09-1-2023/emptyB4.s2p")
    # ball15mmB = rf.Network(fileLoc+"09-1-2023/ball15mmB.s2p")
    #  
    # ##now changing settings to test for best settings on VNA:
    # ##  IF BW = 200Hz, 7-15GHz, 0 dB power, 3000 pts. no averaging
    # emptyC1 = rf.Network(fileLoc+"09-1-2023/emptyC1.s2p")
    # emptyC2 = rf.Network(fileLoc+"09-1-2023/emptyC2.s2p")
    #  
    # ##  IF BW = 50Hz, 7-15GHz, 0 dB power, 3000 pts. no averaging
    # emptyD1 = rf.Network(fileLoc+"09-1-2023/emptyD1.s2p")
    # emptyD2 = rf.Network(fileLoc+"09-1-2023/emptyD2.s2p")
    #  
    # ##  IF BW = 10Hz, 7-15GHz, 0 dB power, 3000 pts. no averaging
    # emptyE1 = rf.Network(fileLoc+"09-1-2023/emptyE1.s2p")
    # emptyE2 = rf.Network(fileLoc+"09-1-2023/emptyE2.s2p")
    #  
    #  
    # ##  IF BW = 2000Hz, 7-15GHz, 0 dB power, 3000 pts. no averaging
    # emptyF1 = rf.Network(fileLoc+"09-1-2023/emptyF1.s2p")
    # emptyF2 = rf.Network(fileLoc+"09-1-2023/emptyF2.s2p")
    #  
    # ##  IF BW = 100Hz, 7-15GHz, 0 dB power, 3000 pts. 4x sweep averaging
    # emptyG1 = rf.Network(fileLoc+"09-1-2023/emptyG1.s2p")
    # emptyG2 = rf.Network(fileLoc+"09-1-2023/emptyG2.s2p")
    #  
    # ##  IF BW = 100Hz, 7-15GHz, 0 dB power, 3000 pts. 8x sweep averaging
    # emptyH1 = rf.Network(fileLoc+"09-1-2023/emptyH1.s2p")
    # emptyH2 = rf.Network(fileLoc+"09-1-2023/emptyH2.s2p")
    #  
    # ##  IF BW = 100Hz, 7-15GHz, 4 dB power, 3000 pts. 4x sweep averaging
    # emptyI1 = rf.Network(fileLoc+"09-1-2023/emptyI1.s2p")
    # emptyI2 = rf.Network(fileLoc+"09-1-2023/emptyI2.s2p")
    #  
    # ##  IF BW = 100Hz, 7-15GHz, 6 dB power, 3000 pts. 4x sweep averaging
    # emptyJ1 = rf.Network(fileLoc+"09-1-2023/emptyJ1.s2p")
    # emptyJ2 = rf.Network(fileLoc+"09-1-2023/emptyJ2.s2p")
    #  
    # ##  IF BW = 100Hz, 7-15GHz, 5 dB power, 1000 pts. 4x sweep averaging
    # emptyK1 = rf.Network(fileLoc+"09-1-2023/emptyK1.s2p")
    # emptyK2 = rf.Network(fileLoc+"09-1-2023/emptyK2.s2p")
    #  
    # ##  IF BW = 100Hz, 7-15GHz, 5 dB power, 6000 pts. 4x sweep averaging
    # emptyL1 = rf.Network(fileLoc+"09-1-2023/emptyL1.s2p")
    # emptyL2 = rf.Network(fileLoc+"09-1-2023/emptyL2.s2p")
    #  
    # ###FINAL CHOICE:  IF BW = 100Hz, 7-15GHz, 5 dB power, 3000 pts. 4x sweep averaging
    #  
    # ##now alternate between backgrounds and spheres - sizes in order in mm: 30mm, 25mm, 20mm, 15mm, 10mm, 8.75mm, 7.5mm, 6.35mm, 5mm, 3.975mm, 2.5mm, 1.5mm
    # emptyM = rf.Network(fileLoc+"09-1-2023/emptyM.s2p")
    # ball30mm = rf.Network(fileLoc+"09-1-2023/ball30mm.s2p")
    # emptyN = rf.Network(fileLoc+"09-1-2023/emptyN.s2p")
    # ball25mm = rf.Network(fileLoc+"09-1-2023/ball25mm.s2p")
    # emptyO = rf.Network(fileLoc+"09-1-2023/emptyO.s2p")
    # ball20mm = rf.Network(fileLoc+"09-1-2023/ball20mm.s2p")
    # emptyP = rf.Network(fileLoc+"09-1-2023/emptyP.s2p")
    # ball15mm = rf.Network(fileLoc+"09-1-2023/ball15mm.s2p")
    # emptyQ = rf.Network(fileLoc+"09-1-2023/emptyQ.s2p")
    # ball10mm = rf.Network(fileLoc+"09-1-2023/ball10mm.s2p")
    # #emptyR = rf.Network(fileLoc+"09-1-2023/emptyR.s2p") ##missing measurement
    # ball875mm = rf.Network(fileLoc+"09-1-2023/ball875mm.s2p")
    # emptyS = rf.Network(fileLoc+"09-1-2023/emptyS.s2p")
    # ball75mm = rf.Network(fileLoc+"09-1-2023/ball75mm.s2p")
    # emptyT = rf.Network(fileLoc+"09-1-2023/emptyT.s2p")
    # ball635mm = rf.Network(fileLoc+"09-1-2023/ball635mm.s2p")
    # emptyU = rf.Network(fileLoc+"09-1-2023/emptyU.s2p")
    # ball5mm = rf.Network(fileLoc+"09-1-2023/ball5mm.s2p")
    # emptyV = rf.Network(fileLoc+"09-1-2023/emptyV.s2p")
    # ball3975mm = rf.Network(fileLoc+"09-1-2023/ball3975mm.s2p")
    # emptyW = rf.Network(fileLoc+"09-1-2023/emptyW.s2p")
    # ball025mm = rf.Network(fileLoc+"09-1-2023/ball025mm.s2p")
    # emptyX = rf.Network(fileLoc+"09-1-2023/emptyX.s2p")
    # ball015mm = rf.Network(fileLoc+"09-1-2023/ball015mm.s2p")
    # emptyY = rf.Network(fileLoc+"09-1-2023/emptyY.s2p")
    #  
    #  
    # ##  IF BW = 500Hz, 7-15GHz, 5 dB power, 2400 pts. no sweep averaging
    # emptyaA1 = rf.Network(fileLoc+"10-1-2023/emptyA1.s2p")
    # emptyaA2 = rf.Network(fileLoc+"10-1-2023/emptyA2.s2p")
    #  
    # ##  IF BW = 500Hz, 7-15GHz, 5 dB power, 2400 pts. 2x sweep averaging
    # emptyaB1 = rf.Network(fileLoc+"10-1-2023/emptyB1.s2p")
    # emptyaB2 = rf.Network(fileLoc+"10-1-2023/emptyB2.s2p")
    #  
    # ##  IF BW = 500Hz, 7-15GHz, 5 dB power, 2400 pts. 4x sweep averaging
    # emptyaC1 = rf.Network(fileLoc+"10-1-2023/emptyC1.s2p")
    # emptyaC2 = rf.Network(fileLoc+"10-1-2023/emptyC2.s2p")
    #  
    # ##  IF BW = 500Hz, 7-15GHz, 5 dB power, 2400 pts. 8x sweep averaging
    # emptyaD1 = rf.Network(fileLoc+"10-1-2023/emptyD1.s2p")
    # emptyaD2 = rf.Network(fileLoc+"10-1-2023/emptyD2.s2p")
    #  
    #  
    # ##more balls, IF BW = 500Hz, 7-15GHz, 5 dB power, 2400 pts. 2x sweep averaging
    # emptyaD = rf.Network(fileLoc+"10-1-2023/emptyD.s2p")
    # balla635mm = rf.Network(fileLoc+"10-1-2023/ball635mm.s2p")
    # emptyaE = rf.Network(fileLoc+"10-1-2023/emptyE.s2p")
    # balla5mm = rf.Network(fileLoc+"10-1-2023/ball5mm.s2p")
    # emptyaF = rf.Network(fileLoc+"10-1-2023/emptyF.s2p")
    # balla3975mm = rf.Network(fileLoc+"10-1-2023/ball3975mm.s2p")
    # emptyaG = rf.Network(fileLoc+"10-1-2023/emptyG.s2p")
    # balla025mm = rf.Network(fileLoc+"10-1-2023/ball025mm.s2p")
    # emptyaH = rf.Network(fileLoc+"10-1-2023/emptyH.s2p")
    # balla30mm = rf.Network(fileLoc+"10-1-2023/ball30mm.s2p")
    # emptyaI = rf.Network(fileLoc+"10-1-2023/emptyI.s2p")
    #  
    #  
    # ##more balls, IF BW = 30Hz, 7-15GHz, 5 dB power, 2400 pts. 2x sweep averaging
    # emptyaJ = rf.Network(fileLoc+"10-1-2023/emptyJ.s2p")
    # ballb3975mm = rf.Network(fileLoc+"10-1-2023/ballb3975mm.s2p")
    # emptyaK = rf.Network(fileLoc+"10-1-2023/emptyK.s2p")
    # ballb025mm = rf.Network(fileLoc+"10-1-2023/ballb025mm.s2p")
    # emptyaL = rf.Network(fileLoc+"10-1-2023/emptyL.s2p")
    #  
    #  
    # ##first antenna measurements - standing in holder: IF BW = 50Hz, 7-15GHz, 5 dB power, 2400 pts. 2x sweep averaging
    # emptybA = rf.Network(fileLoc+"10-1-2023/emptybA.s2p")
    # shortUncloakedb = rf.Network(fileLoc+"10-1-2023/short uncloakedb.s2p")
    # emptybB = rf.Network(fileLoc+"10-1-2023/emptybB.s2p")
    # uncloakedb = rf.Network(fileLoc+"10-1-2023/uncloakedb.s2p")
    # emptybC = rf.Network(fileLoc+"10-1-2023/emptybC.s2p")
    # longCloakedb = rf.Network(fileLoc+"10-1-2023/long cloakedb.s2p") ##long time gap after this one
    # emptybD = rf.Network(fileLoc+"10-1-2023/emptybD.s2p")
    # shortCloakedb = rf.Network(fileLoc+"10-1-2023/short cloakedb.s2p")
    # emptybE = rf.Network(fileLoc+"10-1-2023/emptybE.s2p")
    #  
    # ##second antenna measurements - lying on foam block: IF BW = 50Hz, 7-15GHz, 5 dB power, 2400 pts. 2x sweep averaging. taken immediately after first antenna measurements
    # shortUncloakedc = rf.Network(fileLoc+"10-1-2023/short uncloakedc.s2p")
    # emptycB = rf.Network(fileLoc+"10-1-2023/emptycB.s2p")
    # uncloakedc = rf.Network(fileLoc+"10-1-2023/uncloakedc.s2p")
    # ##I forgot to stop using the holder. starting again:
    # emptydA = rf.Network(fileLoc+"10-1-2023/emptydA.s2p")
    # shortUncloakedd = rf.Network(fileLoc+"10-1-2023/short uncloakedd.s2p")
    # emptydB = rf.Network(fileLoc+"10-1-2023/emptydB.s2p")
    # uncloakedd = rf.Network(fileLoc+"10-1-2023/uncloakedd.s2p")
    # emptydC = rf.Network(fileLoc+"10-1-2023/emptydC.s2p")
    # longCloakedd = rf.Network(fileLoc+"10-1-2023/long cloakedd.s2p") ##long time gap after this one
    # emptydD = rf.Network(fileLoc+"10-1-2023/emptydD.s2p")
    # shortCloakedd = rf.Network(fileLoc+"10-1-2023/short cloakedd.s2p")
    # emptydE = rf.Network(fileLoc+"10-1-2023/emptydE.s2p") ##accidentally left the light on for this one
    #  
    #  
    # ##now more balls before going back to antenna measurements - measuring s21 and s11 now - using antenna holder: IF BW = 50Hz, 7-15GHz, 5 dB power, 5400 pts. 0x sweep averaging
    #  
    # emptyeA = rf.Network(fileLoc+"11-1-2023/emptyeA.s2p")
    # ball30mme = rf.Network(fileLoc+"11-1-2023/ball30mme.s2p")
    # emptyeB = rf.Network(fileLoc+"11-1-2023/emptyeB.s2p")
    # ball25mme = rf.Network(fileLoc+"11-1-2023/ball25mme.s2p")
    # emptyeC = rf.Network(fileLoc+"11-1-2023/emptyeC.s2p")
    # ball20mme = rf.Network(fileLoc+"11-1-2023/ball20mme.s2p")
    # emptyeD = rf.Network(fileLoc+"11-1-2023/emptyeD.s2p")
    # ball15mme = rf.Network(fileLoc+"11-1-2023/ball15mme.s2p")
    # emptyeE = rf.Network(fileLoc+"11-1-2023/emptyeE.s2p")
    # ball10mme = rf.Network(fileLoc+"11-1-2023/ball10mme.s2p")
    # emptyeF = rf.Network(fileLoc+"11-1-2023/emptyeF.s2p") ##missing measurement
    # ball875mme = rf.Network(fileLoc+"11-1-2023/ball875mme.s2p")
    # emptyeG = rf.Network(fileLoc+"11-1-2023/emptyeG.s2p")
    # ball75mme = rf.Network(fileLoc+"11-1-2023/ball75mme.s2p")
    # emptyeH = rf.Network(fileLoc+"11-1-2023/emptyeH.s2p")
    # ball635mme = rf.Network(fileLoc+"11-1-2023/ball635mme.s2p")
    # emptyeI = rf.Network(fileLoc+"11-1-2023/emptyeI.s2p")
    # ball5mme = rf.Network(fileLoc+"11-1-2023/ball5mme.s2p")
    # emptyeJ = rf.Network(fileLoc+"11-1-2023/emptyeJ.s2p")
    # ball3975mme = rf.Network(fileLoc+"11-1-2023/ball3975mme.s2p")
    # emptyeK = rf.Network(fileLoc+"11-1-2023/emptyeK.s2p")
    # ball025mme = rf.Network(fileLoc+"11-1-2023/ball025mme.s2p")
    # emptyeL = rf.Network(fileLoc+"11-1-2023/emptyeL.s2p")
    # ball015mme = rf.Network(fileLoc+"11-1-2023/ball015mme.s2p")
    # emptyeM = rf.Network(fileLoc+"11-1-2023/emptyeM.s2p")
    # shortUncloakede = rf.Network(fileLoc+"11-1-2023/short uncloakede.s2p")
    # emptyeN = rf.Network(fileLoc+"11-1-2023/emptyeN.s2p")
    # uncloakede = rf.Network(fileLoc+"11-1-2023/uncloakede.s2p")
    # emptyeO = rf.Network(fileLoc+"11-1-2023/emptyeO.s2p")
    # longCloakede = rf.Network(fileLoc+"11-1-2023/long cloakede.s2p")
    # emptyeP = rf.Network(fileLoc+"11-1-2023/emptyeP.s2p")
    # shortCloakede = rf.Network(fileLoc+"11-1-2023/short cloakede.s2p")
    # emptyeQ = rf.Network(fileLoc+"11-1-2023/emptyeQ.s2p")
    # shortUncloakede2 = rf.Network(fileLoc+"11-1-2023/short uncloakede2.s2p")
    # emptyeN2 = rf.Network(fileLoc+"11-1-2023/emptyeN2.s2p")
    # uncloakede2 = rf.Network(fileLoc+"11-1-2023/uncloakede2.s2p")
    # emptyeO2 = rf.Network(fileLoc+"11-1-2023/emptyeO2.s2p")
    # longCloakede2 = rf.Network(fileLoc+"11-1-2023/long cloakede2.s2p")
    # emptyeP2 = rf.Network(fileLoc+"11-1-2023/emptyeP2.s2p")
    # shortCloakede2 = rf.Network(fileLoc+"11-1-2023/short cloakede2.s2p")
    # emptyeQ2 = rf.Network(fileLoc+"11-1-2023/emptyeQ2.s2p") ###short break after this one. 
    # shortUncloakede3 = rf.Network(fileLoc+"11-1-2023/short uncloakede3.s2p") ##for these next antenna measurements, I tried to leave it sticking out of the holder as much as I could
    # emptyeN3 = rf.Network(fileLoc+"11-1-2023/emptyeN3.s2p")
    # uncloakede3 = rf.Network(fileLoc+"11-1-2023/uncloakede3.s2p")
    # emptyeO3 = rf.Network(fileLoc+"11-1-2023/emptyeO3.s2p")
    # longCloakede3 = rf.Network(fileLoc+"11-1-2023/long cloakede3.s2p")
    # emptyeP3 = rf.Network(fileLoc+"11-1-2023/emptyeP3.s2p")
    # shortCloakede3 = rf.Network(fileLoc+"11-1-2023/short cloakede3.s2p")
    # emptyeQ3 = rf.Network(fileLoc+"11-1-2023/emptyeQ3.s2p")
    # emptyeR3 = rf.Network(fileLoc+"11-1-2023/emptyeR3.s2p")
    # emptyeS3 = rf.Network(fileLoc+"11-1-2023/emptyeS3.s2p")
    # emptyeT3 = rf.Network(fileLoc+"11-1-2023/emptyeT3.s2p")
    # emptyeU3 = rf.Network(fileLoc+"11-1-2023/emptyeU3.s2p")
    # emptyeV3 = rf.Network(fileLoc+"11-1-2023/emptyeV3.s2p")
    # emptyeW3 = rf.Network(fileLoc+"11-1-2023/emptyeW3.s2p")
    # emptyeX3 = rf.Network(fileLoc+"11-1-2023/emptyeX3.s2p")
    # emptyeY3 = rf.Network(fileLoc+"11-1-2023/emptyeY3.s2p")
    #  
    # ##now we moved the antennas closer together, allowing us to also eliminate one of the longer cables previously needed. New distance: approx. 167 cm to target from front of horn
    #  
    # ###spheres again:  IF BW = 70Hz, 7-15GHz, 5 dB power, 2500 pts. 2x sweep averaging
    # emptyfA = rf.Network(fileLoc+"11-1-2023/emptyFA.s2p")
    # emptyfA2 = rf.Network(fileLoc+"11-1-2023/emptyfA2.s2p")
    # emptyfA3 = rf.Network(fileLoc+"11-1-2023/emptyfA3.s2p")
    # ball30mmf = rf.Network(fileLoc+"11-1-2023/ball30mmf.s2p")
    # emptyfB = rf.Network(fileLoc+"11-1-2023/emptyfB.s2p")
    # ball25mmf = rf.Network(fileLoc+"11-1-2023/ball25mmf.s2p")
    # emptyfC = rf.Network(fileLoc+"11-1-2023/emptyfC.s2p")
    # ball20mmf = rf.Network(fileLoc+"11-1-2023/ball20mmf.s2p")
    # emptyfD = rf.Network(fileLoc+"11-1-2023/emptyfD.s2p")
    # ball15mmf = rf.Network(fileLoc+"11-1-2023/ball15mmf.s2p")
    # emptyfE = rf.Network(fileLoc+"11-1-2023/emptyfE.s2p")
    # ball10mmf = rf.Network(fileLoc+"11-1-2023/ball10mmf.s2p")
    # emptyfF = rf.Network(fileLoc+"11-1-2023/emptyfF.s2p") ##missing measurement
    # ball875mmf = rf.Network(fileLoc+"11-1-2023/ball875mmf.s2p")
    # emptyfG = rf.Network(fileLoc+"11-1-2023/emptyfG.s2p")
    # ball75mmf = rf.Network(fileLoc+"11-1-2023/ball75mmf.s2p")
    # emptyfH = rf.Network(fileLoc+"11-1-2023/emptyfH.s2p")
    # ball635mmf = rf.Network(fileLoc+"11-1-2023/ball635mmf.s2p")
    # emptyfI = rf.Network(fileLoc+"11-1-2023/emptyfI.s2p")
    # ball5mmf = rf.Network(fileLoc+"11-1-2023/ball5mmf.s2p")
    # emptyfJ = rf.Network(fileLoc+"11-1-2023/emptyfJ.s2p")
    # ball3975mmf = rf.Network(fileLoc+"11-1-2023/ball3975mmf.s2p")
    # emptyfK = rf.Network(fileLoc+"11-1-2023/emptyfK.s2p")
    # ball025mmf = rf.Network(fileLoc+"11-1-2023/ball025mmf.s2p")
    # emptyfL = rf.Network(fileLoc+"11-1-2023/emptyfL.s2p")
    # ball015mmf = rf.Network(fileLoc+"11-1-2023/ball015mmf.s2p")
    # emptyfM = rf.Network(fileLoc+"11-1-2023/emptyfM.s2p")
    #  
    # ##break, the continue with a measurement of the large spheres and some for smaller, then antennas over and over again.
    # emptygA = rf.Network(fileLoc+"11-1-2023/emptygA.s2p")
    # ball35mmg = rf.Network(fileLoc+"11-1-2023/ball35mmg.s2p")
    # emptygB = rf.Network(fileLoc+"11-1-2023/emptygB.s2p")
    # ball15mmg = rf.Network(fileLoc+"11-1-2023/ball15mmg.s2p")
    # emptygC = rf.Network(fileLoc+"11-1-2023/emptygC.s2p")
    # ball75mmg = rf.Network(fileLoc+"11-1-2023/ball75mmg.s2p")
    # emptygD = rf.Network(fileLoc+"11-1-2023/emptygD.s2p")
    # ball5mmg = rf.Network(fileLoc+"11-1-2023/ball5mmg.s2p")
    # emptygE = rf.Network(fileLoc+"11-1-2023/emptygE.s2p")
    # ball3975mmg = rf.Network(fileLoc+"11-1-2023/ball3975mmg.s2p")
    # emptygF = rf.Network(fileLoc+"11-1-2023/emptygF.s2p")
    # ball025mmg = rf.Network(fileLoc+"11-1-2023/ball025mmg.s2p")
    # emptygF2 = rf.Network(fileLoc+"11-1-2023/emptygF2.s2p")
    # shortUncloakedg = rf.Network(fileLoc+"11-1-2023/short uncloakedg.s2p")
    # emptygG = rf.Network(fileLoc+"11-1-2023/emptygG.s2p")
    # uncloakedg = rf.Network(fileLoc+"11-1-2023/uncloakedg.s2p")
    # emptygH = rf.Network(fileLoc+"11-1-2023/emptygH.s2p")
    # longCloakedg = rf.Network(fileLoc+"11-1-2023/long cloakedg.s2p")
    # emptygI = rf.Network(fileLoc+"11-1-2023/emptygI.s2p")
    # shortCloakedg = rf.Network(fileLoc+"11-1-2023/short cloakedg.s2p")
    # emptygJ = rf.Network(fileLoc+"11-1-2023/emptygJ.s2p")
    # shortUncloakedg2 = rf.Network(fileLoc+"11-1-2023/short uncloakedg2.s2p")
    # emptygG2 = rf.Network(fileLoc+"11-1-2023/emptygG2.s2p")
    # uncloakedg2 = rf.Network(fileLoc+"11-1-2023/uncloakedg2.s2p")
    # emptygH2 = rf.Network(fileLoc+"11-1-2023/emptygH2.s2p")
    # longCloakedg2 = rf.Network(fileLoc+"11-1-2023/long cloakedg2.s2p")
    # emptygI2 = rf.Network(fileLoc+"11-1-2023/emptygI2.s2p")
    # shortCloakedg2 = rf.Network(fileLoc+"11-1-2023/short cloakedg2.s2p")
    # emptygJ2 = rf.Network(fileLoc+"11-1-2023/emptygJ2.s2p")
    # shortUncloakedg3 = rf.Network(fileLoc+"11-1-2023/short uncloakedg3.s2p")
    # emptygG3 = rf.Network(fileLoc+"11-1-2023/emptygG3.s2p")
    # uncloakedg3 = rf.Network(fileLoc+"11-1-2023/uncloakedg3.s2p")
    # emptygH3 = rf.Network(fileLoc+"11-1-2023/emptygH3.s2p")
    # longCloakedg3 = rf.Network(fileLoc+"11-1-2023/long cloakedg3.s2p")
    # emptygI3 = rf.Network(fileLoc+"11-1-2023/emptygI3.s2p")
    # shortCloakedg3 = rf.Network(fileLoc+"11-1-2023/short cloakedg3.s2p")
    # emptygJ3 = rf.Network(fileLoc+"11-1-2023/emptygJ3.s2p")
    #  
    #  
    # ##one final set with antennas lying face-up on the block, rather than in holder facing wall antenna. no break between this and previous measurements
    # emptygF5 = rf.Network(fileLoc+"11-1-2023/emptygF5.s2p")
    # shortUncloakedg5 = rf.Network(fileLoc+"11-1-2023/short uncloakedg5.s2p")
    # emptygG5 = rf.Network(fileLoc+"11-1-2023/emptygG5.s2p")
    # uncloakedg5 = rf.Network(fileLoc+"11-1-2023/uncloakedg5.s2p")
    # emptygH5 = rf.Network(fileLoc+"11-1-2023/emptygH5.s2p")
    # longCloakedg55 = rf.Network(fileLoc+"11-1-2023/long cloakedg5.s2p")
    # emptygI5 = rf.Network(fileLoc+"11-1-2023/emptygI5.s2p")
    # shortCloakedg5 = rf.Network(fileLoc+"11-1-2023/short cloakedg5.s2p")
    # emptygJ5 = rf.Network(fileLoc+"11-1-2023/emptygJ5.s2p")
    # justHolderg = rf.Network(fileLoc+"11-1-2023/just holderg.s2p")
    # emptygJ6 = rf.Network(fileLoc+"11-1-2023/emptygJ6.s2p")
    # justHolderg2 = rf.Network(fileLoc+"11-1-2023/just holderg2.s2p")
    # emptygJ7 = rf.Network(fileLoc+"11-1-2023/emptygJ7.s2p")
    #  
    # ###another day, coming back for measurements. we now have an absober on the receiving antenna's backplate and the front tripod legs.
    # ##  IF BW = 50Hz, 6.6-15GHz, -5 dBm power, 1001 pts., 0x sweep averaging
    #  
    # ball30mmSetE = rf.Network(fileLoc+"20-1-2023/ball30mmSetE.s2p")
    # ball30mmeSetE = rf.Network(fileLoc+"20-1-2023/ball30mmeSetE.s2p")
    # ball10mmSetE = rf.Network(fileLoc+"20-1-2023/ball10mmSetE.s2p")
    # ball10mmeSetE = rf.Network(fileLoc+"20-1-2023/ball10mmeSetE.s2p")
    # ball5mmSetE = rf.Network(fileLoc+"20-1-2023/ball5mmSetE.s2p")
    # ball5mmeSetE = rf.Network(fileLoc+"20-1-2023/ball5mmeSetE.s2p")
    # ball025mmSetE = rf.Network(fileLoc+"20-1-2023/ball025mmSetE.s2p")
    # ball025mmeSetE = rf.Network(fileLoc+"20-1-2023/ball025mmeSetE.s2p")
    # ball015mmSetE = rf.Network(fileLoc+"20-1-2023/ball015mmSetE.s2p")
    # ball015mmeSetE = rf.Network(fileLoc+"20-1-2023/ball015mmeSetE.s2p")
    #  
    # ##  IF BW = 50Hz, 6.6-15GHz, -15 dBm power, 1001 pts., 0x sweep averaging
    # ball30mmSetF = rf.Network(fileLoc+"20-1-2023/ball30mmSetF.s2p")
    # ball30mmeSetF = rf.Network(fileLoc+"20-1-2023/ball30mmeSetF.s2p")
    # ball10mmSetF = rf.Network(fileLoc+"20-1-2023/ball10mmSetF.s2p")
    # ball10mmeSetF = rf.Network(fileLoc+"20-1-2023/ball10mmeSetF.s2p")
    # ball5mmSetF = rf.Network(fileLoc+"20-1-2023/ball5mmSetF.s2p")
    # ball5mmeSetF = rf.Network(fileLoc+"20-1-2023/ball5mmeSetF.s2p")
    # ball3975mmSetF = rf.Network(fileLoc+"20-1-2023/ball3975mmSetF.s2p")
    # ball3975mmeSetF = rf.Network(fileLoc+"20-1-2023/ball3975mmeSetF.s2p")
    # ball025mmSetF = rf.Network(fileLoc+"20-1-2023/ball025mmSetF.s2p")
    # ball025mmeSetF = rf.Network(fileLoc+"20-1-2023/ball025mmeSetF.s2p")
    # ball015mmSetF = rf.Network(fileLoc+"20-1-2023/ball015mmSetF.s2p")
    # ball015mmeSetF = rf.Network(fileLoc+"20-1-2023/ball015mmeSetF.s2p")
    #  
    # ##  IF BW = 50Hz, 6.6-15GHz, -10 dBm power, 1001 pts., 0x sweep averaging
    # ball30mmSetG = rf.Network(fileLoc+"20-1-2023/ball30mmSetG.s2p")
    # ball30mmeSetG = rf.Network(fileLoc+"20-1-2023/ball30mmeSetG.s2p")
    # ball10mmSetG = rf.Network(fileLoc+"20-1-2023/ball10mmSetG.s2p")
    # ball10mmeSetG = rf.Network(fileLoc+"20-1-2023/ball10mmeSetG.s2p")
    # ball5mmSetG = rf.Network(fileLoc+"20-1-2023/ball5mmSetG.s2p")
    # ball5mmeSetG = rf.Network(fileLoc+"20-1-2023/ball5mmeSetG.s2p")
    # ball3975mmSetG = rf.Network(fileLoc+"20-1-2023/ball3975mmSetG.s2p")
    # ball3975mmeSetG = rf.Network(fileLoc+"20-1-2023/ball3975mmeSetG.s2p")
    # ball025mmSetG = rf.Network(fileLoc+"20-1-2023/ball025mmSetG.s2p")
    # ball025mmeSetG = rf.Network(fileLoc+"20-1-2023/ball025mmeSetG.s2p")
    # ball015mmSetG = rf.Network(fileLoc+"20-1-2023/ball015mmSetG.s2p")
    # ball015mmeSetG = rf.Network(fileLoc+"20-1-2023/ball015mmeSetG.s2p")
    #  
    # ##  IF BW = 50Hz, 6.6-15GHz, -10 dBm power, 1001 pts., 0x sweep averaging
    # ball30mmSetG = rf.Network(fileLoc+"20-1-2023/ball30mmSetG.s2p")
    # ball30mmeSetG = rf.Network(fileLoc+"20-1-2023/ball30mmeSetG.s2p")
    # ball10mmSetG = rf.Network(fileLoc+"20-1-2023/ball10mmSetG.s2p")
    # ball10mmeSetG = rf.Network(fileLoc+"20-1-2023/ball10mmeSetG.s2p")
    # ball5mmSetG = rf.Network(fileLoc+"20-1-2023/ball5mmSetG.s2p")
    # ball5mmeSetG = rf.Network(fileLoc+"20-1-2023/ball5mmeSetG.s2p")
    # ball3975mmSetG = rf.Network(fileLoc+"20-1-2023/ball3975mmSetG.s2p")
    # ball3975mmeSetG = rf.Network(fileLoc+"20-1-2023/ball3975mmeSetG.s2p")
    # ball025mmSetG = rf.Network(fileLoc+"20-1-2023/ball025mmSetG.s2p")
    # ball025mmeSetG = rf.Network(fileLoc+"20-1-2023/ball025mmeSetG.s2p")
    # ball015mmSetG = rf.Network(fileLoc+"20-1-2023/ball015mmSetG.s2p")
    # ball015mmeSetG = rf.Network(fileLoc+"20-1-2023/ball015mmeSetG.s2p")
    #  
    # ##  IF BW = 20Hz, 6.6-15GHz, -10 dBm power, 5000 pts., 0x sweep averaging
    # ball30mmSetH = rf.Network(fileLoc+"20-1-2023/ball30mmSetH.s2p")
    # ball30mmeSetH = rf.Network(fileLoc+"20-1-2023/ball30mmeSetH.s2p")
    # ball10mmSetH = rf.Network(fileLoc+"20-1-2023/ball10mmSetH.s2p")
    # ball10mmeSetH = rf.Network(fileLoc+"20-1-2023/ball10mmeSetH.s2p")
    # ball5mmSetH = rf.Network(fileLoc+"20-1-2023/ball5mmSetH.s2p")
    # ball5mmeSetH = rf.Network(fileLoc+"20-1-2023/ball5mmeSetH.s2p")
    # ball3975mmSetH = rf.Network(fileLoc+"20-1-2023/ball3975mmSetH.s2p")
    # ball3975mmeSetH = rf.Network(fileLoc+"20-1-2023/ball3975mmeSetH.s2p")
    # ball025mmSetH = rf.Network(fileLoc+"20-1-2023/ball025mmSetH.s2p")
    # ball025mmeSetH = rf.Network(fileLoc+"20-1-2023/ball025mmeSetH.s2p")
    # ball015mmSetH = rf.Network(fileLoc+"20-1-2023/ball015mmSetH.s2p")
    # ball015mmeSetH = rf.Network(fileLoc+"20-1-2023/ball015mmeSetH.s2p")
    #  
    # ##  IF BW = 200Hz, 6.6-15GHz, -10 dBm power, 1000 pts., 0x sweep averaging
    # ball30mmSetI = rf.Network(fileLoc+"20-1-2023/ball30mmSetI.s2p")
    # ball30mmeSetI = rf.Network(fileLoc+"20-1-2023/ball30mmeSetI.s2p")
    # ball10mmSetI = rf.Network(fileLoc+"20-1-2023/ball10mmSetI.s2p")
    # ball10mmeSetI = rf.Network(fileLoc+"20-1-2023/ball10mmeSetI.s2p")
    # ball5mmSetI = rf.Network(fileLoc+"20-1-2023/ball5mmSetI.s2p")
    # ball5mmeSetI = rf.Network(fileLoc+"20-1-2023/ball5mmeSetI.s2p")
    # ball3975mmSetI = rf.Network(fileLoc+"20-1-2023/ball3975mmSetI.s2p")
    # ball3975mmeSetI = rf.Network(fileLoc+"20-1-2023/ball3975mmeSetI.s2p")
    # ball025mmSetI = rf.Network(fileLoc+"20-1-2023/ball025mmSetI.s2p")
    # ball025mmeSetI = rf.Network(fileLoc+"20-1-2023/ball025mmeSetI.s2p")
    #  
    # ##  IF BW = 30Hz, 6.6-15GHz, 3 dBm power, 1000 pts., 0x sweep averaging
    # ball30mmSetJ = rf.Network(fileLoc+"20-1-2023/ball30mmSetJ.s2p")
    # ball30mmeSetJ = rf.Network(fileLoc+"20-1-2023/ball30mmeSetJ.s2p")
    # ball10mmSetJ = rf.Network(fileLoc+"20-1-2023/ball10mmSetJ.s2p")
    # ball10mmeSetJ = rf.Network(fileLoc+"20-1-2023/ball10mmeSetJ.s2p")
    # ball5mmSetJ = rf.Network(fileLoc+"20-1-2023/ball5mmSetJ.s2p")
    # ball5mmeSetJ = rf.Network(fileLoc+"20-1-2023/ball5mmeSetJ.s2p")
    # ball3975mmSetJ = rf.Network(fileLoc+"20-1-2023/ball3975mmSetJ.s2p")
    # ball3975mmeSetJ = rf.Network(fileLoc+"20-1-2023/ball3975mmeSetJ.s2p")
    # ball025mmSetJ = rf.Network(fileLoc+"20-1-2023/ball025mmSetJ.s2p")
    # ball025mmeSetJ = rf.Network(fileLoc+"20-1-2023/ball025mmeSetJ.s2p")
    #===========================================================================
    
    
    ##  IF BW = 30Hz, 6.6-15GHz, 3 dBm power, 1000 pts., 0x sweep averaging
    uncloakedSetK = rf.Network(fileLoc+"20-1-2023/uncloakedSetK.s2p")
    uncloakedeSetK = rf.Network(fileLoc+"20-1-2023/uncloakedeSetK.s2p")
    shortUncloakedSetK = rf.Network(fileLoc+"20-1-2023/short uncloakedSetK.s2p")
    shortUncloakedeSetK = rf.Network(fileLoc+"20-1-2023/short uncloakedeSetK.s2p")
    shortCloakedSetK = rf.Network(fileLoc+"20-1-2023/short cloakedSetK.s2p")
    shortCloakedeSetK = rf.Network(fileLoc+"20-1-2023/short cloakedeSetK.s2p")
    longCloakedSetK = rf.Network(fileLoc+"20-1-2023/long cloakedSetK.s2p")
    longCloakedeSetK = rf.Network(fileLoc+"20-1-2023/long cloakedeSetK.s2p")
    ball30mmSetK = rf.Network(fileLoc+"20-1-2023/ball30mmSetK.s2p")
    ball30mmeSetK = rf.Network(fileLoc+"20-1-2023/ball30mmeSetK.s2p")
    ball25mmSetK = rf.Network(fileLoc+"20-1-2023/ball25mmSetK.s2p")
    ball25mmeSetK = rf.Network(fileLoc+"20-1-2023/ball25mmeSetK.s2p")
    ball20mmSetK = rf.Network(fileLoc+"20-1-2023/ball20mmSetK.s2p")
    ball20mmeSetK = rf.Network(fileLoc+"20-1-2023/ball20mmeSetK.s2p")
    ball15mmSetK = rf.Network(fileLoc+"20-1-2023/ball15mmSetK.s2p")
    ball15mmeSetK = rf.Network(fileLoc+"20-1-2023/ball15mmeSetK.s2p")
    ball10mmSetK = rf.Network(fileLoc+"20-1-2023/ball10mmSetK.s2p")
    ball10mmeSetK = rf.Network(fileLoc+"20-1-2023/ball10mmeSetK.s2p")
    ball875mmSetK = rf.Network(fileLoc+"20-1-2023/ball875mmSetK.s2p")
    ball875mmeSetK = rf.Network(fileLoc+"20-1-2023/ball875mmeSetK.s2p")
    ball635mmSetK = rf.Network(fileLoc+"20-1-2023/ball635mmSetK.s2p")
    ball635mmeSetK = rf.Network(fileLoc+"20-1-2023/ball635mmeSetK.s2p")
    ball5mmSetK = rf.Network(fileLoc+"20-1-2023/ball5mmSetK.s2p")
    ball5mmeSetK = rf.Network(fileLoc+"20-1-2023/ball5mmeSetK.s2p")
    ball3975mmSetK = rf.Network(fileLoc+"20-1-2023/ball3975mmSetK.s2p")
    ball3975mmeSetK = rf.Network(fileLoc+"20-1-2023/ball3975mmeSetK.s2p")
    ball025mmSetK = rf.Network(fileLoc+"20-1-2023/ball025mmSetK.s2p")
    ball025mmeSetK = rf.Network(fileLoc+"20-1-2023/ball025mmeSetK.s2p")
    ball015mmSetK = rf.Network(fileLoc+"20-1-2023/ball015mmSetK.s2p")
    ball015mmeSetK = rf.Network(fileLoc+"20-1-2023/ball015mmeSetK.s2p")
    uncloakedSetK2 = rf.Network(fileLoc+"20-1-2023/uncloakedSetK2.s2p")
    uncloakedeSetK2 = rf.Network(fileLoc+"20-1-2023/uncloakedeSetK2.s2p")
    shortUncloakedSetK2 = rf.Network(fileLoc+"20-1-2023/short uncloakedSetK2.s2p")
    shortUncloakedeSetK2 = rf.Network(fileLoc+"20-1-2023/short uncloakedeSetK2.s2p")
    shortCloakedSetK2 = rf.Network(fileLoc+"20-1-2023/short cloakedSetK2.s2p")
    shortCloakedeSetK2 = rf.Network(fileLoc+"20-1-2023/short cloakedeSetK2.s2p")
    longCloakedSetK2 = rf.Network(fileLoc+"20-1-2023/long cloakedSetK2.s2p")
    longCloakedeSetK2 = rf.Network(fileLoc+"20-1-2023/long cloakedeSetK2.s2p")
    ball50mmSetK2 = rf.Network(fileLoc+"20-1-2023/ball50mmSetK2.s2p")
    ball50mmeSetK2 = rf.Network(fileLoc+"20-1-2023/ball50mmeSetK2.s2p")
    ball5mmSetK2 = rf.Network(fileLoc+"20-1-2023/ball5mmSetK2.s2p")
    ball5mmeSetK2 = rf.Network(fileLoc+"20-1-2023/ball5mmeSetK2.s2p")
    ball3975mmSetK2 = rf.Network(fileLoc+"20-1-2023/ball3975mmSetK2.s2p")
    ball3975mmeSetK2 = rf.Network(fileLoc+"20-1-2023/ball3975mmeSetK2.s2p")
    ball025mmSetK2 = rf.Network(fileLoc+"20-1-2023/ball025mmSetK2.s2p")
    ball025mmeSetK2 = rf.Network(fileLoc+"20-1-2023/ball025mmeSetK2.s2p")
    ball015mmSetK2 = rf.Network(fileLoc+"20-1-2023/ball015mmSetK2.s2p")
    ball015mmeSetK2 = rf.Network(fileLoc+"20-1-2023/ball015mmeSetK2.s2p")
    uncloakedSetK3 = rf.Network(fileLoc+"20-1-2023/uncloakedSetK3.s2p")
    uncloakedeSetK3 = rf.Network(fileLoc+"20-1-2023/uncloakedeSetK3.s2p")
    shortUncloakedSetK3 = rf.Network(fileLoc+"20-1-2023/short uncloakedSetK3.s2p")
    shortUncloakedeSetK3 = rf.Network(fileLoc+"20-1-2023/short uncloakedeSetK3.s2p")
    shortCloakedSetK3 = rf.Network(fileLoc+"20-1-2023/short cloakedSetK3.s2p")
    shortCloakedeSetK3 = rf.Network(fileLoc+"20-1-2023/short cloakedeSetK3.s2p")
    longCloakedSetK3 = rf.Network(fileLoc+"20-1-2023/long cloakedSetK3.s2p")
    longCloakedeSetK3 = rf.Network(fileLoc+"20-1-2023/long cloakedeSetK3.s2p")
    
    backgroundsTimePassed = ((1*60 + 26)*60 + 46) ## in s, 0:13:16-22:46:30
    listOfBackgrounds = [uncloakedeSetK,shortUncloakedeSetK,shortCloakedeSetK,longCloakedeSetK,ball30mmeSetK,ball25mmeSetK,ball20mmeSetK,ball15mmeSetK,ball10mmeSetK,ball875mmeSetK,ball635mmeSetK,ball5mmeSetK,ball3975mmeSetK,ball025mmeSetK,ball015mmeSetK,uncloakedeSetK2,shortUncloakedeSetK2,shortCloakedeSetK2,longCloakedeSetK2,ball50mmeSetK2,ball5mmeSetK2,ball3975mmeSetK2,ball025mmeSetK2,ball015mmeSetK2,uncloakedeSetK3,shortUncloakedeSetK3,shortCloakedeSetK3,longCloakedeSetK3]
    listOfNonBackgrounds = [uncloakedSetK, shortUncloakedSetK, shortCloakedSetK, longCloakedSetK, ball30mmSetK, ball25mmSetK, ball20mmSetK, ball15mmSetK, ball10mmSetK, ball875mmSetK, ball635mmSetK, ball5mmSetK, ball3975mmSetK, ball025mmSetK, ball015mmSetK, uncloakedSetK2, shortUncloakedSetK2, shortCloakedSetK2, longCloakedSetK2, ball50mmSetK2, ball5mmSetK2, ball3975mmSetK2, ball025mmSetK2, ball015mmSetK2, uncloakedSetK3, shortUncloakedSetK3, shortCloakedSetK3, longCloakedSetK3]
    
    
    #===========================================================================
    # ###Rasmus Measurements
    # ## these were IF BW = 50Hz, 6.6-15GHz, -15 dB power, 1001 pts., 0x sweep averaging
    # fileLocR = 'C:/Users/al8032pa/Work Folders/Downloads/Rasmus Data and Code/170123/'
    # 
    # ball30mmR = rf.Network(fileLocR+"xband_sphere_r3o0.s2p")
    # ball30mmRe = rf.Network(fileLocR+"xband_sphere_r3o0e.s2p")
    # ball25mmR = rf.Network(fileLocR+"xband_sphere_r2o5.s2p")
    # ball25mmRe = rf.Network(fileLocR+"xband_sphere_r2o5e.s2p")
    # ball20mmR = rf.Network(fileLocR+"xband_sphere_r2o0.s2p")
    # ball20mmRe = rf.Network(fileLocR+"xband_sphere_r2o0e.s2p")
    # ball15mmR = rf.Network(fileLocR+"xband_sphere_r1o5.s2p")
    # ball15mmRe = rf.Network(fileLocR+"xband_sphere_r1o5e.s2p")
    # ball10mmR = rf.Network(fileLocR+"xband_sphere_r1o0.s2p")
    # ball10mmRe = rf.Network(fileLocR+"xband_sphere_r1o0e.s2p")
    # ball875mmR = rf.Network(fileLocR+"xband_sphere_r0o875.s2p")
    # ball875mmRe = rf.Network(fileLocR+"xband_sphere_r0o875e.s2p")
    # ball75mmR = rf.Network(fileLocR+"xband_sphere_r0o75.s2p")
    # ball75mmRe = rf.Network(fileLocR+"xband_sphere_r0o75e.s2p")
    # ball635mmR = rf.Network(fileLocR+"xband_sphere_r0o635.s2p")
    # ball635mmRe = rf.Network(fileLocR+"xband_sphere_r0o635e.s2p")
    # ball5mmR = rf.Network(fileLocR+"xband_sphere_r0o5.s2p")
    # ball5mmRe = rf.Network(fileLocR+"xband_sphere_r0o5e.s2p")
    # ball3975mmR = rf.Network(fileLocR+"xband_sphere_r0o3975.s2p")
    # ball3975mmRe = rf.Network(fileLocR+"xband_sphere_r0o3975e.s2p")
    # ball025mmR = rf.Network(fileLocR+"xband_sphere_r0o25.s2p")
    # ball025mmRe = rf.Network(fileLocR+"xband_sphere_r0o25e.s2p")
    #===========================================================================
    
    
    
    print(ball30mmSetK)
    #plot((ball30mmR['7-13ghz'], ball10mmR['7-13ghz'], ball5mmR['7-13ghz']))
    #plot((uncloakedSetK.s21,shortCloakedSetK.s21, uncloakedeSetK.s21))
    #plot(((uncloakedSetK-uncloakedeSetK).s21,(shortCloakedSetK - shortCloakedeSetK).s21))
    #plot(((ball30mmSetK-ball30mmeSetK)['7-13ghz'].s21,(ball5mmSetK-ball5mmeSetK)['7-13ghz'].s21))
    #plot(((ball30mmSetK)['8-12ghz'].s21,(ball5mmSetK)['8-12ghz'].s21))
    #ball30mmSetK.s21 = ball30mmSetK*10
    
    
    dAntennas = 4.65 ##approx, 4.65 front-to-front, 5.3 back-to-back
    hAntennas = 1.25##approx
    pad = [20000, 'hamming']
    trpeakB = 118.6*1e-9 #s
    trpeakBwidth =8.0*1e-9 #s
    
    #===========================================================================
    # f1, sigball15mmB = processOneData(ball15mmB.s21, emptyB4.s21, trpeakB, trpeakBwidth, dAntennas, pad, 'Uncloaked', makePlots = 0)
    #===========================================================================
    
    trpeakballs = 118.5*1e-9 #s
    trpeakballswidth = 7.0*1e-9 #s
    
    #===============================================================================
    # f1, sigball30mm = processOneData(ball30mm.s21, emptyN.s21, trpeakballs, trpeakballswidth, dAntennas, pad, 'b30', makePlots = 0)
    # f1, sigball25mm = processOneData(ball25mm.s21, emptyO.s21, trpeakballs, trpeakballswidth, dAntennas, pad, 'b25', makePlots = 0)
    # f1, sigball20mm = processOneData(ball20mm.s21, emptyP.s21, trpeakballs, trpeakballswidth, dAntennas, pad, 'b20', makePlots = 0)
    # f1, sigball15mm = processOneData(ball15mm.s21, emptyQ.s21, trpeakballs, trpeakballswidth, dAntennas, pad, 'b15', makePlots = 0)
    # f1, sigball10mm = processOneData(ball10mm.s21, emptyQ.s21, trpeakballs, trpeakballswidth, dAntennas, pad, 'b10', makePlots = 0)
    # f1, sigball875mm = processOneData(ball875mm.s21, emptyS.s21, trpeakballs, trpeakballswidth, dAntennas, pad, 'b8.75', makePlots = 0)
    # f1, sigball75mm = processOneData(ball75mm.s21, emptyT.s21, trpeakballs, trpeakballswidth, dAntennas, pad, 'b7.5', makePlots = 0)
    # f1, sigball635mm = processOneData(ball635mm.s21, emptyU.s21, trpeakballs, trpeakballswidth, dAntennas, pad, 'b6.35', makePlots = 0)
    # f1, sigball5mm = processOneData(ball5mm.s21, emptyV.s21, trpeakballs, trpeakballswidth, dAntennas, pad, 'b5', makePlots = 0)
    # f1, sigball3975mm = processOneData(ball3975mm.s21, emptyW.s21, trpeakballs, trpeakballswidth, dAntennas, pad, 'b3.975', makePlots = 0)
    # f1, sigball025mm = processOneData(ball025mm.s21, emptyX.s21, trpeakballs, trpeakballswidth, dAntennas, pad, 'b2.5', makePlots = 0)
    # f1, sigball015mm = processOneData(ball015mm.s21, emptyY.s21, trpeakballs, trpeakballswidth, dAntennas, pad, 'b1.5', makePlots = 0)
    #===============================================================================
    
    #===============================================================================
    # f1, sigball30mm = processOneData(ball30mm.s21, emptyN.s21, trpeakballs, trpeakballswidth, dAntennas, pad, 'b30', makePlots = 0, sphereCalData=ball15mm.s21)
    # f1, sigball25mm = processOneData(ball25mm.s21, emptyO.s21, trpeakballs, trpeakballswidth, dAntennas, pad, 'b25', makePlots = 0, sphereCalData=ball15mm.s21)
    # f1, sigball20mm = processOneData(ball20mm.s21, emptyP.s21, trpeakballs, trpeakballswidth, dAntennas, pad, 'b20', makePlots = 0, sphereCalData=ball15mm.s21)
    # f1, sigball15mm = processOneData(ball15mm.s21, emptyQ.s21, trpeakballs, trpeakballswidth, dAntennas, pad, 'b15', makePlots = 0, sphereCalData=ball15mm.s21)
    # f1, sigball10mm = processOneData(ball10mm.s21, emptyQ.s21, trpeakballs, trpeakballswidth, dAntennas, pad, 'b10', makePlots = 0, sphereCalData=ball15mm.s21)
    # f1, sigball875mm = processOneData(ball875mm.s21, emptyS.s21, trpeakballs, trpeakballswidth, dAntennas, pad, 'b8.75', makePlots = 0, sphereCalData=ball15mm.s21)
    # f1, sigball75mm = processOneData(ball75mm.s21, emptyT.s21, trpeakballs, trpeakballswidth, dAntennas, pad, 'b7.5', makePlots = 0, sphereCalData=ball15mm.s21)
    # f1, sigball635mm = processOneData(ball635mm.s21, emptyU.s21, trpeakballs, trpeakballswidth, dAntennas, pad, 'b6.35', makePlots = 0, sphereCalData=ball15mm.s21)
    # f1, sigball5mm = processOneData(ball5mm.s21, emptyV.s21, trpeakballs, trpeakballswidth, dAntennas, pad, 'b5', makePlots = 0, sphereCalData=ball15mm.s21)
    # f1, sigball3975mm = processOneData(ball3975mm.s21, emptyW.s21, trpeakballs, trpeakballswidth, dAntennas, pad, 'b3.975', makePlots = 0, sphereCalData=ball15mm.s21)
    # f1, sigball025mm = processOneData(ball025mm.s21, emptyX.s21, trpeakballs, trpeakballswidth, dAntennas, pad, 'b2.5', makePlots = 0, sphereCalData=ball15mm.s21)
    # f1, sigball015mm = processOneData(ball015mm.s21, emptyY.s21, trpeakballs, trpeakballswidth, dAntennas, pad, 'b1.5', makePlots = 0, sphereCalData=ball15mm.s21)
    #===============================================================================
    
    
    #===============================================================================
    # f1, sigball30mm = processOneData(balla30mm.s21, emptyaD.s21, trpeakballs, trpeakballswidth, dAntennas, pad, 'b30', makePlots = 0)
    # f1, sigball635mm = processOneData(balla635mm.s21, emptyaE.s21, trpeakballs, trpeakballswidth, dAntennas, pad, 'b6.35', makePlots = 0)
    # f1, sigball5mm = processOneData(balla5mm.s21, emptyaF.s21, trpeakballs, trpeakballswidth, dAntennas, pad, 'b5', makePlots = 0)
    # f1, sigball3975mm = processOneData(balla3975mm.s21, emptyaG.s21, trpeakballs, trpeakballswidth, dAntennas, pad, 'b3.975', makePlots = 0)
    # f1, sigball025mm = processOneData(balla025mm.s21, emptyaH.s21, trpeakballs, trpeakballswidth, dAntennas, pad, 'b2.5', makePlots = 0)
    #===============================================================================
    
    
    #===============================================================================
    # f1, sigball3975mm = processOneData(ballb3975mm.s21, emptyaK.s21, trpeakballs, trpeakballswidth, dAntennas, pad, 'b3.975', makePlots = 0)
    # f1, sigball025mm = processOneData(ballb025mm.s21, emptyaL.s21, trpeakballs, trpeakballswidth, dAntennas, pad, 'b2.5', makePlots = 0)
    #===============================================================================
    
    
    #===============================================================================
    # f1, sigShortUncloakedb= processOneData(shortUncloakedb.s21, emptybB.s21, trpeakballs, trpeakballswidth, dAntennas, pad, 'Uncloaked', makePlots = 0)
    # f1, sigUncloakedb= processOneData(uncloakedb.s21, emptybC.s21, trpeakballs, trpeakballswidth, dAntennas, pad, 'Uncloaked', makePlots = 0)
    # f1, sigLongCloakedb= processOneData(longCloakedb.s21, emptybC.s21, trpeakballs, trpeakballswidth, dAntennas, pad, 'Uncloaked', makePlots = 0)
    # f1, sigShortCloakedb= processOneData(shortCloakedb.s21, emptybE.s21, trpeakballs, trpeakballswidth, dAntennas, pad, 'Uncloaked', makePlots = 0)
    #===============================================================================
    
    
    #===============================================================================
    # f1, sigShortUncloakedd= processOneData(shortUncloakedd.s21, emptydB.s21, trpeakballs, trpeakballswidth, dAntennas, pad, 'Uncloaked', makePlots = 0)
    # f1, sigUncloakedd= processOneData(uncloakedd.s21, emptydC.s21, trpeakballs, trpeakballswidth, dAntennas, pad, 'Uncloaked', makePlots = 0)
    # f1, sigLongCloakedd= processOneData(longCloakedd.s21, emptydD.s21, trpeakballs, trpeakballswidth, dAntennas, pad, 'Uncloaked', makePlots = 0)
    # f1, sigShortCloakedd= processOneData(shortCloakedd.s21, emptydE.s21, trpeakballs, trpeakballswidth, dAntennas, pad, 'Uncloaked', makePlots = 0)
    #===============================================================================
    
    #===========================================================================
    # f1, sigShortUncloakede= processOneData(shortUncloakede.s21, emptyeN.s21, trpeakballs, trpeakballswidth, dAntennas, pad, 'Uncloaked', makePlots = 0)
    # f1, sigUncloakede= processOneData(uncloakede.s21, emptyeO.s21, trpeakballs, trpeakballswidth, dAntennas, pad, 'Uncloaked', makePlots = 0)
    # f1, sigLongCloakede= processOneData(longCloakede.s21, emptyeP.s21, trpeakballs, trpeakballswidth, dAntennas, pad, 'Uncloaked', makePlots = 0)
    # f1, sigShortCloakede= processOneData(shortCloakede.s21, emptyeQ.s21, trpeakballs, trpeakballswidth, dAntennas, pad, 'Uncloaked', makePlots = 0)
    # f1, sigShortUncloakede2= processOneData(shortUncloakede2.s21, emptyeN2.s21, trpeakballs, trpeakballswidth, dAntennas, pad, 'Uncloaked', makePlots = 0)
    # f1, sigUncloakede2= processOneData(uncloakede2.s21, emptyeO2.s21, trpeakballs, trpeakballswidth, dAntennas, pad, 'Uncloaked', makePlots = 0)
    # f1, sigLongCloakede2= processOneData(longCloakede2.s21, emptyeP2.s21, trpeakballs, trpeakballswidth, dAntennas, pad, 'Uncloaked', makePlots = 0)
    # f1, sigShortCloakede2= processOneData(shortCloakede2.s21, emptyeQ2.s21, trpeakballs, trpeakballswidth, dAntennas, pad, 'Uncloaked', makePlots = 0)
    # f1, sigShortUncloakede3= processOneData(shortUncloakede3.s21, emptyeN3.s21, trpeakballs, trpeakballswidth, dAntennas, pad, 'Uncloaked', makePlots = 0)
    # f1, sigUncloakede3= processOneData(uncloakede3.s21, emptyeO3.s21, trpeakballs, trpeakballswidth, dAntennas, pad, 'Uncloaked', makePlots = 0)
    # f1, sigLongCloakede3= processOneData(longCloakede3.s21, emptyeP3.s21, trpeakballs, trpeakballswidth, dAntennas, pad, 'Uncloaked', makePlots = 0)
    # f1, sigShortCloakede3= processOneData(shortCloakede3.s21, emptyeQ3.s21, trpeakballs, trpeakballswidth, dAntennas, pad, 'Uncloaked', makePlots = 0)
    #===========================================================================
    
    #===============================================================================
    # f1, sigShortUncloakede= processOneData(shortUncloakede.s21, emptyeN.s21, trpeakballs, trpeakballswidth, dAntennas, pad, 'Uncloaked', makePlots = 0, sphereCalData=[ball15mme.s21, 15e-3])
    # f1, sigUncloakede= processOneData(uncloakede.s21, emptyeO.s21, trpeakballs, trpeakballswidth, dAntennas, pad, 'Uncloaked', makePlots = 0, sphereCalData=[ball15mme.s21, 15e-3])
    # f1, sigLongCloakede= processOneData(longCloakede.s21, emptyeP.s21, trpeakballs, trpeakballswidth, dAntennas, pad, 'Uncloaked', makePlots = 0, sphereCalData=[ball15mme.s21, 15e-3])
    # f1, sigShortCloakede= processOneData(shortCloakede.s21, emptyeQ.s21, trpeakballs, trpeakballswidth, dAntennas, pad, 'Uncloaked', makePlots = 0, sphereCalData=[ball15mme.s21, 15e-3])
    # f1, sigShortUncloakede2= processOneData(shortUncloakede2.s21, emptyeN2.s21, trpeakballs, trpeakballswidth, dAntennas, pad, 'Uncloaked', makePlots = 0, sphereCalData=[ball15mme.s21, 15e-3])
    # f1, sigUncloakede2= processOneData(uncloakede2.s21, emptyeO2.s21, trpeakballs, trpeakballswidth, dAntennas, pad, 'Uncloaked', makePlots = 0, sphereCalData=[ball15mme.s21, 15e-3])
    # f1, sigLongCloakede2= processOneData(longCloakede2.s21, emptyeP2.s21, trpeakballs, trpeakballswidth, dAntennas, pad, 'Uncloaked', makePlots = 0, sphereCalData=[ball15mme.s21, 15e-3])
    # f1, sigShortCloakede2= processOneData(shortCloakede2.s21, emptyeQ2.s21, trpeakballs, trpeakballswidth, dAntennas, pad, 'Uncloaked', makePlots = 0, sphereCalData=[ball15mme.s21, 15e-3])
    # f1, sigShortUncloakede3= processOneData(shortUncloakede3.s21, emptyeN3.s21, trpeakballs, trpeakballswidth, dAntennas, pad, 'Uncloaked', makePlots = 0, sphereCalData=[ball15mme.s21, 15e-3])
    # f1, sigUncloakede3= processOneData(uncloakede3.s21, emptyeO3.s21, trpeakballs, trpeakballswidth, dAntennas, pad, 'Uncloaked', makePlots = 0, sphereCalData=[ball15mme.s21, 15e-3])
    # f1, sigLongCloakede3= processOneData(longCloakede3.s21, emptyeP3.s21, trpeakballs, trpeakballswidth, dAntennas, pad, 'Uncloaked', makePlots = 0, sphereCalData=[ball15mme.s21, 15e-3])
    # f1, sigShortCloakede3= processOneData(shortCloakede3.s21, emptyeQ3.s21, trpeakballs, trpeakballswidth, dAntennas, pad, 'Uncloaked', makePlots = 0, sphereCalData=[ball15mme.s21, 15e-3])
    #===============================================================================
    
    
    #===============================================================================
    # f1, sigball30mme = processOneData(ball30mme.s21, emptyeB.s21, trpeakballs, trpeakballswidth, dAntennas, pad, 'b30', makePlots = 0)
    # f1, sigball25mme = processOneData(ball25mme.s21, emptyeC.s21, trpeakballs, trpeakballswidth, dAntennas, pad, 'b25', makePlots = 0)
    # f1, sigball20mme = processOneData(ball20mme.s21, emptyeD.s21, trpeakballs, trpeakballswidth, dAntennas, pad, 'b20', makePlots = 0)
    # f1, sigball15mme = processOneData(ball15mme.s21, emptyeE.s21, trpeakballs, trpeakballswidth, dAntennas, pad, 'b15', makePlots = 0)
    # f1, sigball10mme = processOneData(ball10mme.s21, emptyeF.s21, trpeakballs, trpeakballswidth, dAntennas, pad, 'b10', makePlots = 0)
    # f1, sigball875mme = processOneData(ball875mme.s21, emptyeG.s21, trpeakballs, trpeakballswidth, dAntennas, pad, 'b8.75', makePlots = 0)
    # f1, sigball75mme = processOneData(ball75mme.s21, emptyeH.s21, trpeakballs, trpeakballswidth, dAntennas, pad, 'b7.5', makePlots = 0)
    # f1, sigball635mme = processOneData(ball635mme.s21, emptyeI.s21, trpeakballs, trpeakballswidth, dAntennas, pad, 'b6.35', makePlots = 0)
    # f1, sigball5mme = processOneData(ball5mme.s21, emptyeJ.s21, trpeakballs, trpeakballswidth, dAntennas, pad, 'b5', makePlots = 0)
    # f1, sigball3975mme = processOneData(ball3975mme.s21, emptyeK.s21, trpeakballs, trpeakballswidth, dAntennas, pad, 'b3.975', makePlots = 0)
    # f1, sigball025mme = processOneData(ball025mme.s21, emptyeL.s21, trpeakballs, trpeakballswidth, dAntennas, pad, 'b2.5', makePlots = 0)
    # f1, sigball015mme = processOneData(ball015mme.s21, emptyeM.s21, trpeakballs, trpeakballswidth, dAntennas, pad, 'b1.5', makePlots = 0)
    #===============================================================================
    
    dAntennas2 = 3.355 ##approx, 3.355 front-to-front, 4.043 back-to-back measured with a laser
    trpeak2 = 84.5*1e-9 #s
    trpeak2width = 9.0*1e-9 #s
    pad = [20000, 'tukey']
    
    #===============================================================================
    # f1, sigball30mmf = processOneData(ball30mmf.s21, emptyfB.s21, trpeak2, trpeak2width, dAntennas2, pad, 'b30', makePlots = 0)
    # f1, sigball25mmf = processOneData(ball25mmf.s21, emptyfC.s21, trpeak2, trpeak2width, dAntennas2, pad, 'b25', makePlots = 0)
    # f1, sigball20mmf = processOneData(ball20mmf.s21, emptyfD.s21, trpeak2, trpeak2width, dAntennas2, pad, 'b20', makePlots = 0)
    # f1, sigball15mmf = processOneData(ball15mmf.s21, emptyfE.s21, trpeak2, trpeak2width, dAntennas2, pad, 'b15', makePlots = 0)
    # f1, sigball10mmf = processOneData(ball10mmf.s21, emptyfF.s21, trpeak2, trpeak2width, dAntennas2, pad, 'b10', makePlots = 0)
    # f1, sigball875mmf = processOneData(ball875mmf.s21, emptyfG.s21, trpeak2, trpeak2width, dAntennas2, pad, 'b8.75', makePlots = 0)
    # f1, sigball75mmf = processOneData(ball75mmf.s21, emptyfH.s21, trpeak2, trpeak2width, dAntennas2, pad, 'b7.5', makePlots = 0)
    # f1, sigball635mmf = processOneData(ball635mmf.s21, emptyfI.s21, trpeak2, trpeak2width, dAntennas2, pad, 'b6.35', makePlots = 0)
    # f1, sigball5mmf = processOneData(ball5mmf.s21, emptyfJ.s21, trpeak2, trpeak2width, dAntennas2, pad, 'b5', makePlots = 0)
    # f1, sigball3975mmf = processOneData(ball3975mmf.s21, emptyfK.s21, trpeak2, trpeak2width, dAntennas2, pad, 'b3.975', makePlots = 0)
    # f1, sigball025mmf = processOneData(ball025mmf.s21, emptyfL.s21, trpeak2, trpeak2width, dAntennas2, pad, 'b2.5', makePlots = 0)
    # f1, sigball015mmf = processOneData(ball015mmf.s21, emptyfM.s21, trpeak2, trpeak2width, dAntennas2, pad, 'b1.5', makePlots = 0)
    #===============================================================================
    
    #===========================================================================
    # f1, sigball30mmf = processOneData(ball30mmf.s21, emptyfB.s21, trpeak2, trpeak2width, dAntennas2, pad, 'b30', makePlots = 0, sphereCalData=[ball10mmf.s21, 10e-3])
    # f1, sigball25mmf = processOneData(ball25mmf.s21, emptyfC.s21, trpeak2, trpeak2width, dAntennas2, pad, 'b25', makePlots = 0, sphereCalData=[ball10mmf.s21, 10e-3])
    # f1, sigball20mmf = processOneData(ball20mmf.s21, emptyfD.s21, trpeak2, trpeak2width, dAntennas2, pad, 'b20', makePlots = 0, sphereCalData=[ball10mmf.s21, 10e-3])
    # f1, sigball15mmf = processOneData(ball15mmf.s21, emptyfE.s21, trpeak2, trpeak2width, dAntennas2, pad, 'b15', makePlots = 0, sphereCalData=[ball10mmf.s21, 10e-3])
    # f1, sigball10mmf = processOneData(ball10mmf.s21, emptyfF.s21, trpeak2, trpeak2width, dAntennas2, pad, 'b10', makePlots = 0, sphereCalData=[ball10mmf.s21, 10e-3])
    # f1, sigball875mmf = processOneData(ball875mmf.s21, emptyfG.s21, trpeak2, trpeak2width, dAntennas2, pad, 'b8.75', makePlots = 0, sphereCalData=[ball10mmf.s21, 10e-3])
    # f1, sigball75mmf = processOneData(ball75mmf.s21, emptyfH.s21, trpeak2, trpeak2width, dAntennas2, pad, 'b7.5', makePlots = 0, sphereCalData=[ball10mmf.s21, 10e-3])
    # f1, sigball635mmf = processOneData(ball635mmf.s21, emptyfI.s21, trpeak2, trpeak2width, dAntennas2, pad, 'b6.35', makePlots = 0, sphereCalData=[ball10mmf.s21, 10e-3])
    # f1, sigball5mmf = processOneData(ball5mmf.s21, emptyfJ.s21, trpeak2, trpeak2width, dAntennas2, pad, 'b5', makePlots = 0, sphereCalData=[ball10mmf.s21, 10e-3])
    # f1, sigball3975mmf = processOneData(ball3975mmf.s21, emptyfK.s21, trpeak2, trpeak2width, dAntennas2, pad, 'b3.975', makePlots = 0, sphereCalData=[ball10mmf.s21, 10e-3])
    # f1, sigball025mmf = processOneData(ball025mmf.s21, emptyfL.s21, trpeak2, trpeak2width, dAntennas2, pad, 'b2.5', makePlots = 0, sphereCalData=[ball10mmf.s21, 10e-3])
    # f1, sigball015mmf = processOneData(ball015mmf.s21, emptyfM.s21, trpeak2, trpeak2width, dAntennas2, pad, 'b1.5', makePlots = 0, sphereCalData=[ball10mmf.s21, 10e-3])
    #===========================================================================
    
    #===============================================================================
    # f1, sigball35mmg = processOneData(ball35mmg.s21, emptygB.s21, trpeak2, trpeak2width, dAntennas2, pad, 'b35', makePlots = 0)
    # f1, sigball15mmg = processOneData(ball15mmg.s21, emptygC.s21, trpeak2, trpeak2width, dAntennas2, pad, 'b15', makePlots = 0)
    # f1, sigball75mmg = processOneData(ball75mmg.s21, emptygD.s21, trpeak2, trpeak2width, dAntennas2, pad, 'b7.5', makePlots = 0)
    # f1, sigball5mmg = processOneData(ball5mmg.s21, emptygE.s21, trpeak2, trpeak2width, dAntennas2, pad, 'b5', makePlots = 0)
    # f1, sigball3975mmg = processOneData(ball3975mmg.s21, emptygF.s21, trpeak2, trpeak2width, dAntennas2, pad, 'b3.975', makePlots = 0)
    # f1, sigball025mmg = processOneData(ball025mmg.s21, emptygG.s21, trpeak2, trpeak2width, dAntennas2, pad, 'b2.5', makePlots = 0)
    #===============================================================================
    
    #===============================================================================
    # f1, sigball35mmg = processOneData(ball35mmg.s21, emptygB.s21, trpeak2, trpeak2width, dAntennas2, pad, 'b35', makePlots = 0, sphereCalData=[ball35mmg.s21, 35e-3])
    # f1, sigball15mmg = processOneData(ball15mmg.s21, emptygC.s21, trpeak2, trpeak2width, dAntennas2, pad, 'b15', makePlots = 0, sphereCalData=[ball35mmg.s21, 35e-3])
    # f1, sigball75mmg = processOneData(ball75mmg.s21, emptygD.s21, trpeak2, trpeak2width, dAntennas2, pad, 'b7.5', makePlots = 0, sphereCalData=[ball35mmg.s21, 35e-3])
    # f1, sigball5mmg = processOneData(ball5mmg.s21, emptygE.s21, trpeak2, trpeak2width, dAntennas2, pad, 'b5', makePlots = 0, sphereCalData=[ball35mmg.s21, 35e-3])
    # f1, sigball3975mmg = processOneData(ball3975mmg.s21, emptygF.s21, trpeak2, trpeak2width, dAntennas2, pad, 'b3.975', makePlots = 0, sphereCalData=[ball35mmg.s21, 35e-3])
    # f1, sigball025mmg = processOneData(ball025mmg.s21, emptygG.s21, trpeak2, trpeak2width, dAntennas2, pad, 'b2.5', makePlots = 0, sphereCalData=[ball35mmg.s21, 35e-3])
    #===============================================================================
    
    #===========================================================================
    # f1, sigShortUncloakedg= processOneData(shortUncloakedg.s21, emptygG.s21, trpeak2, trpeak2width, dAntennas2, pad, 'Uncloaked', makePlots = 0)
    # f1, sigUncloakedg= processOneData(uncloakedg.s21, emptygH.s21, trpeak2, trpeak2width, dAntennas2, pad, 'Uncloaked', makePlots = 0)
    # f1, sigLongCloakedg= processOneData(longCloakedg.s21, emptygI.s21, trpeak2, trpeak2width, dAntennas2, pad, 'Uncloaked', makePlots = 0)
    # f1, sigShortCloakedg= processOneData(shortCloakedg.s21, emptygJ.s21, trpeak2, trpeak2width, dAntennas2, pad, 'Uncloaked', makePlots = 0)
    # f1, sigShortUncloakedg2= processOneData(shortUncloakedg2.s21, emptygG2.s21, trpeak2, trpeak2width, dAntennas2, pad, 'Uncloaked', makePlots = 0)
    # f1, sigUncloakedg2= processOneData(uncloakedg2.s21, emptygH2.s21, trpeak2, trpeak2width, dAntennas2, pad, 'Uncloaked', makePlots = 0)
    # f1, sigLongCloakedg2= processOneData(longCloakedg2.s21, emptygI2.s21, trpeak2, trpeak2width, dAntennas2, pad, 'Uncloaked', makePlots = 0)
    # f1, sigShortCloakedg2= processOneData(shortCloakedg2.s21, emptygJ2.s21, trpeak2, trpeak2width, dAntennas2, pad, 'Uncloaked', makePlots = 0)
    # f1, sigShortUncloakedg3= processOneData(shortUncloakedg3.s21, emptygG3.s21, trpeak2, trpeak2width, dAntennas2, pad, 'Uncloaked', makePlots = 0)
    # f1, sigUncloakedg3= processOneData(uncloakedg3.s21, emptygH3.s21, trpeak2, trpeak2width, dAntennas2, pad, 'Uncloaked', makePlots = 0)
    # f1, sigLongCloakedg3= processOneData(longCloakedg3.s21, emptygI3.s21, trpeak2, trpeak2width, dAntennas2, pad, 'Uncloaked', makePlots = 0)
    # f1, sigShortCloakedg3= processOneData(shortCloakedg3.s21, emptygJ3.s21, trpeak2, trpeak2width, dAntennas2, pad, 'Uncloaked', makePlots = 0)
    #===========================================================================
    
    #===============================================================================
    # f1, sigShortUncloakedg= processOneData(shortUncloakedg.s21, emptygG.s21, trpeak2, trpeak2width, dAntennas2, pad, 'Uncloaked', makePlots = 0, sphereCalData=[ball635mmf.s21, 6.35e-3])
    # f1, sigUncloakedg= processOneData(uncloakedg.s21, emptygH.s21, trpeak2, trpeak2width, dAntennas2, pad, 'Uncloaked', makePlots = 0, sphereCalData=[ball635mmf.s21, 6.35e-3])
    # f1, sigLongCloakedg= processOneData(longCloakedg.s21, emptygI.s21, trpeak2, trpeak2width, dAntennas2, pad, 'Uncloaked', makePlots = 0, sphereCalData=[ball635mmf.s21, 6.35e-3])
    # f1, sigShortCloakedg= processOneData(shortCloakedg.s21, emptygJ.s21, trpeak2, trpeak2width, dAntennas2, pad, 'Uncloaked', makePlots = 0, sphereCalData=[ball635mmf.s21, 6.35e-3])
    # f1, sigShortUncloakedg2= processOneData(shortUncloakedg2.s21, emptygG2.s21, trpeak2, trpeak2width, dAntennas2, pad, 'Uncloaked', makePlots = 0, sphereCalData=[ball635mmf.s21, 6.35e-3])
    # f1, sigUncloakedg2= processOneData(uncloakedg2.s21, emptygH2.s21, trpeak2, trpeak2width, dAntennas2, pad, 'Uncloaked', makePlots = 0, sphereCalData=[ball635mmf.s21, 6.35e-3])
    # f1, sigLongCloakedg2= processOneData(longCloakedg2.s21, emptygI2.s21, trpeak2, trpeak2width, dAntennas2, pad, 'Uncloaked', makePlots = 0, sphereCalData=[ball635mmf.s21, 6.35e-3])
    # f1, sigShortCloakedg2= processOneData(shortCloakedg2.s21, emptygJ2.s21, trpeak2, trpeak2width, dAntennas2, pad, 'Uncloaked', makePlots = 0, sphereCalData=[ball635mmf.s21, 6.35e-3])
    # f1, sigShortUncloakedg3= processOneData(shortUncloakedg3.s21, emptygG3.s21, trpeak2, trpeak2width, dAntennas2, pad, 'Uncloaked', makePlots = 0, sphereCalData=[ball635mmf.s21, 6.35e-3])
    # f1, sigUncloakedg3= processOneData(uncloakedg3.s21, emptygH3.s21, trpeak2, trpeak2width, dAntennas2, pad, 'Uncloaked', makePlots = 0, sphereCalData=[ball635mmf.s21, 6.35e-3])
    # f1, sigLongCloakedg3= processOneData(longCloakedg3.s21, emptygI3.s21, trpeak2, trpeak2width, dAntennas2, pad, 'Uncloaked', makePlots = 0, sphereCalData=[ball635mmf.s21, 6.35e-3])
    # f1, sigShortCloakedg3= processOneData(shortCloakedg3.s21, emptygJ3.s21, trpeak2, trpeak2width, dAntennas2, pad, 'Uncloaked', makePlots = 0, sphereCalData=[ball635mmf.s21, 6.35e-3])
    #===============================================================================
    
    trpeakR = -36*1e-9 #s
    trpeakRwidth = 6*1e-9 #s
    pad = [20000, 'tukey']
    
    #===========================================================================
    # f1, sigball30mmR = processOneData(ball30mmR.s21, ball30mmRe.s21, trpeakR, trpeakRwidth, dAntennas2, pad, 'b30', makePlots = 2, sphereCalData=[ball15mmR.s21, 15e-3])
    # f1, sigball25mmR = processOneData(ball25mmR.s21, ball25mmRe.s21, trpeakR, trpeakRwidth, dAntennas2, pad, 'b25', makePlots = 0, sphereCalData=[ball15mmR.s21, 15e-3])
    # f1, sigball20mmR = processOneData(ball20mmR.s21, ball20mmRe.s21, trpeakR, trpeakRwidth, dAntennas2, pad, 'b20', makePlots = 0, sphereCalData=[ball15mmR.s21, 15e-3])
    # f1, sigball15mmR = processOneData(ball15mmR.s21, ball15mmRe.s21, trpeakR, trpeakRwidth, dAntennas2, pad, 'b15', makePlots = 0, sphereCalData=[ball15mmR.s21, 15e-3])
    # f1, sigball10mmR = processOneData(ball10mmR.s21, ball10mmRe.s21, trpeakR, trpeakRwidth, dAntennas2, pad, 'b10', makePlots = 0, sphereCalData=[ball15mmR.s21, 15e-3])
    # f1, sigball875mmR = processOneData(ball875mmR.s21, ball875mmRe.s21, trpeakR, trpeakRwidth, dAntennas2, pad, 'b8.75', makePlots = 0, sphereCalData=[ball15mmR.s21, 15e-3])
    # f1, sigball75mmR = processOneData(ball75mmR.s21, ball75mmRe.s21, trpeakR, trpeakRwidth, dAntennas2, pad, 'b7.5', makePlots = 0, sphereCalData=[ball15mmR.s21, 15e-3])
    # f1, sigball635mmR = processOneData(ball635mmR.s21, ball15mmRe.s21, trpeakR, trpeakRwidth, dAntennas2, pad, 'b15', makePlots = 0, sphereCalData=[ball15mmR.s21, 15e-3])
    # f1, sigball5mmR = processOneData(ball5mmR.s21, ball5mmRe.s21, trpeakR, trpeakRwidth, dAntennas2, pad, 'b5', makePlots = 0, sphereCalData=[ball15mmR.s21, 15e-3])
    # f1, sigball3975mmR = processOneData(ball3975mmR.s21, ball3975mmRe.s21, trpeakR, trpeakRwidth, dAntennas2, pad, 'b3.975', makePlots = 0, sphereCalData=[ball15mmR.s21, 15e-3])
    # f1, sigball025mmR = processOneData(ball025mmR.s21, ball025mmRe.s21, trpeakR, trpeakRwidth, dAntennas2, pad, 'b2.5', makePlots = 0, sphereCalData=[ball15mmR.s21, 15e-3])
    #===========================================================================
    
    
    ###try monostatic here##
    trpeak2Monostatic = 52*1e-9 ##s
    trpeak2widthMonostatic = 6*1e-9 ##s
    
    
    
    #===========================================================================
    # fF, monoball30mmf = processOneData(ball30mmf.s11, emptyfB.s11, trpeak2Monostatic, trpeak2widthMonostatic, dAntennas2, pad, 'b30', makePlots = 0, monostatic = True)
    # fF, monoball25mmf = processOneData(ball25mmf.s11, emptyfC.s11, trpeak2Monostatic, trpeak2widthMonostatic, dAntennas2, pad, 'b25', makePlots = 0, monostatic = True)
    # fF, monoball20mmf = processOneData(ball20mmf.s11, emptyfD.s11, trpeak2Monostatic, trpeak2widthMonostatic, dAntennas2, pad, 'b20', makePlots = 0, monostatic = True)
    # fF, monoball15mmf = processOneData(ball15mmf.s11, emptyfE.s11, trpeak2Monostatic, trpeak2widthMonostatic, dAntennas2, pad, 'b15', makePlots = 0, monostatic = True)
    # fF, monoball10mmf = processOneData(ball10mmf.s11, emptyfF.s11, trpeak2Monostatic, trpeak2widthMonostatic, dAntennas2, pad, 'b10', makePlots = 0, monostatic = True)
    # fF, monoball75mmf = processOneData(ball75mmf.s11, emptyfH.s11, trpeak2Monostatic, trpeak2widthMonostatic, dAntennas2, pad, 'b7.5', makePlots = 0, monostatic = True)
    # fF, monoball875mmf = processOneData(ball875mmf.s11, emptyfH.s11, trpeak2Monostatic, trpeak2widthMonostatic, dAntennas2, pad, 'b8.75', makePlots = 0, monostatic = True)
    # fF, monoball635mmf = processOneData(ball635mmf.s11, emptyfI.s11, trpeak2Monostatic, trpeak2widthMonostatic, dAntennas2, pad, 'b6.35', makePlots = 0, monostatic = True)
    # fF, monoball5mmf = processOneData(ball5mmf.s11, emptyfJ.s11, trpeak2Monostatic, trpeak2widthMonostatic, dAntennas2, pad, 'b5', makePlots = 0, monostatic = True)
    # fF, monoball3975mmf = processOneData(ball3975mmf.s11, emptyfK.s11, trpeak2Monostatic, trpeak2widthMonostatic, dAntennas2, pad, 'b3.975', makePlots = 0, monostatic = True)
    # fF, monoball025mmf = processOneData(ball025mmf.s11, emptyfL.s11, trpeak2Monostatic, trpeak2widthMonostatic, dAntennas2, pad, 'b2.5', makePlots = 0, monostatic = True)
    # fF, monoball015mmf = processOneData(ball015mmf.s11, emptyfH.s11, trpeak2Monostatic, trpeak2widthMonostatic, dAntennas2, pad, 'b1.5', makePlots = 0, monostatic = True)
    #===========================================================================
    
    
    
    #===========================================================================
    # fF, monoball15mmg = processOneData(ball15mmg.s11, emptygC.s11, trpeak2Monostatic, trpeak2widthMonostatic, dAntennas2, pad, 'b15', makePlots = 0, monostatic = True)
    # 
    # fG, monoShortUncloakedg= processOneData(shortUncloakedg.s11, emptygG.s11, trpeak2Monostatic, trpeak2widthMonostatic, dAntennas2, pad, 'sUncloaked', makePlots = 0, monostatic = True)
    # fG, monoUncloakedg= processOneData(uncloakedg.s11, emptygH.s11, trpeak2Monostatic, trpeak2widthMonostatic, dAntennas2, pad, 'Uncloaked', makePlots = 0, monostatic = True)
    # fG, monoLongCloakedg= processOneData(longCloakedg.s11, emptygI.s11, trpeak2Monostatic, trpeak2widthMonostatic, dAntennas2, pad, 'lcloaked', makePlots = 0, monostatic = True)
    # fG, monoShortCloakedg= processOneData(shortCloakedg.s11, emptygJ.s11, trpeak2Monostatic, trpeak2widthMonostatic, dAntennas2, pad, 'scloaked', makePlots = 0, monostatic = True)
    # 
    # fG, monoShortUncloakedg2= processOneData(shortUncloakedg2.s11, emptygG2.s11, trpeak2Monostatic, trpeak2widthMonostatic, dAntennas2, pad, 'sUncloaked', makePlots = 0, monostatic = True)
    # fG, monoUncloakedg2= processOneData(uncloakedg2.s11, emptygH2.s11, trpeak2Monostatic, trpeak2widthMonostatic, dAntennas2, pad, 'Uncloaked', makePlots = 0, monostatic = True)
    # fG, monoLongCloakedg2= processOneData(longCloakedg2.s11, emptygI2.s11, trpeak2Monostatic, trpeak2widthMonostatic, dAntennas2, pad, 'lcloaked', makePlots = 0, monostatic = True)
    # fG, monoShortCloakedg2= processOneData(shortCloakedg2.s11, emptygJ2.s11, trpeak2Monostatic, trpeak2widthMonostatic, dAntennas2, pad, 'scloaked', makePlots = 0, monostatic = True)
    # 
    # fG, monoShortUncloakedg3= processOneData(shortUncloakedg3.s11, emptygG3.s11, trpeak2Monostatic, trpeak2widthMonostatic, dAntennas2, pad, 'sUncloaked', makePlots = 0, monostatic = True)
    # fG, monoUncloakedg3= processOneData(uncloakedg3.s11, emptygH3.s11, trpeak2Monostatic, trpeak2widthMonostatic, dAntennas2, pad, 'Uncloaked', makePlots = 0, monostatic = True)
    # fG, monoLongCloakedg3= processOneData(longCloakedg3.s11, emptygI3.s11, trpeak2Monostatic, trpeak2widthMonostatic, dAntennas2, pad, 'lcloaked', makePlots = 0, monostatic = True)
    # fG, monoShortCloakedg3= processOneData(shortCloakedg3.s11, emptygJ3.s11, trpeak2Monostatic, trpeak2widthMonostatic, dAntennas2, pad, 'scloaked', makePlots = 0, monostatic = True)
    #===========================================================================

    
    
    trpeakRMonostatic = 52e-9#-55*1e-9 ##s
    trpeakRwidthMonostatic = 6*1e-9 ##s
    
    #===========================================================================
    # f1, monoball30mmR = processOneData(ball30mmR.s11, ball30mmRe.s11, trpeakRMonostatic, trpeakRwidthMonostatic, dAntennas2, pad, 'b30', makePlots = 0, monostatic = True)
    # f1, monoball25mmR = processOneData(ball25mmR.s11, ball25mmRe.s11, trpeakRMonostatic, trpeakRwidthMonostatic, dAntennas2, pad, 'b25', makePlots = 0, monostatic = True)
    # f1, monoball20mmR = processOneData(ball20mmR.s11, ball20mmRe.s11, trpeakRMonostatic, trpeakRwidthMonostatic, dAntennas2, pad, 'b20', makePlots = 0, monostatic = True)
    # f1, monoball15mmR = processOneData(ball15mmR.s11, ball15mmRe.s11, trpeakRMonostatic, trpeakRwidthMonostatic, dAntennas2, pad, 'b15', makePlots = 0, monostatic = True)
    # f1, monoball10mmR = processOneData(ball10mmR.s11, ball10mmRe.s11, trpeakRMonostatic, trpeakRwidthMonostatic, dAntennas2, pad, 'b10', makePlots = 0, monostatic = True)
    # f1, monoball75mmR = processOneData(ball75mmR.s11, ball75mmRe.s11, trpeakRMonostatic, trpeakRwidthMonostatic, dAntennas2, pad, 'b7.5', makePlots = 0, monostatic = True)
    # f1, monoball875mmR = processOneData(ball875mmR.s11, ball875mmRe.s11, trpeakRMonostatic, trpeakRwidthMonostatic, dAntennas2, pad, 'b8.75', makePlots = 0, monostatic = True)
    # f1, monoball635mmR = processOneData(ball635mmR.s11, ball635mmRe.s11, trpeakRMonostatic, trpeakRwidthMonostatic, dAntennas2, pad, 'b6.35', makePlots = 0, monostatic = True)
    # f1, monoball5mmR = processOneData(ball5mmR.s11, ball5mmRe.s11, trpeakRMonostatic, trpeakRwidthMonostatic, dAntennas2, pad, 'b5', makePlots = 0, monostatic = True)
    # f1, monoball3975mmR = processOneData(ball3975mmR.s11, ball3975mmRe.s11, trpeakRMonostatic, trpeakRwidthMonostatic, dAntennas2, pad, 'b3.975', makePlots = 0, monostatic = True)
    # f1, monoball025mmR = processOneData(ball025mmR.s11, ball025mmRe.s11, trpeakRMonostatic, trpeakRwidthMonostatic, dAntennas2, pad, 'b2.5', makePlots = 0, monostatic = True)
    #===========================================================================
    
    dAntennas2 = 3.355 ##approx, 3.355 front-to-front, 4.043 back-to-back measured with a laser
    trpeakE = -36*1e-9 #s
    trpeakEwidth = 9*1e-9 #s
    pad = [20000, 'tukey']
    
    #===========================================================================
    # f1, sigball30mmSetE = processOneData(ball30mmSetE.s21, ball30mmeSetE.s21, trpeakE, trpeakEwidth, dAntennas2, pad, 'b30', makePlots = 0, sphereCalData=[ball10mmSetE.s21, 10e-3])
    # f1, sigball10mmSetE = processOneData(ball10mmSetE.s21, ball10mmeSetE.s21, trpeakE, trpeakEwidth, dAntennas2, pad, 'b10', makePlots = 0, sphereCalData=[ball10mmSetE.s21, 10e-3])
    # f1, sigball5mmSetE = processOneData(ball5mmSetE.s21, ball5mmeSetE.s21, trpeakE, trpeakEwidth, dAntennas2, pad, 'b5', makePlots = 0, sphereCalData=[ball10mmSetE.s21, 10e-3])
    # f1, sigball025mmSetE = processOneData(ball025mmSetE.s21, ball025mmeSetE.s21, trpeakE, trpeakEwidth, dAntennas2, pad, 'b2.5', makePlots = 0, sphereCalData=[ball10mmSetE.s21, 10e-3])
    # f1, sigball015mmSetE = processOneData(ball015mmSetE.s21, ball015mmeSetE.s21, trpeakE, trpeakEwidth, dAntennas2, pad, 'b1.5', makePlots = 0, sphereCalData=[ball10mmSetE.s21, 10e-3])
    #===========================================================================
    
    #===========================================================================
    # f1, sigball30mmSetF = processOneData(ball30mmSetF.s21, ball30mmeSetF.s21, trpeakE, trpeakEwidth, dAntennas2, pad, 'b30', makePlots = 0, sphereCalData=[ball30mmSetF.s21, 30e-3])
    # f1, sigball10mmSetF = processOneData(ball10mmSetF.s21, ball10mmeSetF.s21, trpeakE, trpeakEwidth, dAntennas2, pad, 'b10', makePlots = 0, sphereCalData=[ball30mmSetF.s21, 30e-3])
    # f1, sigball5mmSetF = processOneData(ball5mmSetF.s21, ball5mmeSetF.s21, trpeakE, trpeakEwidth, dAntennas2, pad, 'b5', makePlots = 0, sphereCalData=[ball30mmSetF.s21, 30e-3])
    # f1, sigball3975mmSetF = processOneData(ball3975mmSetF.s21, ball3975mmeSetF.s21, trpeakE, trpeakEwidth, dAntennas2, pad, 'b3975', makePlots = 0, sphereCalData=[ball30mmSetF.s21, 30e-3])
    # f1, sigball025mmSetF = processOneData(ball025mmSetF.s21, ball025mmeSetF.s21, trpeakE, trpeakEwidth, dAntennas2, pad, 'b2.5', makePlots = 0, sphereCalData=[ball30mmSetF.s21, 30e-3])
    # f1, sigball015mmSetF = processOneData(ball015mmSetF.s21, ball015mmeSetF.s21, trpeakE, trpeakEwidth, dAntennas2, pad, 'b1.5', makePlots = 0, sphereCalData=[ball30mmSetF.s21, 30e-3])
    #===========================================================================
    
    #===========================================================================
    # f1, sigball30mmSetG = processOneData(ball30mmSetG.s21, ball30mmeSetG.s21, trpeakE, trpeakEwidth, dAntennas2, pad, 'b30', makePlots = 0, sphereCalData=[ball5mmSetG.s21, 5e-3])
    # f1, sigball10mmSetG = processOneData(ball10mmSetG.s21, ball10mmeSetG.s21, trpeakE, trpeakEwidth, dAntennas2, pad, 'b10', makePlots = 0, sphereCalData=[ball5mmSetG.s21, 5e-3])
    # f1, sigball5mmSetG = processOneData(ball5mmSetG.s21, ball5mmeSetG.s21, trpeakE, trpeakEwidth, dAntennas2, pad, 'b5', makePlots = 0, sphereCalData=[ball5mmSetG.s21, 5e-3])
    # f1, sigball3975mmSetG = processOneData(ball3975mmSetG.s21, ball3975mmeSetG.s21, trpeakE, trpeakEwidth, dAntennas2, pad, 'b3975', makePlots = 0, sphereCalData=[ball5mmSetG.s21, 5e-3])
    # f1, sigball025mmSetG = processOneData(ball025mmSetG.s21, ball025mmeSetG.s21, trpeakE, trpeakEwidth, dAntennas2, pad, 'b2.5', makePlots = 0, sphereCalData=[ball5mmSetG.s21, 5e-3])
    # f1, sigball015mmSetG = processOneData(ball015mmSetG.s21, ball015mmeSetG.s21, trpeakE, trpeakEwidth, dAntennas2, pad, 'b1.5', makePlots = 0, sphereCalData=[ball5mmSetG.s21, 5e-3])
    #===========================================================================
    
    
    trpeakH = 83*1e-9 #s
    trpeakHwidth = 9*1e-9 #s
    
    #===========================================================================
    # f1, sigball30mmSetH = processOneData(ball30mmSetH.s21, ball30mmeSetH.s21, trpeakH, trpeakHwidth, dAntennas2, pad, 'b30', makePlots = 0, sphereCalData=[ball5mmSetH.s21, 5e-3])
    # f1, sigball10mmSetH = processOneData(ball10mmSetH.s21, ball10mmeSetH.s21, trpeakH, trpeakHwidth, dAntennas2, pad, 'b10', makePlots = 0, sphereCalData=[ball5mmSetH.s21, 5e-3])
    # f1, sigball5mmSetH = processOneData(ball5mmSetH.s21, ball5mmeSetH.s21, trpeakH, trpeakHwidth, dAntennas2, pad, 'b5', makePlots = 0, sphereCalData=[ball5mmSetH.s21, 5e-3])
    # f1, sigball3975mmSetH = processOneData(ball3975mmSetH.s21, ball3975mmeSetH.s21, trpeakH, trpeakHwidth, dAntennas2, pad, 'b3975', makePlots = 0, sphereCalData=[ball5mmSetH.s21, 5e-3])
    # f1, sigball025mmSetH = processOneData(ball025mmSetH.s21, ball025mmeSetH.s21, trpeakH, trpeakHwidth, dAntennas2, pad, 'b2.5', makePlots = 0, sphereCalData=[ball5mmSetH.s21, 5e-3])
    # f1, sigball015mmSetH = processOneData(ball015mmSetH.s21, ball015mmeSetH.s21, trpeakH, trpeakHwidth, dAntennas2, pad, 'b1.5', makePlots = 0, sphereCalData=[ball5mmSetH.s21, 5e-3])
    #===========================================================================
    
    trpeakI = -36*1e-9 #s
    trpeakIwidth = 9*1e-9 #s
    
    #===========================================================================
    # f1, sigball30mmSetI = processOneData(ball30mmSetI.s21, ball30mmeSetI.s21, trpeakI, trpeakIwidth, dAntennas2, pad, 'b30', makePlots = 0, sphereCalData=[ball10mmSetI.s21, 10e-3])
    # f1, sigball10mmSetI = processOneData(ball10mmSetI.s21, ball10mmeSetI.s21, trpeakI, trpeakIwidth, dAntennas2, pad, 'b10', makePlots = 0, sphereCalData=[ball10mmSetI.s21, 10e-3])
    # f1, sigball5mmSetI = processOneData(ball5mmSetI.s21, ball5mmeSetI.s21, trpeakI, trpeakIwidth, dAntennas2, pad, 'b5', makePlots = 0, sphereCalData=[ball10mmSetI.s21, 10e-3])
    # f1, sigball3975mmSetI = processOneData(ball3975mmSetI.s21, ball3975mmeSetI.s21, trpeakI, trpeakIwidth, dAntennas2, pad, 'b3975', makePlots = 0, sphereCalData=[ball10mmSetI.s21, 10e-3])
    # f1, sigball025mmSetI = processOneData(ball025mmSetI.s21, ball025mmeSetI.s21, trpeakI, trpeakIwidth, dAntennas2, pad, 'b2.5', makePlots = 0, sphereCalData=[ball10mmSetI.s21, 10e-3])
    #===========================================================================
    
    #===========================================================================
    # fSetJ, sigball30mmSetJ = processOneData(ball30mmSetJ.s21, ball30mmeSetJ.s21, trpeakI, trpeakIwidth, dAntennas2, pad, 'b30', makePlots = 0, sphereCalData=[ball5mmSetJ.s21, 5e-3])
    # fSetJ, sigball10mmSetJ = processOneData(ball10mmSetJ.s21, ball10mmeSetJ.s21, trpeakI, trpeakIwidth, dAntennas2, pad, 'b10', makePlots = 0, sphereCalData=[ball5mmSetJ.s21, 5e-3])
    # fSetJ, sigball5mmSetJ = processOneData(ball5mmSetJ.s21, ball5mmeSetJ.s21, trpeakI, trpeakIwidth, dAntennas2, pad, 'b5', makePlots = 0, sphereCalData=[ball5mmSetJ.s21, 5e-3])
    # fSetJ, sigball3975mmSetJ = processOneData(ball3975mmSetJ.s21, ball3975mmeSetJ.s21, trpeakI, trpeakIwidth, dAntennas2, pad, 'b3975', makePlots = 0, sphereCalData=[ball5mmSetJ.s21, 5e-3])
    # fSetJ, sigball025mmSetJ = processOneData(ball025mmSetJ.s21, ball025mmeSetJ.s21, trpeakI, trpeakIwidth, dAntennas2, pad, 'b2.5', makePlots = 0, sphereCalData=[ball5mmSetJ.s21, 5e-3])
    #===========================================================================
    
    trpeakSetKMonostatic = 52*1e-9 ##s
    trpeakSetKMonostaticwidth = 6*1e-9 ##s
    
    ###use interpolation to compensate for phase/magnitude changes
    
    if(driftCompensating):
        interpPhiS11, interpMS11, interpPhiS21, interpMS21, tbM, tbFM = plotBackgroundSubtractionStats(listOfBackgrounds, tgS21=[pad, trpeakI, trpeakIwidth], tgS11=[pad, trpeakSetKMonostatic, trpeakSetKMonostaticwidth], timePassed = backgroundsTimePassed)
        infoToDoDriftInterpolation = [interpPhiS11, interpMS11, interpPhiS21, interpMS21, tbM, tbFM, listOfBackgrounds, listOfNonBackgrounds] ##S11 interpolations (phi then M), S21 interpolations (phi then M), time between measurements, time between frequency points, list of BGs, list of non BGs
    else: 
        infoToDoDriftInterpolation = None
    
    
    sphereCalK=[ball15mmSetK.s21, 15e-3, ball15mmeSetK.s21]
    fSetK, sigball30mmSetK = processOneData(ball30mmSetK.s21, ball30mmeSetK.s21, trpeakI, trpeakIwidth, dAntennas2, pad, 'b30', makePlots = 0, sphereCalData=sphereCalK, interpData = infoToDoDriftInterpolation)
    fSetK, sigball25mmSetK = processOneData(ball25mmSetK.s21, ball25mmeSetK.s21, trpeakI, trpeakIwidth, dAntennas2, pad, 'b25', makePlots = 0, sphereCalData=sphereCalK, interpData = infoToDoDriftInterpolation)
    fSetK, sigball20mmSetK = processOneData(ball20mmSetK.s21, ball20mmeSetK.s21, trpeakI, trpeakIwidth, dAntennas2, pad, 'b20', makePlots = 0, sphereCalData=sphereCalK, interpData = infoToDoDriftInterpolation)
    fSetK, sigball15mmSetK = processOneData(ball15mmSetK.s21, ball15mmeSetK.s21, trpeakI, trpeakIwidth, dAntennas2, pad, 'b15', makePlots = 0, sphereCalData=sphereCalK, interpData = infoToDoDriftInterpolation)
    fSetK, sigball10mmSetK = processOneData(ball10mmSetK.s21, ball10mmeSetK.s21, trpeakI, trpeakIwidth, dAntennas2, pad, 'b10', makePlots = 0, sphereCalData=sphereCalK, interpData = infoToDoDriftInterpolation)
    fSetK, sigball875mmSetK = processOneData(ball875mmSetK.s21, ball30mmeSetK.s21, trpeakI, trpeakIwidth, dAntennas2, pad, 'b8.75', makePlots = 0, sphereCalData=sphereCalK, interpData = infoToDoDriftInterpolation)
    fSetK, sigball635mmSetK = processOneData(ball635mmSetK.s21, ball30mmeSetK.s21, trpeakI, trpeakIwidth, dAntennas2, pad, 'b6.35', makePlots = 0, sphereCalData=sphereCalK, interpData = infoToDoDriftInterpolation)
    fSetK, sigball5mmSetK = processOneData(ball5mmSetK.s21, ball5mmeSetK.s21, trpeakI, trpeakIwidth, dAntennas2, pad, 'b5', makePlots = 0, sphereCalData=sphereCalK, interpData = infoToDoDriftInterpolation)
    fSetK, sigball3975mmSetK = processOneData(ball3975mmSetK.s21, ball3975mmeSetK.s21, trpeakI, trpeakIwidth, dAntennas2, pad, 'b3975', makePlots = 0, sphereCalData=sphereCalK, interpData = infoToDoDriftInterpolation)
    fSetK, sigball025mmSetK = processOneData(ball025mmSetK.s21, ball025mmeSetK.s21, trpeakI, trpeakIwidth, dAntennas2, pad, 'b2.5', makePlots = 0, sphereCalData=sphereCalK, interpData = infoToDoDriftInterpolation)
    fSetK, sigball015mmSetK = processOneData(ball015mmSetK.s21, ball015mmeSetK.s21, trpeakI, trpeakIwidth, dAntennas2, pad, 'b1.5', makePlots = 0, sphereCalData=sphereCalK, interpData = infoToDoDriftInterpolation)
    
    fSetK2, sigball50mmSetK2 = processOneData(ball50mmSetK2.s21, ball50mmeSetK2.s21, trpeakI, trpeakIwidth, dAntennas2, pad, 'b50', makePlots = 0, sphereCalData=sphereCalK, interpData = infoToDoDriftInterpolation)
    fSetK2, sigball5mmSetK2 = processOneData(ball5mmSetK2.s21, ball5mmeSetK2.s21, trpeakI, trpeakIwidth, dAntennas2, pad, 'b5', makePlots = 0, sphereCalData=sphereCalK, interpData = infoToDoDriftInterpolation)
    fSetK2, sigball3975mmSetK2 = processOneData(ball3975mmSetK2.s21, ball3975mmeSetK2.s21, trpeakI, trpeakIwidth, dAntennas2, pad, 'b3975', makePlots = 0, sphereCalData=sphereCalK, interpData = infoToDoDriftInterpolation)
    fSetK2, sigball025mmSetK2 = processOneData(ball025mmSetK2.s21, ball025mmeSetK2.s21, trpeakI, trpeakIwidth, dAntennas2, pad, 'b2.5', makePlots = 0, sphereCalData=sphereCalK, interpData = infoToDoDriftInterpolation)
    fSetK2, sigball015mmSetK2 = processOneData(ball015mmSetK2.s21, ball015mmeSetK2.s21, trpeakI, trpeakIwidth, dAntennas2, pad, 'b1.5', makePlots = 0, sphereCalData=sphereCalK, interpData = infoToDoDriftInterpolation)
    
    
    sphereCalKuncloak=[ball875mmSetK.s21, 8.75e-3, ball875mmeSetK.s21]
    sphereCalKsuncloak=[ball875mmSetK.s21, 8.75e-3, ball875mmeSetK.s21]
    sphereCalKlocloak=[ball5mmSetK.s21, 5e-3, ball5mmeSetK.s21]
    sphereCalKshcloak=[ball5mmSetK.s21, 5e-3, ball5mmeSetK.s21]
    
    #===========================================================================
    # sphereCalKuncloak=[ball875mmSetK.s21, 8.75e-3]
    # sphereCalKsuncloak=[ball875mmSetK.s21, 8.75e-3]
    # sphereCalKlocloak=[ball5mmSetK.s21, 5e-3]
    # sphereCalKshcloak=[ball5mmSetK.s21, 5e-3]
    #===========================================================================
    
    fSetK, sigShortUncloakedSetK= processOneData(shortUncloakedSetK.s21, shortUncloakedeSetK.s21, trpeakI, trpeakIwidth, dAntennas2, pad, 'sUncloaked', makePlots = 0, sphereCalData=sphereCalKsuncloak, interpData = infoToDoDriftInterpolation)
    fSetK, sigUncloakedSetK= processOneData(uncloakedSetK.s21, uncloakedeSetK.s21, trpeakI, trpeakIwidth, dAntennas2, pad, 'Uncloaked', makePlots = 0, sphereCalData=sphereCalKuncloak, interpData = infoToDoDriftInterpolation)
    fSetK, sigLongCloakedSetK= processOneData(longCloakedSetK.s21, longCloakedeSetK.s21, trpeakI, trpeakIwidth, dAntennas2, pad, 'lcloaked', makePlots = 0, sphereCalData=sphereCalKlocloak, interpData = infoToDoDriftInterpolation)
    fSetK, sigShortCloakedSetK= processOneData(shortCloakedSetK.s21, shortCloakedeSetK.s21, trpeakI, trpeakIwidth, dAntennas2, pad, 'scloaked', makePlots = 0, sphereCalData=sphereCalKshcloak, interpData = infoToDoDriftInterpolation)
    
    fSetK, sigShortUncloakedSetK2= processOneData(shortUncloakedSetK2.s21, shortUncloakedeSetK2.s21, trpeakI, trpeakIwidth, dAntennas2, pad, 'sUncloaked', makePlots = 0, sphereCalData=sphereCalKsuncloak, interpData = infoToDoDriftInterpolation)
    fSetK, sigUncloakedSetK2= processOneData(uncloakedSetK2.s21, uncloakedeSetK2.s21, trpeakI, trpeakIwidth, dAntennas2, pad, 'Uncloaked', makePlots = 0, sphereCalData=sphereCalKuncloak, interpData = infoToDoDriftInterpolation)
    fSetK, sigLongCloakedSetK2= processOneData(longCloakedSetK2.s21, longCloakedeSetK2.s21, trpeakI, trpeakIwidth, dAntennas2, pad, 'lcloaked', makePlots = 0, sphereCalData=sphereCalKlocloak, interpData = infoToDoDriftInterpolation)
    fSetK, sigShortCloakedSetK2= processOneData(shortCloakedSetK2.s21, shortCloakedeSetK2.s21, trpeakI, trpeakIwidth, dAntennas2, pad, 'scloaked', makePlots = 0, sphereCalData=sphereCalKshcloak, interpData = infoToDoDriftInterpolation)
    
    fSetK, sigShortUncloakedSetK3= processOneData(shortUncloakedSetK3.s21, shortUncloakedeSetK3.s21, trpeakI, trpeakIwidth, dAntennas2, pad, 'sUncloaked', makePlots = 0, sphereCalData=sphereCalKsuncloak, interpData = infoToDoDriftInterpolation)
    fSetK, sigUncloakedSetK3= processOneData(uncloakedSetK3.s21, uncloakedeSetK3.s21, trpeakI, trpeakIwidth, dAntennas2, pad, 'Uncloaked', makePlots = 0, sphereCalData=sphereCalKuncloak, interpData = infoToDoDriftInterpolation)
    fSetK, sigLongCloakedSetK3= processOneData(longCloakedSetK3.s21, longCloakedeSetK3.s21, trpeakI, trpeakIwidth, dAntennas2, pad, 'lcloaked', makePlots = 0, sphereCalData=sphereCalKlocloak, interpData = infoToDoDriftInterpolation)
    fSetK, sigShortCloakedSetK3= processOneData(shortCloakedSetK3.s21, shortCloakedeSetK3.s21, trpeakI, trpeakIwidth, dAntennas2, pad, 'scloaked', makePlots = 0, sphereCalData=sphereCalKshcloak, interpData = infoToDoDriftInterpolation)
    
    
    
    fSetK, monoball30mmSetK = processOneData(ball30mmSetK.s11, ball30mmeSetK.s11, trpeakSetKMonostatic, trpeakSetKMonostaticwidth, dAntennas2, pad, 'b30', makePlots = 0, monostatic = True, interpData = infoToDoDriftInterpolation)
    fSetK, monoball25mmSetK = processOneData(ball25mmSetK.s11, ball25mmeSetK.s11, trpeakSetKMonostatic, trpeakSetKMonostaticwidth, dAntennas2, pad, 'b25', makePlots = 0, monostatic = True, interpData = infoToDoDriftInterpolation)
    fSetK, monoball20mmSetK = processOneData(ball20mmSetK.s11, ball20mmeSetK.s11, trpeakSetKMonostatic, trpeakSetKMonostaticwidth, dAntennas2, pad, 'b20', makePlots = 0, monostatic = True, interpData = infoToDoDriftInterpolation)
    fSetK, monoball15mmSetK = processOneData(ball15mmSetK.s11, ball15mmeSetK.s11, trpeakSetKMonostatic, trpeakSetKMonostaticwidth, dAntennas2, pad, 'b15', makePlots = 0, monostatic = True, interpData = infoToDoDriftInterpolation)
    fSetK, monoball10mmSetK = processOneData(ball10mmSetK.s11, ball10mmeSetK.s11, trpeakSetKMonostatic, trpeakSetKMonostaticwidth, dAntennas2, pad, 'b10', makePlots = 0, monostatic = True, interpData = infoToDoDriftInterpolation)
    fSetK, monoball875mmSetK = processOneData(ball875mmSetK.s11, ball30mmeSetK.s11, trpeakSetKMonostatic, trpeakSetKMonostaticwidth, dAntennas2, pad, 'b8.75', makePlots = 0, monostatic = True, interpData = infoToDoDriftInterpolation)
    fSetK, monoball635mmSetK = processOneData(ball635mmSetK.s11, ball30mmeSetK.s11, trpeakSetKMonostatic, trpeakSetKMonostaticwidth, dAntennas2, pad, 'b6.35', makePlots = 0, monostatic = True, interpData = infoToDoDriftInterpolation)
    fSetK, monoball5mmSetK = processOneData(ball5mmSetK.s11, ball5mmeSetK.s11, trpeakSetKMonostatic, trpeakSetKMonostaticwidth, dAntennas2, pad, 'b5', makePlots = 0, monostatic = True, interpData = infoToDoDriftInterpolation)
    fSetK, monoball3975mmSetK = processOneData(ball3975mmSetK.s11, ball3975mmeSetK.s11, trpeakSetKMonostatic, trpeakSetKMonostaticwidth, dAntennas2, pad, 'b3975', makePlots = 0, monostatic = True, interpData = infoToDoDriftInterpolation)
    fSetK, monoball025mmSetK = processOneData(ball025mmSetK.s11, ball025mmeSetK.s11, trpeakSetKMonostatic, trpeakSetKMonostaticwidth, dAntennas2, pad, 'b2.5', makePlots = 0, monostatic = True, interpData = infoToDoDriftInterpolation)
    fSetK, monoball015mmSetK = processOneData(ball015mmSetK.s11, ball015mmeSetK.s11, trpeakSetKMonostatic, trpeakSetKMonostaticwidth, dAntennas2, pad, 'b1.5', makePlots = 0, monostatic = True, interpData = infoToDoDriftInterpolation)
    
    fSetK, monoShortUncloakedSetK= processOneData(shortUncloakedSetK.s11, shortUncloakedeSetK.s11, trpeakSetKMonostatic, trpeakSetKMonostaticwidth, dAntennas2, pad, 'sUncloaked', makePlots = 0, monostatic = True, interpData = infoToDoDriftInterpolation)
    fSetK, monoUncloakedSetK= processOneData(uncloakedSetK.s11, uncloakedeSetK.s11, trpeakSetKMonostatic, trpeakSetKMonostaticwidth, dAntennas2, pad, 'Uncloaked', makePlots = 0, monostatic = True, interpData = infoToDoDriftInterpolation)
    fSetK, monoLongCloakedSetK= processOneData(longCloakedSetK.s11, longCloakedeSetK.s11, trpeakSetKMonostatic, trpeakSetKMonostaticwidth, dAntennas2, pad, 'lcloaked', makePlots = 0, monostatic = True, interpData = infoToDoDriftInterpolation)
    fSetK, monoShortCloakedSetK= processOneData(shortCloakedSetK.s11, shortCloakedeSetK.s11, trpeakSetKMonostatic, trpeakSetKMonostaticwidth, dAntennas2, pad, 'scloaked', makePlots = 0, monostatic = True, interpData = infoToDoDriftInterpolation)
    
    
    fSetK, monoShortUncloakedSetK2= processOneData(shortUncloakedSetK2.s11, shortUncloakedeSetK2.s11, trpeakSetKMonostatic, trpeakSetKMonostaticwidth, dAntennas2, pad, 'sUncloaked', makePlots = 0, monostatic = True, interpData = infoToDoDriftInterpolation)
    fSetK, monoUncloakedSetK2= processOneData(uncloakedSetK2.s11, uncloakedeSetK2.s11, trpeakSetKMonostatic, trpeakSetKMonostaticwidth, dAntennas2, pad, 'Uncloaked', makePlots = 0, monostatic = True, interpData = infoToDoDriftInterpolation)
    fSetK, monoLongCloakedSetK2= processOneData(longCloakedSetK2.s11, longCloakedeSetK2.s11, trpeakSetKMonostatic, trpeakSetKMonostaticwidth, dAntennas2, pad, 'lcloaked', makePlots = 0, monostatic = True, interpData = infoToDoDriftInterpolation)
    fSetK, monoShortCloakedSetK2= processOneData(shortCloakedSetK2.s11, shortCloakedeSetK2.s11, trpeakSetKMonostatic, trpeakSetKMonostaticwidth, dAntennas2, pad, 'scloaked', makePlots = 0, monostatic = True, interpData = infoToDoDriftInterpolation)
    
    fSetK, monoShortUncloakedSetK3= processOneData(shortUncloakedSetK3.s11, shortUncloakedeSetK3.s11, trpeakSetKMonostatic, trpeakSetKMonostaticwidth, dAntennas2, pad, 'sUncloaked', makePlots = 0, monostatic = True, interpData = infoToDoDriftInterpolation)
    fSetK, monoUncloakedSetK3= processOneData(uncloakedSetK3.s11, uncloakedeSetK3.s11, trpeakSetKMonostatic, trpeakSetKMonostaticwidth, dAntennas2, pad, 'Uncloaked', makePlots = 0, monostatic = True, interpData = infoToDoDriftInterpolation)
    fSetK, monoLongCloakedSetK3= processOneData(longCloakedSetK3.s11, longCloakedeSetK3.s11, trpeakSetKMonostatic, trpeakSetKMonostaticwidth, dAntennas2, pad, 'lcloaked', makePlots = 0, monostatic = True, interpData = infoToDoDriftInterpolation)
    fSetK, monoShortCloakedSetK3= processOneData(shortCloakedSetK3.s11, shortCloakedeSetK3.s11, trpeakSetKMonostatic, trpeakSetKMonostaticwidth, dAntennas2, pad, 'scloaked', makePlots = 0, monostatic = True, interpData = infoToDoDriftInterpolation)

    #plotting background subtractions
    
    pad = [20000, 'hamming']
    #plotBackgroundSubtraction([emptyfA,emptyfA2,emptyfA3], names=['Reference: EmptyfA','EmptyfA2','EmptyfA3'], tg=[pad, trpeak2, trpeak2width])
    #plotBackgroundSubtraction([emptygA,ball30mmf,ball25mmf,ball20mmf,ball15mmf,ball10mmf,ball875mmf,ball75mmf,ball635mmf,ball5mmf,ball3975mmf,ball025mmf,ball015mmf], names=['Reference: EmptygA','ball30mmf','ball25mmf','ball20mmf','ball15mmf','ball10mmf','ball875mmf','ball75mmf','ball635mmf','ball5mmf','ball3975mmf','ball025mmf','ball015mmf',], tg=[pad, trpeak2, trpeak2width])
    #plotBackgroundSubtraction([emptyeW3,emptyeX3,emptyeY3], names=['Reference: EmptyeW3','EmptyeX3','EmptyeY3'], tg=[pad, trpeakballs, trpeakballswidth])
    
    #plotBackgroundSubtraction([emptygA,ball30mmf,ball25mmf,ball20mmf,ball15mmf,ball10mmf,ball875mmf,ball75mmf,ball635mmf,ball5mmf,ball3975mmf,ball025mmf,ball015mmf], names=['Reference: EmptygA','ball30mmf','ball25mmf','ball20mmf','ball15mmf','ball10mmf','ball875mmf','ball75mmf','ball635mmf','ball5mmf','ball3975mmf','ball025mmf','ball015mmf',], tg=None)
    
    #plotBackgroundSubtraction([emptyfA,emptyfA2,emptyfA3], names=['Reference: EmptyfA','EmptyfA2','EmptyfA3'], tg=None)
    #plotBackgroundSubtraction([emptyeW3,emptyeX3,emptyeY3], names=['Reference: EmptyeW3','EmptyeX3','EmptyeY3'], tg=None)
    #plotBackgroundSubtraction([emptyeE,emptyeF,emptyeG,emptyeH,emptyeI], names=['Reference: EmptyeE','EmptyeF','EmptyeG','EmptyeH','EmptyeI'], tg=None)
    #plotBackgroundSubtraction([emptyeQ3,emptyeR3,emptyeS3,emptyeT3,emptyeU3,emptyeV3,emptyeW3,emptyeX3,emptyeY3], names=['Reference: EmptyeQ3', 'EmptyeR3', 'EmptyeS3', 'EmptyeT3', 'EmptyeU3', 'EmptyeV3', 'EmptyeW3', 'EmptyeX3', 'EmptyeY3'], tg=None)
    
    #plotBackgroundSubtraction([ball30mmeSetK,ball25mmeSetK,ball20mmeSetK,ball015mmeSetK2], names=['1','2','3','4'], tg=[pad, trpeakI, trpeakIwidth])
    
    
    ##diff timing for just 8-12 GHz data:
    trpeakI = -37.7e-9
    
    #===========================================================================
    # trpeakSetKMonostaticwidth = 1
    # trpeakIwidth = 1
    #===========================================================================
    
    
    linewidth = 1.2
    font = {'size'   : 20}
    plt.rc('font', **font)
    labelFontSize = 23
    titleFontSize = 22
    legendFontSize = 16
    
    if(not toPlotComparisons):
        figsizex = 10.5/1.2
        figsizey = 6.5/1.2
        fig = plt.figure( figsize=(figsizex, figsizey), dpi=80, facecolor='w', edgecolor='k')
        ax1 = fig.add_subplot(111)
        ax1.set_xlabel('Frequency [GHz]', fontsize = labelFontSize)
        ax1.set_ylabel(r'$\sigma_E$ [dBsm]', fontsize = labelFontSize)
        

        
    colors = ('tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan')
    linestyles = ('solid', 'dotted', 'dashed', 'dashdot')
    
    #===============================================================================
    # plt.plot(f1/1e9, sigShortUncloakedb, label = 'short uncloaked b', linewidth=linewidth*1.5)
    # plt.plot(f1/1e9, sigUncloakedb, label = 'uncloaked b', linewidth=linewidth*1.5)
    # plt.plot(f1/1e9, sigLongCloakedb, label = 'long cloaked b', linewidth=linewidth*1.5)
    # plt.plot(f1/1e9, sigShortCloakedb, label = 'short cloaked b', linewidth=linewidth*1.5)
    #===============================================================================
    
    #===============================================================================
    # plt.plot(f1/1e9, sigShortUncloakedd, label = 'short uncloaked d', linewidth=linewidth*1.5)
    # plt.plot(f1/1e9, sigUncloakedd, label = 'uncloaked d', linewidth=linewidth*1.5)
    # plt.plot(f1/1e9, sigLongCloakedd, label = 'long cloaked d', linewidth=linewidth*1.5)
    # plt.plot(f1/1e9, sigShortCloakedd, label = 'short cloaked d', linewidth=linewidth*1.5)
    #===============================================================================
    
    
    #===========================================================================
    # plt.plot(f1/1e9, sigShortUncloakede, label = 'short uncloaked e', linewidth=linewidth*1, color=colors[0%len(colors)])
    # plt.plot(f1/1e9, sigUncloakede, label = 'uncloaked e', linewidth=linewidth*1, color=colors[3%len(colors)])
    # plt.plot(f1/1e9, sigLongCloakede, label = 'long cloaked e', linewidth=linewidth*1, color=colors[1%len(colors)])
    # plt.plot(f1/1e9, sigShortCloakede, label = 'short cloaked e', linewidth=linewidth*1, color=colors[4%len(colors)])
    #   
    # plt.plot(f1/1e9, sigShortUncloakede2, label = 'short uncloaked e2', linewidth=linewidth*1.5, color=colors[0%len(colors)])
    # plt.plot(f1/1e9, sigUncloakede2, label = 'uncloaked e2', linewidth=linewidth*1.5, color=colors[3%len(colors)])
    # plt.plot(f1/1e9, sigLongCloakede2, label = 'long cloaked e2', linewidth=linewidth*1.5, color=colors[1%len(colors)])
    # plt.plot(f1/1e9, sigShortCloakede2, label = 'short cloaked e2', linewidth=linewidth*1.5, color=colors[4%len(colors)])
    #   
    # plt.plot(f1/1e9, sigShortUncloakede3, label = 'short uncloaked e3', linewidth=linewidth*2, color=colors[0%len(colors)])
    # plt.plot(f1/1e9, sigUncloakede3, label = 'uncloaked e3', linewidth=linewidth*2, color=colors[3%len(colors)])
    # plt.plot(f1/1e9, sigLongCloakede3, label = 'long cloaked e3', linewidth=linewidth*2, color=colors[1%len(colors)])
    # plt.plot(f1/1e9, sigShortCloakede3, label = 'short cloaked e3', linewidth=linewidth*2, color=colors[4%len(colors)])
    #===========================================================================
    
    
    
    
    
    #===========================================================================
    # plt.plot(f1/1e9, 10*np.log10(sigball30mmf), label = 'ball 30mmf', linewidth=linewidth*1.5, color=colors[0%len(colors)])
    # plt.plot(f1/1e9, 10*np.log10(sigball25mmf), label = 'ball 25mmf', linewidth=linewidth*1.5, color=colors[1%len(colors)])
    # plt.plot(f1/1e9, 10*np.log10(sigball20mmf), label = 'ball 20mmf', linewidth=linewidth*1.5, color=colors[2%len(colors)])
    # plt.plot(f1/1e9, 10*np.log10(sigball15mmf), label = 'ball 15mmf', linewidth=linewidth*1.5, color=colors[3%len(colors)])
    # plt.plot(f1/1e9, 10*np.log10(sigball10mmf), label = 'ball 10mmf', linewidth=linewidth*1.5, color=colors[4%len(colors)])
    # plt.plot(f1/1e9, 10*np.log10(sigball875mmf), label = 'ball 8.75mmf', linewidth=linewidth*1.5, color=colors[5%len(colors)])
    # plt.plot(f1/1e9, 10*np.log10(sigball75mmf), label = 'ball 7.5mmf', linewidth=linewidth*1.5, color=colors[6%len(colors)])
    # plt.plot(f1/1e9, 10*np.log10(sigball635mmf), label = 'ball 6.35mmf', linewidth=linewidth*1.5, color=colors[7%len(colors)])
    # plt.plot(f1/1e9, 10*np.log10(sigball5mmf), label = 'ball 5mmf', linewidth=linewidth*1.5, color=colors[8%len(colors)])
    # plt.plot(f1/1e9, 10*np.log10(sigball3975mmf), label = 'ball 3.975mmf', linewidth=linewidth*1.5, color=colors[9%len(colors)])
    # plt.plot(f1/1e9, 10*np.log10(sigball025mmf), label = 'ball 2.5mmf', linewidth=linewidth*1.5, color=colors[10%len(colors)])
    # plt.plot(f1/1e9, 10*np.log10(sigball015mmf), label = 'ball 1.5mmf', linewidth=linewidth*1.5, color=colors[11%len(colors)])
    #===========================================================================
    
    #===========================================================================
    # plt.plot(f1/1e9, sigball30mmR, label = 'ball 30mmR', linewidth=linewidth*1.5, color=colors[0%len(colors)])
    # plt.plot(f1/1e9, sigball25mmR, label = 'ball 25mmR', linewidth=linewidth*1.5, color=colors[1%len(colors)])
    # plt.plot(f1/1e9, sigball20mmR, label = 'ball 20mmR', linewidth=linewidth*1.5, color=colors[2%len(colors)])
    # plt.plot(f1/1e9, sigball15mmR, label = 'ball 15mmR', linewidth=linewidth*1.5, color=colors[3%len(colors)])
    # plt.plot(f1/1e9, sigball10mmR, label = 'ball 10mmR', linewidth=linewidth*1.5, color=colors[4%len(colors)])
    # plt.plot(f1/1e9, sigball875mmR, label = 'ball 8.75mmR', linewidth=linewidth*1.5, color=colors[5%len(colors)])
    # plt.plot(f1/1e9, sigball75mmR, label = 'ball 7.5mmR', linewidth=linewidth*1.5, color=colors[6%len(colors)])
    # plt.plot(f1/1e9, sigball635mmR, label = 'ball 6.35mmR', linewidth=linewidth*1.5, color=colors[7%len(colors)])
    # plt.plot(f1/1e9, sigball5mmR, label = 'ball 5mmR', linewidth=linewidth*1.5, color=colors[8%len(colors)])
    # plt.plot(f1/1e9, sigball3975mmR, label = 'ball 3.975mmR', linewidth=linewidth*1.5, color=colors[9%len(colors)])
    # plt.plot(f1/1e9, sigball025mmR, label = 'ball 2.5mmR', linewidth=linewidth*1.5, color=colors[10%len(colors)])
    #===========================================================================
    
    #===============================================================================
    # plt.plot(f1/1e9, sigball35mmg, label = 'ball 35mmg', linewidth=linewidth*1.5, color=colors[0%len(colors)])
    # plt.plot(f1/1e9, sigball15mmg, label = 'ball 15mmg', linewidth=linewidth*1.5, color=colors[3%len(colors)])
    # plt.plot(f1/1e9, sigball75mmg, label = 'ball 7.5mmg', linewidth=linewidth*1.5, color=colors[6%len(colors)])
    # plt.plot(f1/1e9, sigball5mmg, label = 'ball 5mmg', linewidth=linewidth*1.5, color=colors[8%len(colors)])
    # plt.plot(f1/1e9, sigball3975mmg, label = 'ball 3.975mmg', linewidth=linewidth*1.5, color=colors[9%len(colors)])
    # plt.plot(f1/1e9, sigball025mmg, label = 'ball 2.5mmg', linewidth=linewidth*1.5, color=colors[10%len(colors)])
    #===============================================================================
    
    #===========================================================================
    # plt.plot(f1/1e9, 10*np.log10(sigShortUncloakedg), label = 'short uncloaked g', linewidth=linewidth*1, color=colors[0%len(colors)])
    # plt.plot(f1/1e9, 10*np.log10(sigUncloakedg), label = 'uncloaked g', linewidth=linewidth*1, color=colors[3%len(colors)])
    # plt.plot(f1/1e9, 10*np.log10(sigLongCloakedg), label = 'long cloaked g', linewidth=linewidth*1, color=colors[1%len(colors)])
    # plt.plot(f1/1e9, 10*np.log10(sigShortCloakedg), label = 'short cloaked g', linewidth=linewidth*1, color=colors[4%len(colors)])
    #    
    # plt.plot(f1/1e9, 10*np.log10(sigShortUncloakedg2), label = 'short uncloaked g2', linewidth=linewidth*1.5, color=colors[0%len(colors)])
    # plt.plot(f1/1e9, 10*np.log10(sigUncloakedg2), label = 'uncloaked g2', linewidth=linewidth*1.5, color=colors[3%len(colors)])
    # plt.plot(f1/1e9, 10*np.log10(sigLongCloakedg2), label = 'long cloaked g2', linewidth=linewidth*1.5, color=colors[1%len(colors)])
    # plt.plot(f1/1e9, 10*np.log10(sigShortCloakedg2), label = 'short cloaked g2', linewidth=linewidth*1.5, color=colors[4%len(colors)])
    #    
    # plt.plot(f1/1e9, 10*np.log10(sigShortUncloakedg3), label = 'short uncloaked g3', linewidth=linewidth*2, color=colors[0%len(colors)])
    # plt.plot(f1/1e9, 10*np.log10(sigUncloakedg3), label = 'uncloaked g3', linewidth=linewidth*2, color=colors[3%len(colors)])
    # plt.plot(f1/1e9, 10*np.log10(sigLongCloakedg3), label = 'long cloaked g3', linewidth=linewidth*2, color=colors[1%len(colors)])
    # plt.plot(f1/1e9, 10*np.log10(sigShortCloakedg3), label = 'short cloaked g3', linewidth=linewidth*2, color=colors[4%len(colors)])
    #===========================================================================
    
    
    #===========================================================================
    # plt.plot(f1/1e9, 10*np.log10(sigball30mmSetE), label = 'ball 30mmSetE', linewidth=linewidth*1.5, color=colors[0%len(colors)])
    # plt.plot(f1/1e9, 10*np.log10(sigball10mmSetE), label = 'ball 10mmSetE', linewidth=linewidth*1.5, color=colors[4%len(colors)])
    # plt.plot(f1/1e9, 10*np.log10(sigball5mmSetE), label = 'ball 5mmSetE', linewidth=linewidth*1.5, color=colors[8%len(colors)])
    # plt.plot(f1/1e9, 10*np.log10(sigball025mmSetE), label = 'ball 2.5mmSetE', linewidth=linewidth*1.5, color=colors[10%len(colors)])
    # plt.plot(f1/1e9, 10*np.log10(sigball015mmSetE), label = 'ball 1.5mmSetE', linewidth=linewidth*1.5, color=colors[11%len(colors)])
    #===========================================================================
    
    #===========================================================================
    # plt.plot(f1/1e9, 10*np.log10(sigball30mmSetF), label = 'ball 30mmSetF', linewidth=linewidth*1.5, color=colors[0%len(colors)])
    # plt.plot(f1/1e9, 10*np.log10(sigball10mmSetF), label = 'ball 10mmSetF', linewidth=linewidth*1.5, color=colors[4%len(colors)])
    # plt.plot(f1/1e9, 10*np.log10(sigball5mmSetF), label = 'ball 5mmSetF', linewidth=linewidth*1.5, color=colors[8%len(colors)])
    # plt.plot(f1/1e9, 10*np.log10(sigball3975mmSetF), label = 'ball 3.975mmSetF', linewidth=linewidth*1.5, color=colors[9%len(colors)])
    # plt.plot(f1/1e9, 10*np.log10(sigball025mmSetF), label = 'ball 2.5mmSetF', linewidth=linewidth*1.5, color=colors[10%len(colors)])
    # plt.plot(f1/1e9, 10*np.log10(sigball015mmSetF), label = 'ball 1.5mmSetF', linewidth=linewidth*1.5, color=colors[11%len(colors)])
    #===========================================================================
    
    #===========================================================================
    # plt.plot(f1/1e9, 10*np.log10(sigball30mmSetG), label = 'ball 30mmSetG', linewidth=linewidth*1.5, color=colors[0%len(colors)])
    # plt.plot(f1/1e9, 10*np.log10(sigball10mmSetG), label = 'ball 10mmSetG', linewidth=linewidth*1.5, color=colors[4%len(colors)])
    # plt.plot(f1/1e9, 10*np.log10(sigball5mmSetG), label = 'ball 5mmSetG', linewidth=linewidth*1.5, color=colors[8%len(colors)])
    # plt.plot(f1/1e9, 10*np.log10(sigball3975mmSetG), label = 'ball 3.975mmSetG', linewidth=linewidth*1.5, color=colors[9%len(colors)])
    # plt.plot(f1/1e9, 10*np.log10(sigball025mmSetG), label = 'ball 2.5mmSetG', linewidth=linewidth*1.5, color=colors[10%len(colors)])
    # plt.plot(f1/1e9, 10*np.log10(sigball015mmSetG), label = 'ball 1.5mmSetG', linewidth=linewidth*1.5, color=colors[11%len(colors)])
    #===========================================================================
    
    #===========================================================================
    # plt.plot(f1/1e9, 10*np.log10(sigball30mmSetH), label = 'ball 30mmSetH', linewidth=linewidth*1.5, color=colors[0%len(colors)])
    # plt.plot(f1/1e9, 10*np.log10(sigball10mmSetH), label = 'ball 10mmSetH', linewidth=linewidth*1.5, color=colors[4%len(colors)])
    # plt.plot(f1/1e9, 10*np.log10(sigball5mmSetH), label = 'ball 5mmSetH', linewidth=linewidth*1.5, color=colors[8%len(colors)])
    # plt.plot(f1/1e9, 10*np.log10(sigball3975mmSetH), label = 'ball 3.975mmSetH', linewidth=linewidth*1.5, color=colors[9%len(colors)])
    # plt.plot(f1/1e9, 10*np.log10(sigball025mmSetH), label = 'ball 2.5mmSetH', linewidth=linewidth*1.5, color=colors[10%len(colors)])
    # plt.plot(f1/1e9, 10*np.log10(sigball015mmSetH), label = 'ball 1.5mmSetH', linewidth=linewidth*1.5, color=colors[11%len(colors)])
    #===========================================================================
    
    
    #===========================================================================
    # plt.plot(f1/1e9, 10*np.log10(sigball30mmSetI), label = 'ball 30mmSetI', linewidth=linewidth*1.5, color=colors[0%len(colors)])
    # plt.plot(f1/1e9, 10*np.log10(sigball10mmSetI), label = 'ball 10mmSetI', linewidth=linewidth*1.5, color=colors[4%len(colors)])
    # plt.plot(f1/1e9, 10*np.log10(sigball5mmSetI), label = 'ball 5mmSetI', linewidth=linewidth*1.5, color=colors[8%len(colors)])
    # plt.plot(f1/1e9, 10*np.log10(sigball3975mmSetI), label = 'ball 3.975mmSetI', linewidth=linewidth*1.5, color=colors[9%len(colors)])
    # plt.plot(f1/1e9, 10*np.log10(sigball025mmSetI), label = 'ball 2.5mmSetI', linewidth=linewidth*1.5, color=colors[10%len(colors)])
    #===========================================================================
    
    #===========================================================================
    # plt.plot(fSetJ/1e9, 10*np.log10(sigball30mmSetJ), label = 'ball 30mmSetJ', linewidth=linewidth*1.5, color=colors[0%len(colors)])
    # plt.plot(fSetJ/1e9, 10*np.log10(sigball10mmSetJ), label = 'ball 10mmSetJ', linewidth=linewidth*1.5, color=colors[4%len(colors)])
    # plt.plot(fSetJ/1e9, 10*np.log10(sigball5mmSetJ), label = 'ball 5mmSetJ', linewidth=linewidth*1.5, color=colors[8%len(colors)])
    # plt.plot(fSetJ/1e9, 10*np.log10(sigball3975mmSetJ), label = 'ball 3.975mmSetJ', linewidth=linewidth*1.5, color=colors[9%len(colors)])
    # plt.plot(fSetJ/1e9, 10*np.log10(sigball025mmSetJ), label = 'ball 2.5mmSetJ', linewidth=linewidth*1.5, color=colors[10%len(colors)])
    #===========================================================================
    
    
    
    #===========================================================================
    # plt.plot(fSetK/1e9, 10*np.log10(sigball30mmSetK), label = 'ball 30mmSetK', linewidth=linewidth*1.5, color=colors[0%len(colors)])
    # plt.plot(fSetK/1e9, 10*np.log10(sigball25mmSetK), label = 'ball 25mmSetK', linewidth=linewidth*1.5, color=colors[1%len(colors)])
    # plt.plot(fSetK/1e9, 10*np.log10(sigball20mmSetK), label = 'ball 20mmSetK', linewidth=linewidth*1.5, color=colors[2%len(colors)])
    # plt.plot(fSetK/1e9, 10*np.log10(sigball15mmSetK), label = 'ball 15mmSetK', linewidth=linewidth*1.5, color=colors[3%len(colors)])
    # plt.plot(fSetK/1e9, 10*np.log10(sigball10mmSetK), label = 'ball 10mmSetK', linewidth=linewidth*1.5, color=colors[4%len(colors)])
    # plt.plot(fSetK/1e9, 10*np.log10(sigball875mmSetK), label = 'ball 8.75mmSetK', linewidth=linewidth*1.5, color=colors[5%len(colors)])
    # plt.plot(fSetK/1e9, 10*np.log10(sigball635mmSetK), label = 'ball 6.35mmSetK', linewidth=linewidth*1.5, color=colors[7%len(colors)])
    # plt.plot(fSetK/1e9, 10*np.log10(sigball5mmSetK), label = 'ball 5mmSetK', linewidth=linewidth*1.5, color=colors[8%len(colors)])
    # plt.plot(fSetK/1e9, 10*np.log10(sigball3975mmSetK), label = 'ball 3.975mmSetK', linewidth=linewidth*1.5, color=colors[9%len(colors)])
    # plt.plot(fSetK/1e9, 10*np.log10(sigball025mmSetK), label = 'ball 2.5mmSetK', linewidth=linewidth*1.5, color=colors[10%len(colors)])
    # plt.plot(fSetK/1e9, 10*np.log10(sigball015mmSetK), label = 'ball 1.5mmSetK', linewidth=linewidth*1.5, color=colors[11%len(colors)])
    #===========================================================================
    
    #===========================================================================
    # plt.plot(fSetK2/1e9, 10*np.log10(sigball50mmSetK2), label = 'ball 50mmSetK2', linewidth=linewidth*2.5, color=colors[12%len(colors)])
    # plt.plot(fSetK2/1e9, 10*np.log10(sigball5mmSetK2), label = 'ball 5mmSetK2', linewidth=linewidth*2.5, color=colors[8%len(colors)])
    # plt.plot(fSetK2/1e9, 10*np.log10(sigball3975mmSetK2), label = 'ball 3.975mmSetK2', linewidth=linewidth*2.5, color=colors[9%len(colors)])
    # plt.plot(fSetK2/1e9, 10*np.log10(sigball025mmSetK2), label = 'ball 2.5mmSetK2', linewidth=linewidth*2.5, color=colors[10%len(colors)])
    # plt.plot(fSetK2/1e9, 10*np.log10(sigball015mmSetK2), label = 'ball 1.5mmSetK2', linewidth=linewidth*2.5, color=colors[11%len(colors)])
    #===========================================================================
    
    
    #===========================================================================
    # plt.plot(fSetK/1e9, 10*np.log10(sigShortUncloakedSetK), linewidth=linewidth*1, color=colors[0%len(colors)])#, label = 'Short Uncloaked')
    # plt.plot(fSetK/1e9, 10*np.log10(sigUncloakedSetK), linewidth=linewidth*1, color=colors[3%len(colors)])#, label = 'Uncloaked')
    # #plt.plot(fSetK/1e9, 10*np.log10(sigLongCloakedSetK), linewidth=linewidth*1, color=colors[1%len(colors)])#, label = 'Long Cloaked')
    # plt.plot(fSetK/1e9, 10*np.log10(sigShortCloakedSetK), linewidth=linewidth*1, color=colors[4%len(colors)])#, label = 'Short Cloaked')
    #                  
    # plt.plot(fSetK/1e9, 10*np.log10(sigShortUncloakedSetK2), linewidth=linewidth*1.5, color=colors[0%len(colors)])
    # plt.plot(fSetK/1e9, 10*np.log10(sigUncloakedSetK2), linewidth=linewidth*1.5, color=colors[3%len(colors)])
    # #plt.plot(fSetK/1e9, 10*np.log10(sigLongCloakedSetK2), linewidth=linewidth*1.5, color=colors[1%len(colors)])
    # plt.plot(fSetK/1e9, 10*np.log10(sigShortCloakedSetK2), linewidth=linewidth*1.5, color=colors[4%len(colors)])
    #                  
    # plt.plot(fSetK/1e9, 10*np.log10(sigShortUncloakedSetK3), linewidth=linewidth*2, color=colors[0%len(colors)])
    # plt.plot(fSetK/1e9, 10*np.log10(sigUncloakedSetK3), linewidth=linewidth*2, color=colors[3%len(colors)])
    # #plt.plot(fSetK/1e9, 10*np.log10(sigLongCloakedSetK3), linewidth=linewidth*2, color=colors[1%len(colors)])
    # plt.plot(fSetK/1e9, 10*np.log10(sigShortCloakedSetK3), linewidth=linewidth*2, color=colors[4%len(colors)])
    #===========================================================================
    
    
    BRCSSphereCalAvg = np.mean(np.array([BackwardScatteringCS(fSetK,25*1e-3),BackwardScatteringCS(fSetK,20*1e-3),BackwardScatteringCS(fSetK,15*1e-3),BackwardScatteringCS(fSetK,10*1e-3),BackwardScatteringCS(fSetK,8.75*1e-3),BackwardScatteringCS(fSetK,6.35*1e-3),BackwardScatteringCS(fSetK,5*1e-3),BackwardScatteringCS(fSetK,3.975*1e-3),BackwardScatteringCS(fSetK,2.5*1e-3)])/np.abs(np.array([monoball25mmSetK, monoball20mmSetK, monoball15mmSetK, monoball10mmSetK, monoball875mmSetK, monoball635mmSetK, monoball5mmSetK, monoball3975mmSetK, monoball025mmSetK]))**2, axis = 0)
    
    
    #===========================================================================
    # BRCSSphereCal = BackwardScatteringCS(fG, 15e-3)/np.abs(monoball15mmg)**2
    # plt.plot(fG/1e9, 10*np.log10(np.abs(monoShortUncloakedg)**2*BRCSSphereCal), label = 'short uncloaked g', linewidth=linewidth*1, color=colors[0%len(colors)])
    # plt.plot(fG/1e9, 10*np.log10(np.abs(monoUncloakedg)**2*BRCSSphereCal), label = 'uncloaked g', linewidth=linewidth*1, color=colors[3%len(colors)])
    # plt.plot(fG/1e9, 10*np.log10(np.abs(monoLongCloakedg)**2*BRCSSphereCal), label = 'long cloaked g', linewidth=linewidth*1, color=colors[1%len(colors)])
    # plt.plot(fG/1e9, 10*np.log10(np.abs(monoShortCloakedg)**2*BRCSSphereCal), label = 'short cloaked g', linewidth=linewidth*1, color=colors[4%len(colors)])
    #    
    # plt.plot(fG/1e9, 10*np.log10(np.abs(monoShortUncloakedg2)**2*BRCSSphereCal), label = 'short uncloaked g2', linewidth=linewidth*1.5, color=colors[0%len(colors)])
    # plt.plot(fG/1e9, 10*np.log10(np.abs(monoUncloakedg2)**2*BRCSSphereCal), label = 'uncloaked g2', linewidth=linewidth*1.5, color=colors[3%len(colors)])
    # plt.plot(fG/1e9, 10*np.log10(np.abs(monoLongCloakedg2)**2*BRCSSphereCal), label = 'long cloaked g2', linewidth=linewidth*1.5, color=colors[1%len(colors)])
    # plt.plot(fG/1e9, 10*np.log10(np.abs(monoShortCloakedg2)**2*BRCSSphereCal), label = 'short cloaked g2', linewidth=linewidth*1.5, color=colors[4%len(colors)])
    #    
    # plt.plot(fG/1e9, 10*np.log10(np.abs(monoShortUncloakedg3)**2*BRCSSphereCal), label = 'short uncloaked g3', linewidth=linewidth*2, color=colors[0%len(colors)])
    # plt.plot(fG/1e9, 10*np.log10(np.abs(monoUncloakedg3)**2*BRCSSphereCal), label = 'uncloaked g3', linewidth=linewidth*2, color=colors[3%len(colors)])
    # plt.plot(fG/1e9, 10*np.log10(np.abs(monoLongCloakedg3)**2*BRCSSphereCal), label = 'long cloaked g3', linewidth=linewidth*2, color=colors[1%len(colors)])
    # plt.plot(fG/1e9, 10*np.log10(np.abs(monoShortCloakedg3)**2*BRCSSphereCal), label = 'short cloaked g3', linewidth=linewidth*2, color=colors[4%len(colors)])
    #===========================================================================
    
    
 
    plt.plot(fSetK/1e9, 10*np.log10(np.abs(monoShortUncloakedSetK)**2*BRCSSphereCalAvg), linewidth=linewidth*1, color=colors[0%len(colors)])
    plt.plot(fSetK/1e9, 10*np.log10(np.abs(monoUncloakedSetK)**2*BRCSSphereCalAvg), linewidth=linewidth*1, color=colors[3%len(colors)])
    #plt.plot(fSetK/1e9, 10*np.log10(np.abs(monoLongCloakedSetK)**2*BRCSSphereCalAvg), linewidth=linewidth*1, color=colors[1%len(colors)])
    plt.plot(fSetK/1e9, 10*np.log10(np.abs(monoShortCloakedSetK)**2*BRCSSphereCalAvg), linewidth=linewidth*1, color=colors[4%len(colors)])
                        
    plt.plot(fSetK/1e9, 10*np.log10(np.abs(monoShortUncloakedSetK2)**2*BRCSSphereCalAvg), linewidth=linewidth*1.5, color=colors[0%len(colors)])#, label = 'Short Uncloaked'
    plt.plot(fSetK/1e9, 10*np.log10(np.abs(monoUncloakedSetK2)**2*BRCSSphereCalAvg), linewidth=linewidth*1.5, color=colors[3%len(colors)])#, label = 'Uncloaked'
    #plt.plot(fSetK/1e9, 10*np.log10(np.abs(monoLongCloakedSetK2)**2*BRCSSphereCalAvg), label = 'Long Cloaked', linewidth=linewidth*1.5, color=colors[1%len(colors)])
    plt.plot(fSetK/1e9, 10*np.log10(np.abs(monoShortCloakedSetK2)**2*BRCSSphereCalAvg), linewidth=linewidth*1.5, color=colors[4%len(colors)])#, label = 'Prototype Cloaked'
                        
    plt.plot(fSetK/1e9, 10*np.log10(np.abs(monoShortUncloakedSetK3)**2*BRCSSphereCalAvg), linewidth=linewidth*2, color=colors[0%len(colors)])
    plt.plot(fSetK/1e9, 10*np.log10(np.abs(monoUncloakedSetK3)**2*BRCSSphereCalAvg), linewidth=linewidth*2, color=colors[3%len(colors)])
    #plt.plot(fSetK/1e9, 10*np.log10(np.abs(monoLongCloakedSetK3)**2*BRCSSphereCalAvg), linewidth=linewidth*2, color=colors[1%len(colors)])
    plt.plot(fSetK/1e9, 10*np.log10(np.abs(monoShortCloakedSetK3)**2*BRCSSphereCalAvg), linewidth=linewidth*2, color=colors[4%len(colors)])
                
    plotMonostaticSim.plotLines()
    

    #===========================================================================
    # plt.plot(fSetK/1e9, 10*np.log10(np.abs(monoball30mmSetK)**2*BRCSSphereCalAvg), label = 'ball 30mmSetK', linewidth=linewidth*1.5, color=colors[0%len(colors)])
    # plt.plot(fSetK/1e9, 10*np.log10(np.abs(monoball25mmSetK)**2*BRCSSphereCalAvg), label = 'ball 25mmSetK', linewidth=linewidth*1.5, color=colors[1%len(colors)])
    # plt.plot(fSetK/1e9, 10*np.log10(np.abs(monoball20mmSetK)**2*BRCSSphereCalAvg), label = 'ball 20mmSetK', linewidth=linewidth*1.5, color=colors[2%len(colors)])
    # plt.plot(fSetK/1e9, 10*np.log10(np.abs(monoball15mmSetK)**2*BRCSSphereCalAvg), label = 'ball 15mmSetK', linewidth=linewidth*1.5, color=colors[3%len(colors)])
    # plt.plot(fSetK/1e9, 10*np.log10(np.abs(monoball10mmSetK)**2*BRCSSphereCalAvg), label = 'ball 10mmSetK', linewidth=linewidth*1.5, color=colors[4%len(colors)])
    # plt.plot(fSetK/1e9, 10*np.log10(np.abs(monoball875mmSetK)**2*BRCSSphereCalAvg), label = 'ball 8.75mmSetK', linewidth=linewidth*1.5, color=colors[5%len(colors)])
    # plt.plot(fSetK/1e9, 10*np.log10(np.abs(monoball635mmSetK)**2*BRCSSphereCalAvg), label = 'ball 6.35mmSetK', linewidth=linewidth*1.5, color=colors[7%len(colors)])
    # plt.plot(fSetK/1e9, 10*np.log10(np.abs(monoball5mmSetK)**2*BRCSSphereCalAvg), label = 'ball 5mmSetK', linewidth=linewidth*1.5, color=colors[8%len(colors)])
    # plt.plot(fSetK/1e9, 10*np.log10(np.abs(monoball3975mmSetK)**2*BRCSSphereCalAvg), label = 'ball 3.975mmSetK', linewidth=linewidth*1.5, color=colors[9%len(colors)])
    # plt.plot(fSetK/1e9, 10*np.log10(np.abs(monoball025mmSetK)**2*BRCSSphereCalAvg), label = 'ball 2.5mmSetK', linewidth=linewidth*1.5, color=colors[10%len(colors)])
    # plt.plot(fSetK/1e9, 10*np.log10(np.abs(monoball015mmSetK)**2*BRCSSphereCalAvg), label = 'ball 1.5mmSetK', linewidth=linewidth*1.5, color=colors[11%len(colors)])
    #===========================================================================
    
    #===========================================================================
    # BRCSSphereCal = BackwardScatteringCS(fF, 30e-3)/np.abs(monoball30mmf)**2
    # plt.plot(fF/1e9, 10*np.log10(np.abs(monoball30mmf)**2*BRCSSphereCal), label = 'ball 30mmf', linewidth=linewidth*1.5, color=colors[0%len(colors)])
    # plt.plot(fF/1e9, 10*np.log10(np.abs(monoball25mmf)**2*BRCSSphereCal), label = 'ball 25mmf', linewidth=linewidth*1.5, color=colors[1%len(colors)])
    # plt.plot(fF/1e9, 10*np.log10(np.abs(monoball20mmf)**2*BRCSSphereCal), label = 'ball 20mmf', linewidth=linewidth*1.5, color=colors[2%len(colors)])
    # plt.plot(fF/1e9, 10*np.log10(np.abs(monoball15mmf)**2*BRCSSphereCal), label = 'ball 15mmf', linewidth=linewidth*1.5, color=colors[3%len(colors)])
    # plt.plot(fF/1e9, 10*np.log10(np.abs(monoball10mmf)**2*BRCSSphereCal), label = 'ball 10mmf', linewidth=linewidth*1.5, color=colors[4%len(colors)])
    # plt.plot(fF/1e9, 10*np.log10(np.abs(monoball875mmf)**2*BRCSSphereCal), label = 'ball 8.75mmf', linewidth=linewidth*1.5, color=colors[5%len(colors)])
    # plt.plot(fF/1e9, 10*np.log10(np.abs(monoball75mmf)**2*BRCSSphereCal), label = 'ball 7.5mmf', linewidth=linewidth*1.5, color=colors[6%len(colors)])
    # plt.plot(fF/1e9, 10*np.log10(np.abs(monoball635mmf)**2*BRCSSphereCal), label = 'ball 6.35mmf', linewidth=linewidth*1.5, color=colors[7%len(colors)])
    # plt.plot(fF/1e9, 10*np.log10(np.abs(monoball5mmf)**2*BRCSSphereCal), label = 'ball 5mmf', linewidth=linewidth*1.5, color=colors[8%len(colors)])
    # plt.plot(fF/1e9, 10*np.log10(np.abs(monoball3975mmf)**2*BRCSSphereCal), label = 'ball 3.975mmf', linewidth=linewidth*1.5, color=colors[9%len(colors)])
    # plt.plot(fF/1e9, 10*np.log10(np.abs(monoball025mmf)**2*BRCSSphereCal), label = 'ball 2.5mmf', linewidth=linewidth*1.5, color=colors[10%len(colors)])
    # plt.plot(fF/1e9, 10*np.log10(np.abs(monoball015mmf)**2*BRCSSphereCal), label = 'ball 1.5mmf', linewidth=linewidth*1.5, color=colors[11%len(colors)])
    #===========================================================================
 
    
    #plt.plot(f1/1e9, monoball30mmR, label = 'ball 30mmR', linewidth=linewidth*1.5, color=colors[0%len(colors)])
    
    
    #===========================================================================
    # BRCSSphereCal = BackwardScatteringCS(f1, 30e-3)/np.abs(monoball30mmR)**2
    # plt.plot(f1/1e9, 10*np.log10(np.abs(monoball30mmR)**2*BRCSSphereCal), label = 'ball 30mmR', linewidth=linewidth*1.5, color=colors[0%len(colors)])
    # plt.plot(f1/1e9, 10*np.log10(np.abs(monoball25mmR)**2*BRCSSphereCal), label = 'ball 25mmR', linewidth=linewidth*1.5, color=colors[1%len(colors)])
    # plt.plot(f1/1e9, 10*np.log10(np.abs(monoball20mmR)**2*BRCSSphereCal), label = 'ball 20mmR', linewidth=linewidth*1.5, color=colors[2%len(colors)])
    # plt.plot(f1/1e9, 10*np.log10(np.abs(monoball15mmR)**2*BRCSSphereCal), label = 'ball 15mmR', linewidth=linewidth*1.5, color=colors[3%len(colors)])
    # plt.plot(f1/1e9, 10*np.log10(np.abs(monoball10mmR)**2*BRCSSphereCal), label = 'ball 10mmR', linewidth=linewidth*1.5, color=colors[4%len(colors)])
    # plt.plot(f1/1e9, 10*np.log10(np.abs(monoball875mmR)**2*BRCSSphereCal), label = 'ball 8.75mmR', linewidth=linewidth*1.5, color=colors[5%len(colors)])
    # plt.plot(f1/1e9, 10*np.log10(np.abs(monoball75mmR)**2*BRCSSphereCal), label = 'ball 7.5mmR', linewidth=linewidth*1.5, color=colors[6%len(colors)])
    # plt.plot(f1/1e9, 10*np.log10(np.abs(monoball635mmR)**2*BRCSSphereCal), label = 'ball 6.35mmR', linewidth=linewidth*1.5, color=colors[7%len(colors)])
    # plt.plot(f1/1e9, 10*np.log10(np.abs(monoball5mmR)**2*BRCSSphereCal), label = 'ball 5mmR', linewidth=linewidth*1.5, color=colors[8%len(colors)])
    # plt.plot(f1/1e9, 10*np.log10(np.abs(monoball3975mmR)**2*BRCSSphereCal), label = 'ball 3.975mmR', linewidth=linewidth*1.5, color=colors[9%len(colors)])
    # plt.plot(f1/1e9, 10*np.log10(np.abs(monoball025mmR)**2*BRCSSphereCal), label = 'ball 2.5mmR', linewidth=linewidth*1.5, color=colors[10%len(colors)])
    #===========================================================================
    
    #===========================================================================
    # ##un-CAL'd
    # gain = 15 ##approx. gain for standard gain horn antennas
    # BRCSSphereCal = (4*pi)**3 * (dAntennas2/2)**4 * (gain)**-2 / 10 * 2
    # plt.plot(fSetK/1e9, 10*np.log10(np.abs(monoball30mmSetK)**2*BRCSSphereCal/(c/fSetK)**2), label = 'ball 30mmSetK', linewidth=linewidth*1.5, color=colors[0%len(colors)])
    # plt.plot(fSetK/1e9, 10*np.log10(np.abs(monoball25mmSetK)**2*BRCSSphereCal/(c/fSetK)**2), label = 'ball 25mmSetK', linewidth=linewidth*1.5, color=colors[1%len(colors)])
    # plt.plot(fSetK/1e9, 10*np.log10(np.abs(monoball20mmSetK)**2*BRCSSphereCal/(c/fSetK)**2), label = 'ball 20mmSetK', linewidth=linewidth*1.5, color=colors[2%len(colors)])
    # plt.plot(fSetK/1e9, 10*np.log10(np.abs(monoball15mmSetK)**2*BRCSSphereCal/(c/fSetK)**2), label = 'ball 15mmSetK', linewidth=linewidth*1.5, color=colors[3%len(colors)])
    # plt.plot(fSetK/1e9, 10*np.log10(np.abs(monoball10mmSetK)**2*BRCSSphereCal/(c/fSetK)**2), label = 'ball 10mmSetK', linewidth=linewidth*1.5, color=colors[4%len(colors)])
    # plt.plot(fSetK/1e9, 10*np.log10(np.abs(monoball875mmSetK)**2*BRCSSphereCal/(c/fSetK)**2), label = 'ball 8.75mmSetK', linewidth=linewidth*1.5, color=colors[5%len(colors)])
    # plt.plot(fSetK/1e9, 10*np.log10(np.abs(monoball635mmSetK)**2*BRCSSphereCal/(c/fSetK)**2), label = 'ball 6.35mmSetK', linewidth=linewidth*1.5, color=colors[7%len(colors)])
    # plt.plot(fSetK/1e9, 10*np.log10(np.abs(monoball5mmSetK)**2*BRCSSphereCal/(c/fSetK)**2), label = 'ball 5mmSetK', linewidth=linewidth*1.5, color=colors[8%len(colors)])
    # plt.plot(fSetK/1e9, 10*np.log10(np.abs(monoball3975mmSetK)**2*BRCSSphereCal/(c/fSetK)**2), label = 'ball 3.975mmSetK', linewidth=linewidth*1.5, color=colors[9%len(colors)])
    # plt.plot(fSetK/1e9, 10*np.log10(np.abs(monoball025mmSetK)**2*BRCSSphereCal/(c/fSetK)**2), label = 'ball 2.5mmSetK', linewidth=linewidth*1.5, color=colors[10%len(colors)])
    # plt.plot(fSetK/1e9, 10*np.log10(np.abs(monoball015mmSetK)**2*BRCSSphereCal/(c/fSetK)**2), label = 'ball 1.5mmSetK', linewidth=linewidth*1.5, color=colors[11%len(colors)])
    #===========================================================================
    
    
    #===========================================================================
    # BRCSSphereCal = BackwardScatteringCS(fSetK, 30e-3)/np.abs(monoball30mmSetK)**2
    # BRCSSphereCal = BRCSSphereCalAvg
    # plt.plot(fSetK/1e9, 10*np.log10(np.abs(monoball30mmSetK)**2*BRCSSphereCal), label = 'ball 30mmSetK', linewidth=linewidth*1.5, color=colors[0%len(colors)])
    # plt.plot(fSetK/1e9, 10*np.log10(np.abs(monoball25mmSetK)**2*BRCSSphereCal), label = 'ball 25mmSetK', linewidth=linewidth*1.5, color=colors[1%len(colors)])
    # plt.plot(fSetK/1e9, 10*np.log10(np.abs(monoball20mmSetK)**2*BRCSSphereCal), label = 'ball 20mmSetK', linewidth=linewidth*1.5, color=colors[2%len(colors)])
    # plt.plot(fSetK/1e9, 10*np.log10(np.abs(monoball15mmSetK)**2*BRCSSphereCal), label = 'ball 15mmSetK', linewidth=linewidth*1.5, color=colors[3%len(colors)])
    # plt.plot(fSetK/1e9, 10*np.log10(np.abs(monoball10mmSetK)**2*BRCSSphereCal), label = 'ball 10mmSetK', linewidth=linewidth*1.5, color=colors[4%len(colors)])
    # plt.plot(fSetK/1e9, 10*np.log10(np.abs(monoball875mmSetK)**2*BRCSSphereCal), label = 'ball 8.75mmSetK', linewidth=linewidth*1.5, color=colors[5%len(colors)])
    # plt.plot(fSetK/1e9, 10*np.log10(np.abs(monoball635mmSetK)**2*BRCSSphereCal), label = 'ball 6.35mmSetK', linewidth=linewidth*1.5, color=colors[7%len(colors)])
    # plt.plot(fSetK/1e9, 10*np.log10(np.abs(monoball5mmSetK)**2*BRCSSphereCal), label = 'ball 5mmSetK', linewidth=linewidth*1.5, color=colors[8%len(colors)])
    # plt.plot(fSetK/1e9, 10*np.log10(np.abs(monoball3975mmSetK)**2*BRCSSphereCal), label = 'ball 3.975mmSetK', linewidth=linewidth*1.5, color=colors[9%len(colors)])
    # plt.plot(fSetK/1e9, 10*np.log10(np.abs(monoball025mmSetK)**2*BRCSSphereCal), label = 'ball 2.5mmSetK', linewidth=linewidth*1.5, color=colors[10%len(colors)])
    # plt.plot(fSetK/1e9, 10*np.log10(np.abs(monoball015mmSetK)**2*BRCSSphereCal), label = 'ball 1.5mmSetK', linewidth=linewidth*1.5, color=colors[11%len(colors)])
    #===========================================================================
    
    ###PLOT theoretical SPHERES
    
    
    freqs = np.arange(6.5,15.5, .001)*1e9
    sizes = np.array([30, 25, 20, 15, 10, 8.75, 7.5, 6.35, 5, 3.975, 2.5, 1.5])*1e-3 #radii in m
    
    #===========================================================================
    # ##ECS
    # for i in range(len(sizes)):
    #     if(i == 6):
    #         pass
    #     else:
    #         F = ForwardFarField(freqs, sizes[i])
    #         sigma_ext = -np.imag(F*2*c/freqs)
    #         plt.plot(freqs/1e9, 10*np.log10(sigma_ext), color=colors[i%len(colors)], linestyle = '--')#, label = 'ball '+str(sizes[i]*1e3)+' mm sim')
    #   
    # ax1.set_title(r"$\sigma_E$s by Frequency, Padded + Time-Gated", fontsize = '22')
    # plt.ylim(-75,-20)
    #===========================================================================
    
    #===========================================================================
    # ###backwards RCS
    # for i in range(len(sizes)):
    #     BRCS = BackwardScatteringCS(freqs, sizes[i])
    #     plt.plot(freqs/1e9, 10*np.log10(BRCS), color=colors[i%len(colors)], linestyle = '--')#, label = 'ball '+str(sizes[i]*1e3)+' mm sim')
    #===========================================================================
        
    plt.xlim(6,15.6)
    ax1.set_ylabel(r'$\sigma_{\mathrm{mono}}$ [dBsm]', fontsize = labelFontSize)
    ax1.set_title(r"Measured Monostatic RCS $\sigma_{\mathrm{mono}}$ by Frequency", fontsize = titleFontSize)
    plt.ylim(-80,-20)
    
    ######
    
    if(not toPlotComparisons):
        ax1.grid()
        #plt.xlim(f[0]/1e9,f[np.alen(f)-1]/1e9)
        fig.tight_layout()
        ax1.legend(fontsize = legendFontSize, framealpha=0.66, ncol=1, loc = 'best')
        plt.show()



if __name__ == '__main__':
    theMainStuff()
