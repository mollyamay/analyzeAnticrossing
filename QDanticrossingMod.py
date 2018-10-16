
# coding: utf-8

# In[950]:


import matplotlib as mpl
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from scipy import optimize
from pandas import Series
from pandas import DataFrame
from pandas import TimeGrouper
import math
import warnings
import matplotlib.colorbar as cb
import matplotlib.colors as colors


# In[126]:


def initializeSingle(fileName):
    '''this function reads the file at fileName into a pandas dataframe and creates a variable for the wavelength.
    to append multiple single spectra, comment out line 10 after loading the first spectrum'''
    from scipy import optimize
    global data
    global wavelength
    global series
    #series = []
    #define variable for number of pixels in scan
    from matplotlib import cm
    #read in csv file to create pandas dataframe
    dataTrans = pd.read_csv(fileName, header = None, sep=',')
    #transpose data to make pixel information contained in the columns
    data = dataTrans.transpose()
    for i in range(1,data.shape[1],2):
        series.append(data[i])
    #extract wavelength axis
    wavelength = data[0].copy()
    return print('data loaded')


# In[138]:


initializeSingle('QD_Data/QD_PL53.csv')


# In[1149]:


def initializeBatch(fileName):
    '''this function reads the files with file names fileName + i defined by range in line 13
    into a pandas dataframe and creates a variable for the wavelength'''
    from scipy import optimize
    global data
    global wavelength
    global series
    global energy
    #series = []
    #define variable for number of pixels in scan
    from matplotlib import cm
    #read in csv file to create pandas dataframe
    for i in range(37,44):
        print(i)
        getFile = fileName + str(i) + '.csv'
        dataTrans = pd.read_csv(getFile, header = None, sep=',')
        #transpose data to make pixel information contained in the columns
        data = dataTrans.transpose()
        for i in range(1,data.shape[1],2):
            series.append(data[i])
    #extract wavelength axis
    wavelength = data[0].copy()
    energy = (4.135668*10**-6*3*10**8)/wavelength
    return print('data loaded')


# In[133]:


initializeBatch('QD_Data/QD_PL')


# In[613]:


def reload(fileName):
    '''this function reads the file at fileName into a pandas dataframe and creates a variable for the wavelength.
    to append multiple single spectra, comment out line 10 after loading the first spectrum'''
    from scipy import optimize
    global data
    global wavelength
    global series
    series = []
    #define variable for number of pixels in scan
    from matplotlib import cm
    #read in csv file to create pandas dataframe
    dataTrans = pd.read_csv(fileName, header = None, sep=',')
    #transpose data to make pixel information contained in the columns
    data = dataTrans.transpose()
    for i in range(1,data.shape[1]):
        series.append(data[i])
    return print('data loaded, length is: %d' %len(series))


# In[614]:


reload('QD_Data/QD_PL_1005_series.csv')


# In[494]:


fileString = 'QD_Data/QD_PL_1005'
seriesTemp = pd.DataFrame(series)
seriesTemp.to_csv(fileString + '_series.csv', header = False, index = False)
wlTemp = pd.DataFrame(wavelength)
wlTemp.to_csv(fileString + '_wavelength.csv', header = False, index = False)


# In[1150]:


def plotSpectra(indexStep):
    '''plots loaded spectra. Subset of images can be plotted 
    by increasing variable indexStep.'''
    import matplotlib.pyplot as plt
    global wavelength
    global series
    global energy
    #plot forward scan spectra 
    for i in range(1,len(series),indexStep):
        plt.plot(energy, series[i])
    #plt.axis([640, 750, 590, 2500])
    #plotName = 'Pushing/intensitySpectra'+'.png'
    #plt.savefig(plotName)
    plt.show()


# In[616]:


plotSpectra(50)


# In[499]:


plt.matshow(series, interpolation=None, aspect='auto', vmin = 0, vmax = 1500)
plt.axis([290, 550, 200, 250])
plt.show()


# In[201]:


def setRange(specNum,rangeLow,rangeHigh):
    '''plots spectrum number specNum in range rangeLow to rangeHigh'''
    global x_data
    global y_data
    global rangeH
    global rangeL
    get_ipython().run_line_magic('matplotlib', 'inline')
    rangeH = rangeHigh
    rangeL = rangeLow
    x_data = energy[rangeLow:rangeHigh]
    y_data = series[specNum][rangeLow:rangeHigh]
    # And plot it
    plt.figure(figsize=(6, 4))
    return plt.scatter(x_data, y_data)


# In[232]:


setRange(20, 295, 1000)


# In[233]:


def coupled(x, a, b, c, d, e, f, g):
    A = 1j*(x + c-2*d)
    B = (a + b)/4 - 1j*(c-d)/2 - 1j*(x-d)
    G = math.sqrt((e/2)**2 + ((c-d)/2)**2 - ((a-b)/4)**2)
    return f*a/math.pi*abs((b/2-A)/(B**2+G**2))**2 + g;


# In[617]:


def fitSpectrum():
    '''fits to coupled oscillator model with params: two widths, two energies, coupling strength, intensity, offset.'''
    global x_data
    global y_data
    global initParams
    global paramBounds
    global params
    initParams=[0.08, 0.12, 1.80, 1.86, 0.08, 200, 600]
    paramBounds=([0.05, 0.1, 1.6, 1.8, 0.02, 0, 0],[0.09, 0.17, 1.85, 2, 0.2, 100000, 10000])
    params, params_covariance = optimize.curve_fit(coupled, x_data, y_data,p0=initParams,sigma=None,absolute_sigma=False,check_finite=True,bounds=paramBounds)
    print(params)
    plt.figure(figsize=(6, 4))
    plt.scatter(x_data, y_data, label='Data')
    plt.plot(x_data, coupled(x_data, params[0], params[1], params[2], params[3], params[4], params[5], params[6]),
            label='Fitted function')
    plt.legend(loc='best')
    print('Error is:')
    print(np.sqrt(np.diag(params_covariance)))
    return plt.show()


# In[618]:


fitSpectrum()


# In[619]:


def fitSpectraSubset(rangeLow,rangeHigh, indexStep):
    '''fit a subset of spectra defined by "indexStep" to test_func from bounds (rangeLow to rangeHigh) to check fits'''
    global energy
    x_data = energy[rangeLow:rangeHigh]
    for i in range(1,len(series),indexStep):
        try:
            y_data = series[i][rangeLow:rangeHigh]
            params, params_covariance = optimize.curve_fit(coupled, x_data, y_data,p0=initParams,sigma=None,absolute_sigma=False,check_finite=True,bounds=paramBounds)
            plt.figure(figsize=(6, 4))
            plt.scatter(x_data, y_data, label='Data')
            plt.plot(x_data, coupled(x_data, params[0], params[1], params[2], params[3], params[4], params[5], params[6]),
                label='Fitted function')
            plt.legend(loc='best')
        except RuntimeError:
            print(i)
    return print('finished!')


# In[620]:


fitSpectraSubset(rangeL,rangeH, 50)


# In[621]:


def fitSpectra(rangeLow,rangeHigh):
    '''fit all spectra to coupled function and store fit params in rows of allParams and error in covarianceParams array'''
    global allParams
    global covarianceParams
    global energy
    global seriesFull
    global indexTemp
    
    allParams = []
    covarianceParams = []
    x_data = energy[rangeLow:rangeHigh]
    indexTemp = []
    seriesFull = series.copy()
    for i in range(0,len(series)):
        try:
            y_data = series[i][rangeLow:rangeHigh]
            params, params_covariance = optimize.curve_fit(coupled, x_data, y_data,p0=initParams,sigma=None,absolute_sigma=False,check_finite=True,bounds=paramBounds)
            allParams.append(params)
            errorTemp = np.sqrt(np.diag(params_covariance))
            covarianceParams.append(errorTemp)
        except RuntimeError:
            print(i)
            indexTemp.append(i)
        except ValueError:
            print(i)
            indexTemp.append(i)
    #delete unfitted spectra from series (seriesFull is backup)
    for i in indexTemp:
        del(series[i])
    return print('finished fitting! Series in %d long.'%len(series))


# In[633]:


fitSpectra(rangeL, rangeH)


# In[634]:


allParamsBack = allParams.copy()
covarParamsBack = covarianceParams.copy()


# In[686]:


def removeError():
    global indexTemp1
    global allParams
    global covarianceParams
    indexTemp1 = []
    allParams = allParamsBack.copy()
    covarianceParams = covarParamsBack.copy()
    print(len(allParams))
    for i in range(0, len(allParams)):
        div = [x/2 for x in allParams[i]]
        div1 = [x for x in covarianceParams[i]]
        sub = [div[x]-div1[x] for x in range(0,5)]
        for x in sub:
            if x < 0:
                indexTemp1.append(i)
                break
            else: continue
    #delete high error fit data
    print(len(indexTemp1))
    for i in reversed(indexTemp1):
        del(allParams[i])
        del(covarianceParams[i])
    return print('finished removing high error fit data, length is now %d.'%len(allParams))


# In[687]:


removeError()


# In[688]:


def plotSeriesParams():
    global allParamsFR
    global time
    time = list(range(0,len(allParams)))
    allParamsFR = pd.DataFrame(allParams)
    for i in range(0,5):
        plt.figure(figsize=(6, 4))
        plt.scatter(time,allParamsFR[i])
        plt.show()


# In[689]:


plotSeriesParams()


# In[690]:


def plotCovarParams():
    global covarianceParamsFR
    global covarTime
    covarTime = list(range(0,len(covarianceParams)))
    covarianceParamsFR = pd.DataFrame(covarianceParams)
    for i in range(0,5):
        plt.figure(figsize=(6, 4))
        plt.scatter(covarTime,covarianceParamsFR[i])
        plt.show()


# In[691]:


plotCovarParams()


# In[877]:


def color1(final,initial, fun,i):
    global m
    global b
    m = (final - initial)/(max(fun) - min(fun))
    b = initial - min(fun)*m
    return (m*fun[i]+b)/255


# In[1075]:


def makeCrossingFig(colorStr, dataNum):
    '''uses color map "jet" to plot lines'''
    import matplotlib.cm as cmx
    
    gName = 'g'+str(dataNum)
    name1 = 'LowRange' + str(dataNum)
    name2 = 'HighRange' + str(dataNum)
    w_qd = 'w_qd' + str(dataNum)
    w_sp = 'w_sp' + str(dataNum)
    nameDet = 'detuningPeaks' + str(dataNum)
    LpeaksName = 'Lpeaks' + str(dataNum)
    HpeaksName = 'Hpeaks' + str(dataNum)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # replace the next line 
    #jet = colors.Colormap('jet')
    # with
    jet = cm = plt.get_cmap(colorStr) 
    cNorm  = colors.Normalize(vmin=min(globals()[gName]), vmax=max(globals()[gName]))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

    lines = []
    for idx in range(0,len(globals()[name1])):
        line1 = globals()[name1][idx]
        line2 = globals()[name2][idx]
        colorVal = scalarMap.to_rgba(globals()[gName][idx])
        colorText = (
            'color: (%4.2f,%4.2f,%4.2f)'%(colorVal[0],colorVal[1],colorVal[2])
            )
        retLine1, = ax.plot(detuning, line1, color=colorVal, linewidth = 2, zorder = globals()[gName][idx])
        retLine2, = ax.plot(detuning, line2, color=colorVal, linewidth = 2, zorder = globals()[gName][idx])
        retLine3, = ax.plot(globals()[nameDet][idx], globals()[LpeaksName][idx], color=colorVal,marker='o', markersize = 7, markeredgecolor=(0.3,0.3,0.3), markeredgewidth = 0.5, zorder = globals()[gName][idx]*10)
        retLine4, = ax.plot(globals()[nameDet][idx], globals()[HpeaksName][idx], color=colorVal, marker='o', markersize = 7, markeredgecolor=(0.3,0.3,0.3), markeredgewidth = 0.5, zorder = globals()[gName][idx]*10)
        lines.append(retLine1)
        lines.append(retLine2)
        lines.append(retLine3)
        lines.append(retLine4)
    
    plt.plot(detuning,globals()[w_qd],color = (0.4,0.4,0.4),linewidth = 2)
    plt.plot(detuning,globals()[w_sp],color = 'k',linewidth = 2)
    plt.ylabel('Energy (eV))')
    plt.xlabel('QD Detuning (meV)')
    plt.tight_layout()
    plotName = '1005anticrossing'+str(dataNum)+'.png'
    plt.savefig(plotName)
    return plt.show()


# In[1077]:


makeCrossingFig('cool', 1)


# In[1147]:



def crossingCmap(gFun, color, str1):
    '''makes color scale for anticrossing curves. gFun is the g function to set the min 
    and max values, color is the python color, and str1 make the file name unique'''
    # Make a figure and axes with dimensions as desired.
    fig = plt.figure(figsize=(3, 2))
    ax1 = fig.add_axes([0.05, 0.8, 0.9, 0.15])

    # Set the colormap and norm to correspond to the data for which
    # the colorbar will be used.
    cmap = color
    norm = mpl.colors.Normalize(vmin=min(gFun), vmax=max(gFun))

    # ColorbarBase derives from ScalarMappable and puts a colorbar
    # in a specified axes, so it has everything needed for a
    # standalone colorbar.  There are many more kwargs, but the
    # following gives a basic continuous colorbar with ticks
    # and labels.
    cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap,
                                    norm=norm,
                                    orientation='horizontal')
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(18)
    cb1.set_label('g (meV)', fontsize=22)
    plotName = '1005CrossingColor'+str1+'.png'
    plt.savefig(plotName)
    return plt.show()


# In[1148]:


crossingCmap(g3, mpl.cm.plasma, 'g3')


# In[1127]:


def anticrossing(thresh1, thresh2, thresh3, thresh4):
    
    global detuningData
    global detuning
    global peaks1
    global peaks2
    global LowRange1
    global HighRange1
    global g1
    global QDRange1
    global SPRange1
    global detuningPeaks1
    global Hpeaks1
    global Lpeaks1
    global w_qd1
    global w_sp1
    global LowRange2
    global HighRange2
    global g2
    global QDRange2
    global SPRange2
    global detuningPeaks2
    global Hpeaks2
    global Lpeaks2
    global w_qd2
    global w_sp2
    global LowRange3
    global HighRange3
    global g3
    global QDRange3
    global SPRange3
    global detuningPeaks3
    global Hpeaks3
    global Lpeaks3
    global w_qd3
    global w_sp3
    global LowRange4
    global HighRange4
    global g4
    global QDRange4
    global SPRange4
    global detuningPeaks4
    global Hpeaks4
    global Lpeaks4
    global w_qd4
    global w_sp4
    global LowRange5
    global HighRange5
    global g5
    global QDRange5
    global SPRange5
    global detuningPeaks5
    global Hpeaks5
    global Lpeaks5
    global w_qd5
    global w_sp5
    
    #complex number warnings "taking real part of a number throws away imaginary part" ignored
    warnings.filterwarnings("ignore")
    
    detuningData = (allParamsFR[2] - allParamsFR[3])*1000
    a = allParamsFR[0]*1000  #gamma_1 in mev
    b = allParamsFR[1]*1000  #gamma_2 in mev
    c = allParamsFR[3]*1000  #w_sp in mev 
    g = allParamsFR[4]*1000  #Rabi frequency in mev
    
    detuning = list(range(-150,150))
    
    peaks1 = []
    peaks2 = []
    QDRange1 = []
    SPRange1=[]
    LowRange1=[]
    HighRange1=[]
    g1 = []
    QDRange2 = []
    SPRange2=[]
    LowRange2=[]
    HighRange2=[]
    g2 = []
    QDRange3 = []
    SPRange3=[]
    LowRange3=[]
    HighRange3=[]
    g3 = []
    QDRange4 = []
    SPRange4=[]
    LowRange4=[]
    HighRange4=[]
    g4 = []
    QDRange5 = []
    SPRange5=[]
    LowRange5=[]
    HighRange5=[]
    g5 = []
    Lpeaks1 = []
    Hpeaks1 = []
    detuningPeaks1 = []
    Lpeaks2 = []
    Hpeaks2 = []
    detuningPeaks2 = []
    Lpeaks3 = []
    Hpeaks3 = []
    detuningPeaks3 = []
    Lpeaks4 = []
    Hpeaks4 = []
    detuningPeaks4 = [] 
    Lpeaks5 = []
    Hpeaks5 = []
    detuningPeaks5 = []
    
    for i in range(0,len(allParams),3):     
        if allParams[i][3] < thresh1:
            #Find QD and SP dispersions for this detuning
            w_qd = [x/1000+c[i]/1000 for x in detuning]
            w_sp = [x*0+c[i]/1000 for x in detuning]
            #find anticrossings from these fit data
            pks1 = [np.real((x+2*c[i])/2-1j*(a[i]+b[i])/4 + math.sqrt((g[i]/2)**2 + 0.25*(x-1j*(b[i]-a[i])/2)**2))/1000 for x in detuning]
            pks2 = [np.real((x+2*c[i])/2-1j*(a[i]+b[i])/4 - math.sqrt((g[i]/2)**2 + 0.25*(x-1j*(b[i]-a[i])/2)**2))/1000 for x in detuning]
            #create array with these fit values
            QDRange1.append(w_qd)
            SPRange1.append(w_sp)
            LowRange1.append(pks1)
            HighRange1.append(pks2)
            g1.append(g[i])
            try:
                pks1 = [np.real((x+2*c[i])/2-1j*(a[i]+b[i])/4 + math.sqrt((g[i]/2)**2 + 0.25*(x-1j*(b[i]-a[i])/2)**2))/1000 for x in detuning]
                pks2 = [np.real((x+2*c[i])/2-1j*(a[i]+b[i])/4 - math.sqrt((g[i]/2)**2 + 0.25*(x-1j*(b[i]-a[i])/2)**2))/1000 for x in detuning]
                findMin = [abs(x-detuningData[i]) for x in detuning]
                val, idx = min((val, idx) for (idx, val) in enumerate(findMin))
                #plt.scatter(detuningData[i],pks1[idx])
                #plt.scatter(detuningData[i],pks2[idx])
                Lpeaks1.append(pks1[idx])
                Hpeaks1.append(pks2[idx])
                detuningPeaks1.append(detuning[idx])
            except ValueError:
                print(i)
        elif allParams[i][3] < thresh2:
            w_qd = [x/1000+c[i]/1000 for x in detuning]
            w_sp = [x*0+c[i]/1000 for x in detuning]
            #find anticrossings from these fit data
            pks1 = [np.real((x+2*c[i])/2-1j*(a[i]+b[i])/4 + math.sqrt((g[i]/2)**2 + 0.25*(x-1j*(b[i]-a[i])/2)**2))/1000 for x in detuning]
            pks2 = [np.real((x+2*c[i])/2-1j*(a[i]+b[i])/4 - math.sqrt((g[i]/2)**2 + 0.25*(x-1j*(b[i]-a[i])/2)**2))/1000 for x in detuning]
            #create array with these fit values
            QDRange2.append(w_qd)
            SPRange2.append(w_sp)
            LowRange2.append(pks1)
            HighRange2.append(pks2)
            g2.append(g[i])
            try:
                pks1 = [np.real((x+2*c[i])/2-1j*(a[i]+b[i])/4 + math.sqrt((g[i]/2)**2 + 0.25*(x-1j*(b[i]-a[i])/2)**2))/1000 for x in detuning]
                pks2 = [np.real((x+2*c[i])/2-1j*(a[i]+b[i])/4 - math.sqrt((g[i]/2)**2 + 0.25*(x-1j*(b[i]-a[i])/2)**2))/1000 for x in detuning]
                findMin = [abs(x-detuningData[i]) for x in detuning]
                val, idx = min((val, idx) for (idx, val) in enumerate(findMin))
                #plt.scatter(detuningData[i],pks1[idx])
                #plt.scatter(detuningData[i],pks2[idx])
                Lpeaks2.append(pks1[idx])
                Hpeaks2.append(pks2[idx])
                detuningPeaks2.append(detuning[idx])
            except ValueError:
                print(i)
        elif allParams[i][3] < thresh3:
            w_qd = [x/1000+c[i]/1000 for x in detuning]
            w_sp = [x*0+c[i]/1000 for x in detuning]
            #find anticrossings from these fit data
            pks1 = [np.real((x+2*c[i])/2-1j*(a[i]+b[i])/4 + math.sqrt((g[i]/2)**2 + 0.25*(x-1j*(b[i]-a[i])/2)**2))/1000 for x in detuning]
            pks2 = [np.real((x+2*c[i])/2-1j*(a[i]+b[i])/4 - math.sqrt((g[i]/2)**2 + 0.25*(x-1j*(b[i]-a[i])/2)**2))/1000 for x in detuning]
            #create array with these fit values
            QDRange3.append(w_qd)
            SPRange3.append(w_sp)
            LowRange3.append(pks1)
            HighRange3.append(pks2) 
            g3.append(g[i])
            try:
                pks1 = [np.real((x+2*c[i])/2-1j*(a[i]+b[i])/4 + math.sqrt((g[i]/2)**2 + 0.25*(x-1j*(b[i]-a[i])/2)**2))/1000 for x in detuning]
                pks2 = [np.real((x+2*c[i])/2-1j*(a[i]+b[i])/4 - math.sqrt((g[i]/2)**2 + 0.25*(x-1j*(b[i]-a[i])/2)**2))/1000 for x in detuning]
                findMin = [abs(x-detuningData[i]) for x in detuning]
                val, idx = min((val, idx) for (idx, val) in enumerate(findMin))
                #plt.scatter(detuningData[i],pks1[idx])
                #plt.scatter(detuningData[i],pks2[idx])
                Lpeaks3.append(pks1[idx])
                Hpeaks3.append(pks2[idx])
                detuningPeaks3.append(detuning[idx])
            except ValueError:
                print(i)
        elif allParams[i][3] < thresh4:
            w_qd = [x/1000+c[i]/1000 for x in detuning]
            w_sp = [x*0+c[i]/1000 for x in detuning]
            #find anticrossings from these fit data
            pks1 = [np.real((x+2*c[i])/2-1j*(a[i]+b[i])/4 + math.sqrt((g[i]/2)**2 + 0.25*(x-1j*(b[i]-a[i])/2)**2))/1000 for x in detuning]
            pks2 = [np.real((x+2*c[i])/2-1j*(a[i]+b[i])/4 - math.sqrt((g[i]/2)**2 + 0.25*(x-1j*(b[i]-a[i])/2)**2))/1000 for x in detuning]
            #create array with these fit values
            QDRange4.append(w_qd)
            SPRange4.append(w_sp)
            LowRange4.append(pks1)
            HighRange4.append(pks2) 
            g4.append(g[i])
            try:
                pks1 = [np.real((x+2*c[i])/2-1j*(a[i]+b[i])/4 + math.sqrt((g[i]/2)**2 + 0.25*(x-1j*(b[i]-a[i])/2)**2))/1000 for x in detuning]
                pks2 = [np.real((x+2*c[i])/2-1j*(a[i]+b[i])/4 - math.sqrt((g[i]/2)**2 + 0.25*(x-1j*(b[i]-a[i])/2)**2))/1000 for x in detuning]
                findMin = [abs(x-detuningData[i]) for x in detuning]
                val, idx = min((val, idx) for (idx, val) in enumerate(findMin))
                #plt.scatter(detuningData[i],pks1[idx])
                #plt.scatter(detuningData[i],pks2[idx])
                Lpeaks4.append(pks1[idx])
                Hpeaks4.append(pks2[idx])
                detuningPeaks4.append(detuning[idx])
            except ValueError:
                print(i)        
        elif allParams[i][3] >= thresh4:
            w_qd = [x/1000+c[i]/1000 for x in detuning]
            w_sp = [x*0+c[i]/1000 for x in detuning]
            #find anticrossings from these fit data
            pks1 = [np.real((x+2*c[i])/2-1j*(a[i]+b[i])/4 + math.sqrt((g[i]/2)**2 + 0.25*(x-1j*(b[i]-a[i])/2)**2))/1000 for x in detuning]
            pks2 = [np.real((x+2*c[i])/2-1j*(a[i]+b[i])/4 - math.sqrt((g[i]/2)**2 + 0.25*(x-1j*(b[i]-a[i])/2)**2))/1000 for x in detuning]
            #create array with these fit values
            QDRange5.append(w_qd)
            SPRange5.append(w_sp)
            LowRange5.append(pks1)
            HighRange5.append(pks2)
            g5.append(g[i])
            try:
                pks1 = [np.real((x+2*c[i])/2-1j*(a[i]+b[i])/4 + math.sqrt((g[i]/2)**2 + 0.25*(x-1j*(b[i]-a[i])/2)**2))/1000 for x in detuning]
                pks2 = [np.real((x+2*c[i])/2-1j*(a[i]+b[i])/4 - math.sqrt((g[i]/2)**2 + 0.25*(x-1j*(b[i]-a[i])/2)**2))/1000 for x in detuning]
                findMin = [abs(x-detuningData[i]) for x in detuning]
                val, idx = min((val, idx) for (idx, val) in enumerate(findMin))
                #plt.scatter(detuningData[i],pks1[idx])
                #plt.scatter(detuningData[i],pks2[idx])
                Lpeaks5.append(pks1[idx])
                Hpeaks5.append(pks2[idx])
                detuningPeaks5.append(detuning[idx])
            except ValueError:
                print(i)
        else:
            print(i)
      
    w_qd1 = [x/1000+np.mean(SPRange1) for x in detuning]
    w_sp1 = [x*0+np.mean(SPRange1) for x in detuning]
    w_qd2 = [x/1000+np.mean(SPRange2) for x in detuning]
    w_sp2 = [x*0+np.mean(SPRange2) for x in detuning]
    w_qd3 = [x/1000+np.mean(SPRange3) for x in detuning]
    w_sp3 = [x*0+np.mean(SPRange3) for x in detuning]
    w_qd4 = [x/1000+np.mean(SPRange4) for x in detuning]
    w_sp4 = [x*0+np.mean(SPRange4) for x in detuning]
    w_qd5 = [x/1000+np.mean(SPRange5) for x in detuning]
    w_sp5 = [x*0+np.mean(SPRange5) for x in detuning]
    
    print(w_sp1[0])
    makeCrossingFig('viridis',1)
    crossingCmap(g1, mpl.cm.viridis, 'g1')
    print(w_sp2[0])
    makeCrossingFig('plasma',2)
    crossingCmap(g2, mpl.cm.plasma, 'g2')
    print(w_sp3[0])
    makeCrossingFig('plasma',3)
    crossingCmap(g3, mpl.cm.plasma, 'g3')
    print(w_sp4[0])
    makeCrossingFig('copper',4)
    crossingCmap(g4, mpl.cm.copper, 'g4')
    print(w_sp5[0])
    makeCrossingFig('summer',5)
    crossingCmap(g5, mpl.cm.copper, 'g5')
    
    return print('yeehaw!')


# In[1128]:


anticrossing(1.8335, 1.837, 1.841, 1.855)


# In[1117]:


print(w_sp1[0])


# In[1121]:


print(w_sp3[0])

