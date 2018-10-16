
# coding: utf-8

# In[1]:


def makeCrossingFig(colorStr, dataNum):
    '''uses color map "jet" to plot lines'''
    import matplotlib.cm as cmx
    import matplotlib.pyplot as plt
    from pandas import DataFrame
    import pandas as pd
    import pickle
    import warnings
    import numpy as np
    import matplotlib.colorbar as cb
    import matplotlib.colors as colors
    import math
    
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


# In[7]:


def anticrossingPlots(thresh1, thresh2, thresh3, thresh4):
    
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from pandas import DataFrame
    import pandas as pd
    import pickle
    import warnings
    import numpy as np
    import math
    from crossingCmap import crossingCmap
    
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
    
    with open('QD_Data/allParamsFR.obj', 'rb') as f:  
        allParamsFR = pickle.load(f)
    with open('QD_Data/allParams.obj', 'rb') as f:  
        allParams = pickle.load(f)
  
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
    try:
        makeCrossingFig('viridis',1)
        crossingCmap(g1, mpl.cm.viridis, 'g1')
    except ValueError:
        print(1)
    print(w_sp2[0])
    try:
        makeCrossingFig('plasma',2)
        crossingCmap(g2, mpl.cm.plasma, 'g2')
    except ValueError:
        print(2)
    print(w_sp3[0])
    try:
        makeCrossingFig('plasma',3)
        crossingCmap(g3, mpl.cm.plasma, 'g3')
    except ValueError:
        print(3)
    print(w_sp4[0])
    try: 
        makeCrossingFig('copper',4)
        crossingCmap(g4, mpl.cm.copper, 'g4')
    except ValueError:
        print(4)
    print(w_sp5[0])
    try:
        makeCrossingFig('summer',5)
        crossingCmap(g5, mpl.cm.copper, 'g5')
    except ValueError:
        print(5)
    return print('yeehaw!')

