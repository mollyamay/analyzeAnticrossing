
# coding: utf-8

# In[27]:


def plotSpectra(indexStep):
    '''plots loaded spectra. Subset of images can be plotted 
    by increasing variable indexStep.'''
    get_ipython().run_line_magic('matplotlib', 'inline')

    import matplotlib
    import numpy as np
    import matplotlib.pyplot as plt
    import pickle
    
    with open('QD_Data/seriesName.obj', 'rb') as f:  
        seriesName = pickle.load(f)
    with open(str(seriesName), 'rb') as f: 
        series = pickle.load(f)
    with open('QD_Data/energyName.obj', 'rb') as f:  
        energyName = pickle.load(f)
    with open(str(energyName), 'rb') as f:  
        energy = pickle.load(f)
    #plot subset of spectra 
    for i in range(1,len(series),indexStep):
        plt.plot(energy, series[i])
    #plt.axis([640, 750, 590, 2500])
    #plotName = 'Pushing/intensitySpectra'+'.png'
    #plt.savefig(plotName)
    plt.matshow(series, interpolation=None, aspect='auto', vmin = 0, vmax = 1500)
    plt.axis([290, 550, 0, len(series)])
    plt.show()
    return plt.show()

