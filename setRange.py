
# coding: utf-8

# In[6]:


def setRange(specNum,rangeLow,rangeHigh):
    '''plots spectrum number specNum in range rangeLow to rangeHigh'''
    import matplotlib.pyplot as plt
    import pickle   
    get_ipython().run_line_magic('matplotlib', 'inline')
    
    with open('QD_Data/seriesName.obj', 'rb') as f:  
        seriesName = pickle.load(f)
    with open(str(seriesName), 'rb') as f: 
        series = pickle.load(f)
    with open('QD_Data/energyName.obj', 'rb') as f:  
        energyName = pickle.load(f)
    with open(str(energyName), 'rb') as f:  
        energy = pickle.load(f)
        
    rangeH = rangeHigh
    rangeL = rangeLow
    x_data = energy[rangeLow:rangeHigh]
    y_data = series[specNum][rangeLow:rangeHigh]
    
    with open('QD_Data/rangeH.obj', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(rangeH, f)
    with open('QD_Data/rangeL.obj', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(rangeL, f)
    with open('QD_Data/x_data.obj', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(x_data, f)
    with open('QD_Data/y_data.obj', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(y_data, f)
    # And plot it
    plt.figure(figsize=(6, 4))
    return plt.scatter(x_data, y_data)

