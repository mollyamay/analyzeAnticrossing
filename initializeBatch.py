
# coding: utf-8

# In[1]:


def initializeBatch(fileName, start, end):
    '''this function reads the files with file names fileName + i defined by "start" and "end"
    into a pandas dataframe and creates a variable for the wavelength/energy. These are saved as filename_series and fileName_energy'''
    import pandas as pd
    from scipy import optimize
    import pickle
    series = []
    #define variable for number of pixels in scan
    from matplotlib import cm
    #read in csv file to create pandas dataframe
    for i in range(int(start),int(end)):
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
    seriesName = fileName + str(start) + '_series.obj'
    energyName = fileName + str(start) + '_energy.obj'
    with open(seriesName, 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(series, f)    
    with open(energyName, 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(energy, f)
    with open('QD_Data/seriesName.obj', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(seriesName, f)
    with open('QD_Data/energyName.obj', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(energyName, f)
    return print('data loaded')

