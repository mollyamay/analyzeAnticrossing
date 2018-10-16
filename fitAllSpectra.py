
# coding: utf-8

# In[10]:


def fitAllSpectra(rangeLow,rangeHigh,figStep):
    '''fit all spectra to coupled function and store fit params in rows of allParams and error in covarianceParams array'''
    import matplotlib.pyplot as plt
    import pickle
    from scipy import optimize
    from coupledEqn import coupled
    import math
    import numpy as np
    
    with open('QD_Data/x_data.obj', 'rb') as f:  
        x_data = pickle.load(f)
    
    with open('QD_Data/seriesName.obj', 'rb') as f:  
        seriesName = pickle.load(f)
    with open(str(seriesName), 'rb') as f: 
        series = pickle.load(f)
    with open('QD_Data/initParams.obj', 'rb') as f:  
        initParams = pickle.load(f)
    with open('QD_Data/paramBounds.obj', 'rb') as f:  
        paramBounds = pickle.load(f)
    with open('QD_Data/params.obj', 'rb') as f:  
        params = pickle.load(f)
        
    allParams = []
    covarianceParams = []
    indexTemp = []
    
    for i in range(0,len(series)):
        try:
            y_data = series[i][rangeLow:rangeHigh]
            params, params_covariance = optimize.curve_fit(coupled, x_data, y_data,p0=initParams,sigma=None,absolute_sigma=False,check_finite=True,bounds=paramBounds)
            allParams.append(params)
            errorTemp = np.sqrt(np.diag(params_covariance))
            covarianceParams.append(errorTemp)
            if i%figStep==0:
                plt.figure(figsize=(6, 4))
                plt.scatter(x_data, y_data, label='Data')
                plt.plot(x_data, coupled(x_data, params[0], params[1], params[2], params[3], params[4], params[5], params[6]),
                label='Fitted function')
                plt.legend(loc='best')
        except RuntimeError:
            print(i)
            indexTemp.append(i)
        except ValueError:
            print(i)
            indexTemp.append(i)
    with open('QD_Data/allParams.obj', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(allParams, f)
    with open('QD_Data/covarianceParams.obj', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(covarianceParams, f)
    with open('QD_Data/indexTemp.obj', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(indexTemp, f)
    return print('finished fitting! Series is %d long.'%len(series))

