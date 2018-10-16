
# coding: utf-8

# In[7]:


def fitSpectrum():
    '''fits to coupled oscillator model with params: two widths, two energies, coupling strength, intensity, offset.'''
    import matplotlib.pyplot as plt
    import pickle
    from scipy import optimize
    from coupledEqn import coupled
    import math
    import numpy as np
    
    with open('QD_Data/x_data.obj', 'rb') as f:  
        x_data = pickle.load(f)
    with open('QD_Data/y_data.obj', 'rb') as f:  
        y_data = pickle.load(f)
    initParams=[0.08, 0.12, 1.80, 1.86, 0.08, 200, 600]
    paramBounds=([0.05, 0.1, 1.6, 1.8, 0.02, 0, 0],[0.09, 0.17, 1.85, 2, 0.2, 100000, 10000])
    params, params_covariance = optimize.curve_fit(coupled, x_data, y_data,p0=initParams,sigma=None,absolute_sigma=False,check_finite=True,bounds=paramBounds)
    print(params)
    plt.figure(figsize=(6, 4))
    plt.scatter(x_data, y_data, label='Data')
    plt.plot(x_data, coupled(x_data, params[0], params[1], params[2], params[3], params[4], params[5], params[6]),
            label='Fitted function')
    plt.legend(loc='best')
    with open('QD_Data/initParams.obj', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(initParams, f)
    with open('QD_Data/paramBounds.obj', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(paramBounds, f)
    with open('QD_Data/params.obj', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(params, f)
    print('Error is:')
    print(np.sqrt(np.diag(params_covariance)))
    return plt.show()

