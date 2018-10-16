
# coding: utf-8

# In[11]:


def plotCovarParams():
    import matplotlib.pyplot as plt
    from pandas import DataFrame
    import pandas as pd
    import pickle
    with open('QD_Data/covarianceParams.obj', 'rb') as f:  
        covarParams = pickle.load(f)
    time = list(range(0,len(covarParams)))
    covarParamsFR = pd.DataFrame(covarParams)
    for i in range(0,5):
        plt.figure(figsize=(6, 4))
        plt.scatter(time,covarParamsFR[i])
    with open('QD_Data/covarParamsFR.obj', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(covarParamsFR, f)
    with open('QD_Data/time.obj', 'wb') as f:  
        pickle.dump(time,f) 
    return plt.show()

