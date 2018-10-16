
# coding: utf-8

# In[3]:


def plotSeriesParams():
    import matplotlib.pyplot as plt
    from pandas import DataFrame
    import pandas as pd
    import pickle
    with open('QD_Data/allParams.obj', 'rb') as f:  
        allParams = pickle.load(f)
    time = list(range(0,len(allParams)))
    allParamsFR = pd.DataFrame(allParams)
    for i in range(0,5):
        plt.figure(figsize=(6, 4))
        plt.scatter(time,allParamsFR[i])
    with open('QD_Data/allParamsFR.obj', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(allParamsFR, f)
    with open('QD_Data/time.obj', 'wb') as f:  
        pickle.dump(time,f) 
    return plt.show()

