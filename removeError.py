
# coding: utf-8

# In[7]:


def removeError():
    import matplotlib.pyplot as plt
    from pandas import DataFrame
    import pandas as pd
    import pickle
    with open('QD_Data/covarianceParams.obj', 'rb') as f:  
        covarParams = pickle.load(f)
    with open('QD_Data/allParams.obj', 'rb') as f:  
        allParams = pickle.load(f)
    indexTemp1 = []
    print(len(allParams))
    for i in range(0, len(allParams)):
        div = [x/2 for x in allParams[i]]
        div1 = [x for x in covarParams[i]]
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
        del(covarParams[i])
    with open('QD_Data/allParams.obj', 'wb') as f:  
        pickle.dump(allParams,f)
    with open('QD_Data/covarParams.obj', 'wb') as f:  
        pickle.dump(covarParams,f)
    return print('finished removing high error fit data, length is now %d.'%len(allParams))

