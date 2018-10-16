
# coding: utf-8

# In[13]:


def coupled(x, a, b, c, d, e, f, g):
    import math
    A = 1j*(x + c-2*d)
    B = (a + b)/4 - 1j*(c-d)/2 - 1j*(x-d)
    G = math.sqrt((e/2)**2 + ((c-d)/2)**2 - ((a-b)/4)**2)
    return f*a/math.pi*abs((b/2-A)/(B**2+G**2))**2 + g;

