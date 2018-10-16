
# coding: utf-8

# In[3]:


def crossingCmap(gFun, color, str1):
    '''makes color scale for anticrossing curves. gFun is the g function to set the min 
    and max values, color is the python color, and str1 make the file name unique'''
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colorbar as cb
    import matplotlib.colors as colors
    # Make a figure and axes with dimensions as desired.
    fig = plt.figure(figsize=(3, 2))
    ax1 = fig.add_axes([0.05, 0.8, 0.9, 0.15])

    # Set the colormap and norm to correspond to the data for which
    # the colorbar will be used.
    cmap = color
    norm = mpl.colors.Normalize(vmin=min(gFun), vmax=max(gFun))

    # ColorbarBase derives from ScalarMappable and puts a colorbar
    # in a specified axes, so it has everything needed for a
    # standalone colorbar.  There are many more kwargs, but the
    # following gives a basic continuous colorbar with ticks
    # and labels.
    cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap,
                                    norm=norm,
                                    orientation='horizontal')
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(18)
    cb1.set_label('g (meV)', fontsize=22)
    plotName = '1005CrossingColor'+str1+'.png'
    plt.savefig(plotName)
    return plt.show()

