from pathlib import Path
import matplotlib.pyplot as plt
from enum import Enum

import numpy as np

def plot_background(locationId,ax=None):
    p = ... # Path('path to backgrounds') # config
    
    if locationId == 1:
        orthoPxToMeter = 0.00814636091724916
    elif locationId ==  2:
        orthoPxToMeter = 0.00814636091724502
    elif locationId == 3:
        orthoPxToMeter = 0.00814635379575616
    elif locationId == 4:
        orthoPxToMeter = 0.0126999352667008
    p = p / f'background_location{locationId}.png'
    
    background = plt.imread(p)
    tmp =  tuple(i *orthoPxToMeter*12 for i in background.shape)
    extent = (0,tmp[1] , -tmp[0], 0) # (left, right, bottom, top),
    if ax is not None:
        ax.imshow(background,extent=extent)
    else:
        plt.imshow(background,extent=extent)

def set_plot_limits(locationId,ax=None): 
    xlims = [[20,87],[7,99],[5,90],[47,187]]
    #ylims = [[-70,5],[-57,0],[-75,0],[-118,0]]
    ylims = [[-70,5],[-49,0],[-75,0],[-118,0]]
    if ax is not None:
        ax.set_xlim(xlims[locationId-1])
        ax.set_ylim(ylims[locationId-1])
    else:
        plt.xlim(xlims[locationId-1])
        plt.ylim(ylims[locationId-1])
        
def fig_size(width, fraction=1, subplots=(1, 1)): # https://jwalton.info/Embed-Publication-Matplotlib-Latex/
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == 'thesis':
        width_pt = 426.79135
    elif width == 'beamer':
        width_pt = 307.28987
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)

class LatexWidth(Enum): 
    """
    Text widths in pt units.
    """
    IEEE_JOURNAL_COLUMN = 252 
    IEEE_CONFERENCE_COLUMN = 245.71811
    IEEE_JOURNAL_TEXT = 516

def despine(ax):
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

