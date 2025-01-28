import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLineCollection
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection


# ==================================
# === for rates and theory =========

def harmonic_number(t):
    if t == 0:
        return 0
    else:
        return (1/np.arange(1, t+1)).sum()


# ==================================
# === for plotting =================

# default figsuree sizes for different layouts

FIGSIZE11 = (4,3)
FIGSIZE12 = (8,3)


def set_plot_aesthetics():
    plt.rcParams["font.family"] = "serif"
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rc('text', usetex=True)
    plt.rc('legend',fontsize=10) 


class HandlerDashedLines(HandlerLineCollection):
    """
    Copied from https://matplotlib.org/stable/gallery/text_labels_and_annotations/legend_demo.html
    Custom Handler for LineCollection instances.
    """
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        # figure out how many lines there are
        numlines = len(orig_handle.get_segments())
        xdata, xdata_marker = self.get_xdata(legend, xdescent, ydescent,
                                             width, height, fontsize)
        leglines = []
        # divide the vertical space where the lines will go
        # into equal parts based on the number of lines
        ydata = np.full_like(xdata, height / (numlines + 1))
        # for each line, create the line at the proper location
        # and set the dash pattern
        for i in range(numlines):
            legline = Line2D(xdata, ydata * (numlines - i) - ydescent)
            self.update_prop(legline, orig_handle, legend)
            # set color, dash pattern, and linewidth to that
            # of the lines in linecollection
            try:
                color = orig_handle.get_colors()[i]
            except IndexError:
                color = orig_handle.get_colors()[0]
            try:
                dashes = orig_handle.get_dashes()[i]
            except IndexError:
                dashes = orig_handle.get_dashes()[0]
            try:
                lw = orig_handle.get_linewidths()[i]
            except IndexError:
                lw = orig_handle.get_linewidths()[0]
            if dashes[1] is not None:
                legline.set_dashes(dashes[1])
            legline.set_color(color)
            legline.set_transform(trans)
            legline.set_linewidth(lw)
            leglines.append(legline)
        return leglines


def do_fancy_legend(ax, 
                    labels, 
                    color_list,
                    loc='upper right',
                    lw=1.0,
                    handlelength=2.5,
                    handleheight=3,
                    **kwargs
    ):
    handles = list()
    for _colors in color_list:
        line = [[(0, 0)]]
        handles.append(LineCollection(len(_colors)*line, colors=_colors, linewidths=lw))
        
    
    ax.legend(handles, 
              labels,
              loc=loc, 
              handler_map={type(handles[0]): HandlerDashedLines()},
              handlelength=handlelength, 
              handleheight=handleheight,
              **kwargs
    )

    return