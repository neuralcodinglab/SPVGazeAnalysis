"""
Module with visualization functions using pandas and seaborn

"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# local modules
import utils



## Default global variables

# Figurestyle
BASE_PATH = os.path.dirname(__file__)
FIGURESTYLE = f'{BASE_PATH}/_figurestyle/seaborn-v0_8.mplstyle'
COLORS = utils.load_yaml(f'{BASE_PATH}/_figurestyle/colors.yaml')

# Default order for using consistent colors for conditions
ORDERED_CONDITIONS = ['GazeIgnored', 'GazeAssistedSampling', 'SimulationFixedToGaze'] 

# Mapping to different label or scalar
COND_AS_SCALAR = {k:i for i,k in enumerate(ORDERED_CONDITIONS)} # {..: 0, ..: 1, ..: 2}
COND_AS_COLOR_LABEL = {k:f'C{i}' for i,k in enumerate(ORDERED_CONDITIONS)} # {..: 'C0', ..: 'C1', ..: 'C2'}
COND_REDEFINED = {'GazeIgnored': 'Gaze Ignored',
                  'SimulationFixedToGaze': 'Gaze Locked', 
                  'GazeAssistedSampling' : 'Gaze Contingent'} # Replace with names that are consistent with the paper

TITLE_REDEFINED = {} # TODO

PANEL_INDEX_SIZE = 20

FIGSIZE = (4,4)

def set_figurestyle(figurestyle=FIGURESTYLE, colors=COLORS):
#     sns.axes_style("darkgrid")
    plt.style.use(figurestyle)
    sns.set_context("paper") # OVERRIDES EXISTING STYLE PARAMS
    sns.set_palette(sns.color_palette(colors.values())) #, n_colors=len(colors), desat=0.1))


def create_subplots(n_figs=3, figsize=FIGSIZE):
    return plt.subplots(1,n_figs,figsize=(figsize[0]*n_figs,figsize[1]), dpi=100)

def violin_plots(data, endpoints, x='GazeCondition',
                 axs=None, fig=None,
                 order=ORDERED_CONDITIONS,
                 saturation=1, **kwargs):
    # Create axes
    if axs is None:
        fig, axs = create_subplots(len(endpoints))
    
    # Plot violins
    for i, y in enumerate(endpoints):
        sns.violinplot(data=data, x=x, y=y, ax=axs[i], order=order)
        axs[i].set(title=y)
    return fig, axs

def bar_plots(data, endpoints, x='GazeCondition',
              axs=None, fig=None,
              order=ORDERED_CONDITIONS,
              saturation=1, **kwargs):
    # Create axes
    if axs is None:
        fig, axs = create_subplots(len(endpoints))
    
    # Plot bars
    if len(endpoints) == 1:
        y = endpoints[0]
        sns.barplot(data=data, x=x, y=y, ax= axs, order=order, **kwargs)
        axs.set(title=y)
        return fig, axs
        
    for i, y in enumerate(endpoints):
        sns.barplot(data=data, x=x, y=y, ax= axs[i], order=order, **kwargs)
        axs[i].set(title=y)
    return fig, axs


def swarm_plots(data, endpoints, group = 'Subject',
                axs=None, fig=None,
                x = 'GazeCondition',
                scalar_mapping = COND_AS_SCALAR,
                color_mapping = COND_AS_COLOR_LABEL,
                jitter=0.2, alpha=0.3):
    
    # Replace categorical data x with scalar data x_
    x_ = data[x].replace(scalar_mapping)
    x_ += jitter * (np.random.rand(len(x_)) -.5) 
    x_ = x_.sort_values()
    sorted_data = data.loc[x_.index]
    
    
    # Create axes
    if axs is None:
        n_figs = len(endpoints)
        fig, axs = plt.subplots(1,n_figs,figsize=(6*n_figs,4), dpi=100)
    
    # Plot for each endpoint in a separate axis
    if len(endpoints) == 1:
        axs = [axs,]
    for i, y in enumerate(endpoints):
        axs[i].set(title=y)
        
        # Draw a line with markers for each instance in a group (e.g. for each subject)
        for category in data[group].unique():
            mask = sorted_data[group] == category
            colors = sorted_data.loc[mask,x].replace(color_mapping)
            axs[i].plot(x_.loc[mask], sorted_data.loc[mask,y], linestyle='-', color='gray', alpha=alpha )
            axs[i].scatter(x_.loc[mask], sorted_data.loc[mask,y], linestyle='-',color='k', linewidth=0, alpha=alpha)
        
    return fig, axs


def joint_distribution_plots(data, pairs, order=ORDERED_CONDITIONS, regression=False, despine_completely=True, reverse_order=True):
    
    """ 
    Plot multple jointplots of the data. 
    
    Arguments:
        data: pandas dataframe with the data
        pairs: list of label pairs (tuples) found in the data 
    
    """
    
    # Create figure and Gridspec
    N = len(pairs)
    fig = plt.figure(figsize=(5*N, 4), dpi=200)
    gs = fig.add_gridspec(2, 3*N,  width_ratios=(4, 1, 1)*N, height_ratios=(1, 4),
                          left=0.1, right=0.9, bottom=0.1, top=0.9,
                          wspace=0.05, hspace=0.05)
    handles = []

    # Enumerate pairs and create each jointplot 
    for i, (x,y) in enumerate(pairs):
        
        # The scatterplot
        ax0 = fig.add_subplot(gs[1, 0 + 3*i])
        ax0.spines[['right', 'top']].set_visible(False)
#         sns.scatterplot(data=data, x=x, y=y, hue='GazeCondition',ax=ax0, legend=False, alpha=0.3)
    
        if regression:
            for condition in order: # ['GazeIgnored', 'GazeAssistedSampling', 'SimulationFixedToGaze']
                h = sns.regplot(data=data.loc[data.GazeCondition==condition],x=x,y=y, scatter_kws={'alpha':0.25})
                handles.append(h)
        else:
            h = sns.scatterplot(data, x=x, y=y, alpha=0.6, hue='GazeCondition', hue_order=order, legend=False)
            handles.append(h)

        # The distribution plots
        ax1 = fig.add_subplot(gs[0, 0 + 3*i], sharex=ax0)
        ax2 = fig.add_subplot(gs[1, 1 + 3*i], sharey=ax0)
        
        
        if despine_completely:
            ax1.axis('off')
            ax2.axis('off')
        else:
            ax1.spines[['right', 'top', 'left']].set_visible(False)
            ax2.spines[['bottom', 'top', 'right']].set_visible(False)
            plt.setp(ax1.get_xticklabels() + ax2.get_yticklabels() + \
                    [ax1.yaxis,  ax2.xaxis], visible=False)

        # Reverse color order for better visibility
        palette=None
        if reverse_order:
            order = order[::-1]
            palette = sns.color_palette((plt.rcParams['axes.prop_cycle'].by_key()['color'][::-1]))

        h1 = sns.kdeplot(data=data, x=x, hue='GazeCondition',ax=ax1, legend=False, fill=True, hue_order=order, palette=palette)
        h2 = sns.kdeplot(data=data, y=y, hue='GazeCondition',ax=ax2, legend=False, fill=True, hue_order=order, palette=palette)
        handles += [h1,h2,]
    return handles


def redefine_x_ticks(axs, mapping=COND_REDEFINED, remove_xlabel=False):
    
    # recursive loop through all axes
    if type(axs) == np.ndarray:
        for ax in axs:
            redefine_x_ticks(ax, mapping, remove_xlabel)
        return
    
    old_ticks = axs.get_xticklabels()
    new_ticks = [mapping[t.get_text()] for t in old_ticks]
    axs.set_xticklabels(new_ticks)
    if remove_xlabel:
        axs.set(xlabel='')


def add_significance_line(ax, x1, x2, y=None, text='', rel_h=0.02, rel_y=0.9, size=20):
    
    # Compute line height and height of vertical 'line ends'
    y_min, y_max = ax.get_ylim()
    h = (y_max-y_min) * rel_h
    y = rel_y * y_max if y is None else y
    
    # Draw line
    y_ = [y-h, y, y, y-h]
    x_ = [x1,x1, x2, x2]
    ax.plot(x_, y_,'k')
    
    # Asteriks above line
    ax.text((x2+x1)/2, y+(h*2), text,
        horizontalalignment='center',
        verticalalignment='center',
        size=size,)

def increase_ylim(ax, rel_y=1.25):
    y_min, y_max = ax.get_ylim()
    ax.set_ylim([y_min, rel_y*y_max])
    
def add_significance_lines(ax, text,
                           x_pairs=[(0, .95),  (1.05, 2), (0, 2)],
                           rel_y=[0.82, 0.82, 0.9],
                           **kwargs):
    """ Add multiple significance lines to ax. """
    increase_ylim(ax, rel_y=1.25)
    for j, (x1, x2) in enumerate(x_pairs) :
        add_significance_line(ax, x1, x2, text=text[j], rel_y=rel_y[j], **kwargs)
        
        
def add_panel_index(ax, text, rel_x=-0.05, rel_y=1.05, size=PANEL_INDEX_SIZE):
    y_min, y_max = ax.get_ylim()
    x_min, x_max = ax.get_xlim()
    y = (y_max-y_min)*rel_y + y_min
    x = (x_max-x_min)*rel_x + x_min
    ax.text(x,y, text, size=size, weight='bold')