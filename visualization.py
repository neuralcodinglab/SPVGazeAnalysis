"""
Module with visualization functions using pandas and seaborn

"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib
from matplotlib.patches import Patch

# local modules
import utils
import env_params
from env_params import HALLWAY_DIMS as hw_dims
from env_params import Boxes
from env_params import START_POS_X as hw_x_offset
from env_params import get_hallway_layouts, Boxes

from matplotlib.collections import PatchCollection
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import matplotlib.patheffects as PathEffects




## Default global variables

# Figurestyle
BASE_PATH = os.path.dirname(__file__)
FIGURESTYLE = f'{BASE_PATH}/_figurestyle/seaborn-like.mplstyle' # f'{BASE_PATH}/_figurestyle/seaborn-v0_8.mplstyle'
COLORS = utils.load_yaml(f'{BASE_PATH}/_figurestyle/colors.yaml')
COLORMAP = mcolors.ListedColormap(COLORS.values())

# Default order for using consistent colors for conditions
ORDERED_CONDITIONS = ['SimulationFixedToGaze', 'GazeAssistedSampling', 'GazeIgnored',] 

# Mapping to different label or scalar
COND_AS_SCALAR = {k:i for i,k in enumerate(ORDERED_CONDITIONS)} # {..: 0, ..: 1, ..: 2}
COND_AS_COLOR_LABEL = {k:f'C{i}' for i,k in enumerate(ORDERED_CONDITIONS)} # {..: 'C0', ..: 'C1', ..: 'C2'}
COND_REDEFINED = {'SimulationFixedToGaze': 'Gaze Locked',
                  'GazeAssistedSampling' : 'Gaze Contingent',
                  'GazeIgnored': 'Gaze Ignored', } # For replacing names, consistent with the paper

PANEL_INDEX_SIZE = 10 #20

FIGSIZE = (3.5,1.5)

def set_figurestyle(figurestyle=FIGURESTYLE, colors=COLORS):
#     sns.axes_style("darkgrid")
    sns.set_context("paper") # OVERRIDES EXISTING STYLE PARAMS
    plt.style.use(figurestyle)
    sns.set_palette(sns.color_palette(colors.values())) #, n_colors=len(colors), desat=0.1))



def create_subplots(n_figs=3, figsize=FIGSIZE):
    fig = plt.figure(figsize=figsize, dpi=300)
    axs = []
    gs = matplotlib.gridspec.GridSpec(1, n_figs, wspace=0.35, hspace=0)
    for g in gs:
        axs.append(fig.add_subplot(g))
    axs = axs[0] if n_figs==1 else np.array(axs)
    return fig, axs

def plot_legend(figsize=FIGSIZE, size=50, fig=None, ax=None):
    handles = [Patch(facecolor=c, edgecolor=c) for c in COLORS.values()]
    labels = COND_REDEFINED.values()
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    ax.axis(False)
    ax.legend(handles, labels, loc="center", bbox_to_anchor=(0.5, 0.5), prop={"size":size})
    return fig,ax

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

def regression_plots(data, endpoints, x='Block', hue='GazeCondition',
                     hue_order=ORDERED_CONDITIONS, axs=None, fig=None, **kwargs):
    
    # Create axes
    if axs is None:
        fig, axs = create_subplots(len(endpoints))
    
    # recursive loop through all axes
    if type(axs) == np.ndarray:
        for i, ax in enumerate(axs):
            regression_plots(data, endpoints[i], x, hue, hue_order, axs=ax, fig=fig, **kwargs)
            ax.set(title=endpoints[i])
        return fig, axs

    if (type(endpoints) is list) and (len(endpoints)==1):
        y=endpoints[0]
    else:
        y=endpoints
  
    for h in hue_order:
        subset = data.loc[data[hue]==h]
        sns.regplot(data=subset, x=x, y=y, ax=axs, **kwargs)
    return fig, axs
    

def swarm_plots(data, endpoints, group = 'Subject',
                axs=None, fig=None,
                x = 'GazeCondition',
                scalar_mapping = COND_AS_SCALAR,
                linecolor = 'gray',
                markercolor = 'gray',
#                 color_mapping = COND_AS_COLOR_LABEL,
                
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
            axs[i].plot(x_.loc[mask], sorted_data.loc[mask,y], linestyle='-', color=linecolor, alpha=alpha )
            axs[i].scatter(x_.loc[mask], sorted_data.loc[mask,y], linestyle='-',color=markercolor, linewidth=0, alpha=alpha)

        
    return fig, axs

def plot_single_gaze_trajectory(data, as_line=True, tmax=None, fig=None, ax=None, title=None, colorbar=False):
    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=(4,4), dpi=100)
    
    
    # Get valid x and y coordinates
    valid = data.Validity != 0
    x = np.arctan(data.loc[valid, 'GazeDirectionNormInEyeX'] / data.loc[valid, 'GazeDirectionNormInEyeZ']) *180/np.pi
    y = np.arctan(data.loc[valid, 'GazeDirectionNormInEyeY'] / data.loc[valid, 'GazeDirectionNormInEyeZ']) *180/np.pi
    
    # Color
    t = data.loc[valid].SecondsSinceTrialStart
    if tmax is None:
        tmax = t.max()
    norm = plt.Normalize(t.min(), tmax)
    
    # Plot colored line or scatter
    if as_line:
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = matplotlib.collections.LineCollection(segments, cmap='viridis', norm=norm)
        lc.set_array(t)
        lc.set_linewidth(1)
        handle = ax.add_collection(lc)
    else:
        handle = ax.scatter(x, y, alpha = 0.05, c=t, norm=norm, cmap='viridis') #color='#9e1111')
    
    if colorbar:
        fig.colorbar(handle,ax=ax)
    ax.axis('square')
    ax.set(xlim = [-50,50],
           ylim = [-50,50], title=title)
    return fig, ax


def plot_gaze_trajectories(data, trials_of_interest, tmax=90, as_line=False, fig=None, axs=None, figsize=FIGSIZE):
    ny, nx = trials_of_interest.shape
#     fig, axs = plt.subplots(ny,nx,figsize=(4*nx,4*ny))
    if axs is None:
        fig, axs = plt.subplots(ny,nx,figsize=(figsize[0]*nx,figsize[1]*ny))

    for i,t in enumerate(trials_of_interest.flatten()):
        if i >= len(axs.flatten()): 
            break
        trial_data = data.loc[data.TrialIdentifier == t]
        duration = trial_data.TrialDuration.iloc[0]
        trial_data = trial_data.loc[trial_data.SecondsSinceTrialStart < trial_data.TrialDuration]
        condition = trial_data.GazeCondition.iloc[0]
        lbl = COND_REDEFINED[condition]
        plot_single_gaze_trajectory(trial_data, as_line=as_line, tmax=tmax,
                             title=f'Trial {t} ({duration:.1f}s) \n{lbl}', fig=fig, ax=axs.flatten()[i])
        axs.flatten()[i].set(xlabel='Azimuth (Deg)', ylabel='Elevation (Deg)')
    plt.tight_layout()
    return fig, axs




def plot_hallway(hallway, ax):

    # Get hallway layouts
    hallways = env_params.get_hallway_layouts()
    segmentLength = hw_dims['segmentLength']
    hwWidth = hw_dims['hwWidth']
    hwLength = len(hallways)*segmentLength
    hw_names = ['Hallway1', 'Hallway2', 'Hallway3']
    
    # Get box sizes
    smBox = hw_dims['smBox']
    lgBox = hw_dims['lgBox']
   
#     ax.set_axis_off()
    ax.set(xticks=np.linspace(0,hwLength,45), yticks=[hwWidth/2], 
          xticklabels=[], yticklabels=[])
    ax.tick_params(width=0)
    
    ax.set_xlim((0, hwLength))
    ax.set_ylim((0, hwWidth))


    
    # background
#     rect = Rectangle((0, 0), hwLength, hwWidth, edgecolor='none', facecolor='xkcd:ivory', zorder=0)
    rect = Rectangle((0,0),hwLength, hwWidth, facecolor='none', edgecolor='k', linewidth=5, zorder=10)
    # Add the patch to the Axes
    ax.add_patch(rect)
    
    boxPatches = []
    roomEnd=segmentLength
    for box in hallway:
        anchor = None
        if box.value == Boxes.SmallL.value:
            dY,dX,_ = smBox
            anchor = (roomEnd - dX, hwWidth-dY)
        elif box.value == Boxes.SmallC.value:
            dY,dX,_ = smBox
            anchor = (roomEnd - dX, (hwWidth-dY) / 2)
        elif box.value == Boxes.SmallR.value:
            dY,dX,_ = smBox
            anchor = (roomEnd - dX, 0)
        elif box.value == Boxes.LargLC.value:
            dY,dX,_ = lgBox
            dY *= 2 # *2 because it's 2 boxes next to each other
            anchor = (roomEnd - dX, hwWidth-dY) 
        elif box.value == Boxes.LargCR.value:
            dY,dX,_ = lgBox
            dY *= 2 # *2 because it's 2 boxes next to each other
            anchor = (roomEnd - dX, 0)
        elif box.value == Boxes.LargLR.value:
            dY,dX,_ = lgBox
            anchor = (roomEnd - dX, hwWidth - dY)
            boxPatches.append(box_patch(anchor, dX, dY))
            anchor = (roomEnd - dX, 0)
            boxPatches.append(box_patch(anchor, dX, dY))
            roomEnd += segmentLength
            continue
            
        if anchor is not None:
            boxPatches.append(box_patch(anchor, dX, dY))
        roomEnd += segmentLength
    
#     pc = PatchCollection(boxPatches, edgecolor='none',
#                  facecolor='xkcd:chocolate', zorder=10)
    pc = PatchCollection(boxPatches, edgecolor='k',
             facecolor=(0.6,0.6,0.6), zorder=5)
    ax.add_collection(pc)
    
#     roomEnd = segmentLength
#     dividers = []
#     while roomEnd < hwLength:
#         dividers.append(Rectangle((roomEnd, 0), .1*segmentLength, hwWidth))
#         roomEnd += segmentLength
#     pc = PatchCollection(dividers, edgecolor='none', 
#                          facecolor= 'xkcd:grey', alpha=.6, zorder=5)
    ax.axvline(5, color='k')
    ax.axvline(42, color='k')

    
    ax.add_collection(pc)
                        
def box_patch(anchor, dX, dY):
    return Rectangle(anchor, dX, dY)

def plot_mobility_trajectories(data, trials, fig=None, axs=None, hue=None, cmap=None, hue_order=None):
    hallways = env_params.get_hallway_layouts()
    
    # Make subplot
    if axs is None:
        hwLength = hw_dims['segmentLength'] * len(hallways)
        hwWidth = hw_dims['hwWidth']
        fig, axs = plt.subplots(3,1, figsize=(hwLength/2,3.1 * hwWidth/2), dpi=300)

    if cmap is None:
        cmap = plt.colormaps['viridis']
        
    # Lookup which path in which plot 
    plot_idx = dict()
    for idx, hw_name in enumerate(hallways):
        plot_hallway(hallways[hw_name],axs[idx])
        plot_idx[hw_name] = idx

    for trial in trials:
        d = data.loc[data.TrialIdentifier == trial]
        d_ = d.loc[d.InsideTrial]
        
        # Which ax (which hallway)?
        hw_name = d.Hallway.iloc[0]
        idx = plot_idx[hw_name]
        ax = axs[idx]
        
        # Label
        subject = d.Subject.iloc[0]
        condition = d.GazeCondition.iloc[0] 
        condition = COND_REDEFINED[condition] # Rename for consistency with paper 
        label = f'{condition} ({subject}) '

        if hue is None:
            h = ax.plot(d.loc[~d.InsideTrial].x,d.loc[~d.InsideTrial].y, '--', alpha=0.5, linewidth=1.5) # Plot entire recording
            ax.plot(d_.x,d_.y, label=label, color=h[0].get_color()) # Plot trial data

        else: 
            if pd.api.types.is_numeric_dtype(d[hue]):
                v = data[hue].replace([np.inf, -np.inf], np.nan).dropna()
                c = (d_[hue]-v.min())/(v.max()-v.min())
                color = cmap(c)
                ax.scatter(d_.x,d_.y, label=label, color=color) # Plot entire recording
            else:
                cat = data[hue].unique()
                if hue_order is None:
                    hue_order = cat
                N = len(cat) 
                c = pd.DataFrame(cmap(np.linspace(0,1,N)), index=hue_order, columns=[*'rgba'])
                color = c.loc[d[hue]].iloc[0].values
                h = ax.plot(d.loc[d.FinishedTrial].x,d.loc[d.FinishedTrial].y, '--',
                            alpha=0.5, linewidth=1.5, color=color,) # Plot entire recording
                h = ax.plot(d.loc[~d.StartedTrial].x,d.loc[~d.StartedTrial].y, '--',
                            alpha=0.5, linewidth=1.5, color=color,) # Plot entire recording
                ax.plot(d_.x,d_.y, label=label, color=color, linewidth=2, alpha=0.6) # Plot trial data



            
        # Plot start, finish and colisions
        if d_.empty:
            continue
        ax.scatter(d_.iloc[0].x,d_.iloc[0].y, marker='.', color ='k')
        collision_mask = d.Collision # or d.FrontalCollision
        cols = d.loc[collision_mask].groupby(['ClosestBoxZone'])[['x', 'y']].first()
        ax.scatter(cols.x,cols.y, marker='o', s=50, facecolor=(0,0,0,0), edgecolor=(1,0,0), linewidth=2)
    
    for i, ax in enumerate(axs):
        legend = ax.legend(loc='center right', bbox_to_anchor=(0, 0.5),
                          title=f'       Obstacle Layout {i+1}',
                          edgecolor = 'grey',
                          # frameon=True,
                          facecolor='white',
                          # fontsize='medium',
                          title_fontproperties = {'weight':'bold','size':'small'},
                          alignment = 'left'
                 
                           
                          )
        # legend.get_frame().set_linewidth(1.0)

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


def redefine_x_ticks(axs, mapping=COND_REDEFINED,
                     remove_xlabel=False,
                     new_line=False,):
    
    # recursive loop through all axes
    if type(axs) == np.ndarray:
        for ax in axs:
            redefine_x_ticks(ax, mapping, remove_xlabel, new_line)
        return
    
    old_ticks = axs.get_xticklabels()
    new_ticks = [mapping[t.get_text()] for t in old_ticks]

    if new_line:
        new_ticks = [t.replace(' ', '\n') for t in new_ticks]
    
    axs.set_xticklabels(new_ticks)
    if remove_xlabel:
        axs.set(xlabel='')

def redefine_legend_labels(axs, mapping=COND_REDEFINED):
    
    # recursive loop through all axes
    if type(axs) == np.ndarray:
        for ax in axs:
            redefine_legend_labels(ax, mapping)
        return
    
    handles, labels = axs.get_legend_handles_labels()
    new_labels = [mapping[lbl] for lbl in labels]
    axs.legend(handles=handles, labels=new_labels)


def add_significance_line(ax, x1, x2, y=None, text='', rel_h=0.015, rel_y=0.9, size=13):
    
    # Compute line height and height of vertical 'line ends'
    y_min, y_max = ax.get_ylim()
    h = (y_max-y_min) * rel_h
    y = rel_y * y_max if y is None else y
    
    # Draw line
    y_ = [y-h, y, y, y-h]
    x_ = [x1,x1, x2, x2]
    ax.plot(x_, y_,'k')
    
    # Asteriks above line
    ax.text((x2+x1)/2, y-(h/2), text,
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
        if text[j]:
            add_significance_line(ax, x1, x2, text=text[j], rel_y=rel_y[j], **kwargs)
        
        
def add_panel_index(ax, text, rel_x=-0.15, rel_y=1.07, size=PANEL_INDEX_SIZE):
    y_min, y_max = ax.get_ylim()
    x_min, x_max = ax.get_xlim()
    y = (y_max-y_min)*rel_y + y_min
    x = (x_max-x_min)*rel_x + x_min
    ax.text(x,y, text, size=size, weight='bold')