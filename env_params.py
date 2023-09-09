from enum import Enum
import pandas as pd
import numpy as np


class Boxes(Enum):
    NoBox = 0
    SmallL = 10
    SmallC = 11
    SmallR = 12
    LargLC = 20
    LargCR = 21
    LargLR = 22
    

def get_hallway_layouts(prepend=[Boxes.NoBox, Boxes.NoBox], append=[Boxes.NoBox]):
    return pd.DataFrame({'Hallway1': prepend + HALLWAY1 + append,
                         'Hallway2': prepend + HALLWAY2 + append,
                         'Hallway3': prepend + HALLWAY3 + append})   

def get_box_bb(box:Boxes, segmentEnd:float=None):
    hwWidth = HALLWAY_DIMS['hwWidth']
    smBox =  HALLWAY_DIMS['smBox']
    lgBox =  HALLWAY_DIMS['lgBox']
    if segmentEnd is None:
        segmentEnd = HALLWAY_DIMS['segmentLength']
    if box.value == Boxes.SmallL.value:
        dY,dX,_ = smBox
        ll = [segmentEnd-dX, hwWidth-dY], None
    elif box.value == Boxes.SmallC.value:
        dY,dX,_ = smBox
        ll = [segmentEnd-dX, (hwWidth-dY) / 2], None
    elif box.value == Boxes.SmallR.value:
        dY,dX,_ =smBox
        ll = [segmentEnd-dX, 0], None
    elif box.value == Boxes.LargLC.value:
        dY,dX,_ = lgBox
        dY *= 2 # *2 because it's 2 boxes next to each other
        ll = [segmentEnd-dX, hwWidth-dY], None
    elif box.value == Boxes.LargCR.value:
        dY,dX,_ = lgBox
        dY *= 2 # *2 because it's 2 boxes next to each other
        ll = [segmentEnd-dX, 0], None
    elif box.value == Boxes.LargLR.value:
        dY,dX,_ = lgBox
        ll = [segmentEnd-dX, hwWidth - dY], [segmentEnd-dX, 0]
    else:
        return None, None

    ll = np.array(ll[0]), np.array(ll[1])
    lr = ll[0] + [dX, 0], (ll[1] + [dX, 0] if ll[1].shape != () else ll[1])
    ul = ll[0] + [0, dY], (ll[1] + [0, dY] if ll[1].shape != () else ll[1])
    ur = ul[0] + [dX, 0], (ul[1] + [dX, 0] if ul[1].shape != () else ll[1])

    return (ll[0], lr[0], ur[0], ul[0]), (ll[1], lr[1], ur[1], ul[1])


HALLWAY_DIMS = {'smBox': [.6, .3, .9], # w x d x h
                'lgBox': [.95, .3, 1.8],
                'pRadius': .225,
                'hwWidth': 2.85,
                'segmentLength': 2,
                'startLine': 5, 
                'finishLine': 42}

START_POS_X = {'Hallway1': 0, # Unity coordinates (sideways from player perspective)
               'Hallway2': 4,
               'Hallway3': 8}

START_POS_Z = {'Hallway1': 3, # Unity coordinates (forward from player perspective)
               'Hallway2': 3,
               'Hallway3': 3}

HALLWAY1 = [Boxes.SmallC,
            Boxes.SmallC,
            Boxes.LargCR,
            Boxes.SmallL,
            Boxes.SmallR,
            Boxes.SmallC,
            Boxes.LargLR,
            Boxes.SmallL,
            Boxes.SmallL,
            Boxes.LargLC, # transition to 5
            Boxes.SmallC,
            Boxes.SmallR,
            Boxes.LargLR,
            Boxes.SmallC,
            Boxes.SmallL,
            Boxes.SmallR,
            Boxes.LargCR,
            Boxes.SmallC,
            Boxes.SmallC ]

HALLWAY2 = [Boxes.SmallL,
            Boxes.SmallC,
            Boxes.LargCR,
            Boxes.SmallL,
            Boxes.SmallL,
            Boxes.SmallR,
            Boxes.LargLC,
            Boxes.SmallR,
            Boxes.SmallL,
            Boxes.LargCR, # transition to 3
            Boxes.SmallR,
            Boxes.SmallL,
            Boxes.LargLR,
            Boxes.SmallC,
            Boxes.SmallL,
            Boxes.SmallC,
            Boxes.LargCR,
            Boxes.SmallC,
            Boxes.SmallL ]

HALLWAY3 = [Boxes.SmallL,
            Boxes.SmallC,
            Boxes.LargCR,
            Boxes.SmallR,
            Boxes.SmallL,
            Boxes.SmallC,
            Boxes.LargLR,
            Boxes.SmallL,
            Boxes.SmallC,
            Boxes.LargLC, # transition to 7
            Boxes.SmallR,
            Boxes.SmallR,
            Boxes.LargLC,
            Boxes.SmallC,
            Boxes.SmallR,
            Boxes.SmallL,
            Boxes.LargCR,
            Boxes.SmallC,
            Boxes.SmallL]

BOX_LOCATIONS = {Boxes.NoBox: (np.nan*np.ones((4,2)), np.nan*np.ones((4,2))),
                 Boxes.SmallL: (np.array([[1.7 , 2.25],
                                          [2.  , 2.25],
                                          [2.  , 2.85],
                                          [1.7 , 2.85]]), 
                                np.nan*np.ones((4,2))),
                 Boxes.SmallC: (np.array([[1.7  , 1.125],
                                          [2.   , 1.125],
                                          [2.   , 1.725],
                                          [1.7  , 1.725]]), 
                                np.nan*np.ones((4,2))),
                 Boxes.SmallR: (np.array([[1.7, 0. ], 
                                           [2., 0.],
                                           [2. , 0.6],
                                           [1.7, 0.6]]), 
                                np.nan*np.ones((4,2))),
                 Boxes.LargLC: (np.array([[1.7 , 0.95],
                                          [2.  , 0.95],
                                          [2.  , 2.85],
                                          [1.7 , 2.85]]),
                                np.nan*np.ones((4,2))),
                 Boxes.LargCR: (np.array([[1.7, 0. ],
                                           [2., 0.],
                                           [2. , 1.9],
                                           [1.7, 1.9]]), 
                                np.nan*np.ones((4,2))),
                 Boxes.LargLR: (np.array([[1.7, 1.9],
                                          [2. , 1.9],
                                          [2. , 2.85],
                                          [1.7 , 2.85]]),
                                np.array([[1.7 , 0.],
                                          [2., 0.],
                                          [2.  , 0.95],
                                          [1.7 , 0.95]]))}


#### Map drawing:

def map_hallways():
    length = (4 + len(hallway1)) * segmentLength
    fig, axs = plt.subplots(3,1,figsize=(length, hwWidth*3.3))
    for idx,(ax, hallway) in enumerate(zip(axs, (hallway1, hallway2, hallway3))):
        create_map(hallway, ax)
        ax.text(-segmentLength, hwWidth/2, 
                f"HW-{idx}", fontsize=56, 
                va='center', ha='left')
        
    return fig, axs

def create_map(hallway, ax):
    # total length is 3 empty rooms + no. of obstacles
    length = (3 + len(hallway)) * segmentLength
    
    ax.set_axis_off()
    ax.set_xlim((0, length))
    ax.set_ylim((0, hwWidth))
    # background
    rect = Rectangle((0, 0), length, hwWidth, 
                     edgecolor='none', facecolor='xkcd:ivory', zorder=0)
    # Add the patch to the Axes
    ax.add_patch(rect)
    
    # the hallway obstacles
    # since first 2 rooms are empty, the actual first room starts at segmentLength*2
    # obstacles are placed at the end of a segment so the box starts at
    # segmentLength - boxDepth in relation to the room start
    boxPatches = []
    roomEnd = 3 * segmentLength
    for box in hallway:
        if box == Boxes.SmallL:
            dY,dX,_ = smBox
            anchor = (roomEnd - dX, hwWidth-dY)
        elif box == Boxes.SmallC:
            dY,dX,_ = smBox
            anchor = (roomEnd - dX, (hwWidth-dY) / 2)
        elif box == Boxes.SmallR:
            dY,dX,_ = smBox
            anchor = (roomEnd - dX, 0)
        elif box == Boxes.LargLC:
            dY,dX,_ = lgBox
            dY *= 2 # *2 because it's 2 boxes next to each other
            anchor = (roomEnd - dX, hwWidth-dY) 
        elif box == Boxes.LargCR:
            dY,dX,_ = lgBox
            dY *= 2 # *2 because it's 2 boxes next to each other
            anchor = (roomEnd - dX, 0)
        elif box == Boxes.LargLR:
            dY,dX,_ = lgBox
            anchor = (roomEnd - dX, hwWidth - dY)
            boxPatches.append(box_patch(anchor, dX, dY))
            anchor = (roomEnd - dX, 0)
            boxPatches.append(box_patch(anchor, dX, dY))
            
            roomEnd += segmentLength
            continue
        
        boxPatches.append(box_patch(anchor, dX, dY))
        roomEnd += segmentLength
    
    pc = PatchCollection(boxPatches, edgecolor='none',
                 facecolor='xkcd:chocolate', zorder=10)
    ax.add_collection(pc)
    
    roomEnd = segmentLength
    dividers = []
    while roomEnd < length:
        dividers.append(Rectangle((roomEnd, 0), .1*segmentLength, hwWidth))
        roomEnd += segmentLength
    
    pc = PatchCollection(dividers, edgecolor='none', 
                         facecolor= 'xkcd:grey', alpha=.6, zorder=5)
    ax.add_collection(pc)
                        
def box_patch(anchor, dX, dY):
    return Rectangle(anchor, dX, dY)