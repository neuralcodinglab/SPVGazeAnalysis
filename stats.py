import pandas as pd
import numpy as np
import scipy.stats


# Global Variables
GROUP = 'Subject'

# endpoints = ['ReportedSubjectiveRating', 'TrialDuration', 'CorrectResponse']

def average(data, y, group='Subject', x='GazeCondition', apply_function=None):
    """Averages the endpoint variables <y> (list of column names) over the grouping variable <group>."""
    avg = data.loc[:, [group,]+[x,] + y]
    if apply_function is None or apply_function=='mean':
        avg = avg.groupby([group, x]).mean()
    elif apply_function=='std':
        avg = avg.groupby([group, x]).std()
    else:
        avg = avg.groupby([group, x]).agg(apply_function)
    return avg.reset_index()

def paired_wilcoxon(data, group, endpoints=['ReportedSubjectiveRating', 'TrialDuration', 'CorrectResponse']):
    
    cond1, cond2, cond3 = HUE_ORDER
    pairs = [(cond1, cond2), (cond2, cond3), (cond1, cond3)]
    
    # Extract data per condition
    subdata = dict()
    for cond in [cond1, cond2, cond3]:
        subdata[cond] = data.loc[data.GazeCondition == cond].set_index(group) 
    
    results = dict()
    for y in endpoints:
        results[y] = dict()
        for (cond_a, cond_b) in pairs:
            intersection = subdata[cond_a].index.intersection(subdata[cond_b].index)
            dist_a = subdata[cond_a].loc[intersection].sort_index()[y]
            dist_b = subdata[cond_b].loc[intersection].sort_index()[y]
            _, p_value = scipy.stats.wilcoxon(dist_a, dist_b)
            results[y]['{}X{}'.format(cond_a,cond_b)] = p_value
    return pd.DataFrame(results)