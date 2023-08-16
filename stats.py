import pandas as pd
import numpy as np
import scipy.stats
import visualization

# Global Variables
GROUP = 'Subject'
ORDERED_CONDITIONS = ['GazeIgnored', 'GazeAssistedSampling', 'SimulationFixedToGaze'] 
CORRECTED_ALPHA = 0.05/3 # Significance level after correction for multiple comparisons

# endpoints = ['ReportedSubjectiveRating', 'TrialDuration', 'CorrectResponse']

def average(data, y, group='Subject', x='GazeCondition', apply_function=None):
    """Averages the endpoint variables <y> (list of column names) over the grouping variable <group>."""
    y = y if type(y) is list else [y,]
    x = x if type(x) is list else [x,]
    group = group if type(group) is list else [group,]
    
    avg = data.loc[:, group + x + y]
    if apply_function is None or apply_function=='mean':
        avg = avg.groupby(group + x).mean()
    elif apply_function=='std':
        avg = avg.groupby(group + x).std()
    elif apply_function=='sum':
        avg = avg.groupby(group + x).sum()
    else:
        avg = avg.groupby(group + x).agg(apply_function)
    return avg.reset_index()

def normality_test(data, endpoints, x='GazeCondition', test='Shapiro-Wilk', alpha=0.05):
    
    # Which normality test to use
    if test == 'Shapiro-Wilk':
        func = scipy.stats.shapiro
    elif test == 'DAgostino-Pearson':
        func = scipy.stats.normaltest
    else:
        raise NotImplementedError
    
    # Perform test for each endpoint for each condition
    results = dict()
    for y in endpoints:
        results[y] = dict()
        for cond in data[x].unique():
            distribution = data.loc[data[x] == cond, y]
            _, p_value = func(distribution)
            results[y][cond] = p_value
            
    results = pd.DataFrame(results, dtype='object')
    results.loc['AnyNonNormal'] = (results < alpha).any()
    return results

def binom_test(n_correct, n_total, p=0.25):
    """Takes single-column dataframes <n_correct> and <n_total> for the
    (correct) number of trials, and applies binomial test on the rows.
    note: <n_correct> and <n_total> should have the same index"""
    test_func = lambda x: scipy.stats.binom_test(x.values,
                                                 n=n_total.loc[x.name][0],
                                                 p=p)
    return n_correct.apply(test_func, axis=1)

            
def highlight_significant(p_value, alpha=0.05):
    return 'font-weight: bold' if p_value < alpha else ''

def style(df, alpha=CORRECTED_ALPHA):
    return df.style.applymap(highlight_significant, alpha=alpha)

def count_significance_stars(p_value, alpha=CORRECTED_ALPHA):
    if p_value > alpha:
        return 'n.s.'
    if p_value > alpha/5:
        return '*'
    if p_value > alpha /50:
        return '**'
    return '***'
#     if p_value > alpha /500:
#         return '***'
#     return '****'

def paired_test(data, group, endpoints, relabel_conditions=True, test='Wilcoxon'):
    """Do a paired wilcoxon signed rank test between different gaze conditions,
    for several <endpoints>, where data is paired over <group> (e.g. subject)."""   
    
    if test == 'Wilcoxon':
        stats_func = lambda x1,x2: scipy.stats.wilcoxon(x1,x2, zero_method='pratt')
    elif test == 't-test':
        stats_func = scipy.stats.ttest_rel
    else:
        raise NotImplementedError
    
    # Load the conditions and renamed condition labels 
    conditions = [*visualization.COND_REDEFINED.keys()]
    labels = [*visualization.COND_REDEFINED.values()]
    
    # Create pairs of conditions, for the comparisons
    pairs = [(0,1), (1,2), (0,2)]
    paired_conditions = [(conditions[a], conditions[b]) for a,b in pairs]
    paired_labels = [(labels[a], labels[b]) for a,b in pairs]
    
    # Loop over all pairs and perform test
    results = {y:[] for y in endpoints}
    for (cond_a, cond_b) in paired_conditions:
        x1 = data.loc[data.GazeCondition == cond_a].set_index(group) 
        x2 = data.loc[data.GazeCondition == cond_b].set_index(group) 
        n1, n2 = len(x1.index), len(x2.index)
        
        # Check if all datapoints (e.g. subjects) are represented in both of the experimental conditions
        intersection = x1.index.intersection(x2.index)
        if (n1!=n2) or  len(intersection) < n1 or len(intersection) < n2:
            print(f"WARNING: not all '{group}' in '{cond_a}' are in '{cond_b}' or vice-versa" + 
                  f"results are computed on the intersection of '{group}' (N={len(intersection)})")
            x1 = x1.loc[intersection]
            x2 = x2.loc[intersection]
        
        # Perform statistical test for each endpoint
        for y in endpoints:
            _, p_value = stats_func(x1.sort_index()[y],
                                    x2.sort_index()[y])
            results[y].append(p_value)
        
    # List the comparisons as index of the the output dataframe
    if relabel_conditions:
        df_index = [f'{a} <> {b}' for a,b in paired_labels]  # Use pretty format (with renamed conditions)
    else:
        df_index = [f'{a}X{b}' for a,b in paired_conditions] # Use old names
    df_index = pd.Index(df_index, name='Comparison')
   
    return pd.DataFrame(results, index=df_index)