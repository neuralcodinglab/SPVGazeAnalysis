import yaml


def load_yaml(path: str) -> dict:
    with open(path, 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    return params


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