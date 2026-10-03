import itertools

def generate_cpcv_splits(n_samples: int, n_groups: int, n_test_groups: int, purge_bars: int = 0, embargo_bars: int = 0):
    """
    Generate Combinatorial Purged Cross-Validation splits.
    
    Args:
        n_samples: Total number of samples.
        n_groups: Number of groups to split the data into.
        n_test_groups: Number of groups to use for testing in each split.
        purge_bars: Number of samples to purge around test groups.
        embargo_bars: Number of samples to embargo after test groups.
        
    Yields:
        tuple: (train_indices, test_indices) for each combination.
    """
    group_size = n_samples // n_groups
    groups = []
    for i in range(n_groups):
        start = i * group_size
        end = n_samples if i == n_groups - 1 else (i + 1) * group_size
        groups.append((start, end))
        
    for test_group_indices in itertools.combinations(range(n_groups), n_test_groups):
        test_indices = []
        train_indices = []
        
        for i in test_group_indices:
            start, end = groups[i]
            test_indices.extend(range(start, end))
            
        for i in range(n_groups):
            if i in test_group_indices:
                continue
                
            start, end = groups[i]
            
            adj_left_test = (i - 1) in test_group_indices
            adj_right_test = (i + 1) in test_group_indices
            
            eff_start = start + (embargo_bars if adj_left_test else 0)
            eff_end = end - (purge_bars if adj_right_test else 0)
            
            if eff_start < eff_end:
                train_indices.extend(range(eff_start, eff_end))
                
        yield train_indices, test_indices
