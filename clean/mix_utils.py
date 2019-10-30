

def buckets_interclass(labels, n_collectors, n_picks, *, verbose=False):
    """ Returns n collectors of size n_picks randomly picking from disctinct label buckets """
    print_ = lambda *a, **kwa: print(*a, **kwa) if verbose else None
    
    # C is the array of existing classes
    C = np.unique(labels)
    print_("There exists", len(C), "classes.")
    
    # To keep distinct labels at each collector position, we need n_collectors <= C
    assert n_collectors <= len(C)
    
    # Creates C buckets collecting indices of labels of each class, shuffle
    buckets = np.array([np.argwhere(labels == i).T[0] for i in C])
    np.random.shuffle(buckets.T)
    
    # Creates n_collectors collectors that will be returned of size n_picks
    collectors = np.zeros((n_collectors, n_picks), dtype=int)
    
    for pick_i in range(n_picks):
        # Create the array storing remaining buckets indices to choose
        remaining_buckets = np.arange(buckets.shape[0])
        for collector_i in range(n_collectors):
            # Get random bucket index
            bucket_i = np.random.choice(remaining_buckets)
            
            # Forbid picking this bucket again during this pick (no class collision)
            remaining_buckets = remaining_buckets[remaining_buckets != bucket_i]
            
            # Pick randomly chosen index from new randomly chosen bucket
            choice = np.random.choice(buckets[bucket_i])
            
            collectors[collector_i, pick_i] = choice
    
    return collectors


def buckets_intraclass(labels, n_collectors, n_picks, *, verbose=False):
    """ Returns n collectors of size n_picks randomly picking from a single label bucket """
    print_ = lambda *a, **kwa: print(*a, **kwa) if verbose else None
    
    # C is the array of existing classes
    C, counts = np.unique(labels, return_counts=True)
    print_("There exists", len(C), "classes. The smallest class is", counts.min(), "long.")
        
    # Our smallest class bucket must be at least n_collectors large to get distinct picks if chosen
    assert n_collectors <= counts.min(),\
    f"Ensure that the smallest class is larger than {n_collectors} items (currently:{counts.min()})"
    
    # Creates C buckets collecting indices of labels of each class, shuffle
    buckets = np.array([np.argwhere(labels == i).T[0] for i in C])
    np.random.shuffle(buckets.T)
    
    # Creates n_collectors collectors that will be returned of size n_picks
    collectors = np.zeros((n_collectors, n_picks), dtype=int)
    
    # Picks a single class bucket to always pick from    
    bucket_i = np.random.randint(buckets.shape[0])
    
    for pick_i in range(n_picks):
        # Create the array storing remaining items indices to choose
        remaining_choices = buckets[bucket_i].copy()
        choice = None
        
        for collector_i in range(n_collectors):
            
            # Pick an item from chosen bucket
            choice = np.random.choice(remaining_choices)
            
            # Remove the item for the rest of the choices
            remaining_choices = remaining_choices[remaining_choices != choice]            
            
            collectors[collector_i, pick_i] = choice
    
    return collectors
