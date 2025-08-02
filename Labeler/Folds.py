import numpy as np
from collections import defaultdict, Counter


def group_kfold_with_distribution(data, k, target_distribution):
    # Extract unique groups
    groups = defaultdict(list)
    for idx, (group, label) in enumerate(data):
        groups[group].append((idx, label))

    # Convert groups to a list of (group_id, indices, label_distribution)
    group_items = []
    for group, items in groups.items():
        indices = [idx for idx, _ in items]
        labels = [label for _, label in items]
        label_count = Counter(labels)
        group_dist = {label: label_count.get(label, 0) / len(labels) for label in target_distribution.keys()}
        group_items.append((group, indices, group_dist, len(labels)))

    # Initialize folds
    folds = [[] for _ in range(k)]
    fold_label_counts = [defaultdict(int) for _ in range(k)]

    # Sort groups by size (descending), helps balance folds
    group_items.sort(key=lambda x: -x[3])

    # Assign groups to folds
    for group, indices, group_dist, _ in group_items:
        best_fold = None
        best_diff = float('inf')

        for i in range(k):
            # Calculate the label distribution if we add this group to the fold
            simulated_counts = fold_label_counts[i].copy()
            for idx in indices:
                _, label = data[idx]
                simulated_counts[label] += 1

            # Calculate distribution after adding this group
            simulated_total = sum(simulated_counts.values())
            simulated_distribution = {label: simulated_counts[label] / simulated_total for label in target_distribution}

            # Calculate distance to target distribution
            diff = sum(abs(simulated_distribution[label] - target_distribution[label]) for label in target_distribution)

            if diff < best_diff:
                best_diff = diff
                best_fold = i

        # Assign group to best fold
        folds[best_fold].extend(indices)
        for idx in indices:
            _, label = data[idx]
            fold_label_counts[best_fold][label] += 1

    return folds

def extract_subdatasets(data, group_col, label_col, subset_size, target_distribution):
    """
    Extracts sub-datasets of a fixed length, preserving group integrity and adhering to a given label distribution.
    
    :param data: List of dictionaries, where each dictionary represents a data row.
    :param group_col: Key name indicating group identity.
    :param label_col: Key name indicating label identity.
    :param subset_size: Desired fixed length of each sub-dataset.
    :param target_distribution: Dictionary specifying the desired proportion of each label (e.g., {"A": 0.5, "B": 0.3, "C": 0.2}).
    :return: List of lists representing extracted sub-datasets.
    """
    
    # Shuffle data to avoid order bias
    # np.random.seed(42)
    # np.random.shuffle(data)
    
    # Group data by group_col
    grouped = defaultdict(list)
    for row in data:
        grouped[row[group_col]].append(row)

    
    # Count total instances per label
    label_counts = Counter(row[label_col] for row in data)
    
    # Compute required number of samples per label in each subset
    required_counts = {label: int(subset_size * target_distribution.get(label, 0)) for label in label_counts}
    
    subsets = []
    used_groups = set()
    
    while len(used_groups) < len(grouped):
        subset = []
        subset_label_counts = Counter()
        
        for label, target_count in required_counts.items():
            available_groups = [g for g in grouped if g not in used_groups and any(row[label_col] == label for row in grouped[g])]
            
            for group in available_groups:
                group_data = grouped[group]
                if subset_label_counts[label] < target_count:
                    subset.extend(group_data)
                    subset_label_counts.update(row[label_col] for row in group_data)
                    used_groups.add(group)
                
                if sum(subset_label_counts.values()) >= subset_size:
                    break
            if sum(subset_label_counts.values()) >= subset_size:
                break
        
        if subset:
            np.random.shuffle(subset)
            subsets.append(subset[:subset_size])
    sbs=[]
    for subset in subsets:
        groups = list(set([item['group']for item in subset]))
        sbs.append(groups)
    return sbs