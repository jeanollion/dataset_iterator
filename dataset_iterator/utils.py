import numpy as np

def replace_last(s, old, new):
    # return (s[::-1].replace(old[::-1], new[::-1], 1))[::-1]
    return new.join(s.rsplit(old, 1))

def remove_duplicates(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def ensure_multiplicity(n, object):
    if object is None:
        return [None] * n
    if not isinstance(object, (list, tuple)):
        object = [object]
    if len(object)>1 and len(object)!=n:
        raise ValueError("length should be either 1 either equal to {}".format(n))
    if n>1 and len(object)==1:
        object = object*n
    elif n==0:
        return []
    return object

def flatten_list(l):
    flat_list = []
    for item in l:
        append_to_list(flat_list, item)
    return flat_list

def append_to_list(l, element):
    if isinstance(element, list):
        l.extend(element)
    else:
        l.append(element)

def pick_from_array(array, proportion):
    if proportion<=0:
        return []
    elif proportion<1:
        return np.random.choice(array, replace=False, size=int(len(array)*proportion+0.5))
    elif proportion==1:
        return array
    else:
        rep = int(proportion)
        return np.concatenate( [array]*rep + [pick_from_array(array, proportion - rep) ]).astype(np.int, copy=False)

def enrich_with_hardest_indices(evaluation_result, metrics_weights, hardest_example_percentage:float, enrichment_factor:float, minimize_metric=True):
    """Generates an array of indices that contains enriched hard examples (for active learning).

    Parameters
    ----------
    evaluation_result : ndarray of rank 2
        evaluation_result[:,0] are image indices
        evaluation_result[:,1:] are metric values
    metrics_weights : None or tuple of floats
        for each image index, the weighted average of metrics will be computed using those weights
    hardest_example_percentage : float
        Description of parameter `hardest_example_percentage`.
    enrichment_factor : float
        hardest examples will be enriched by this factor. e.g. if enrichment_factor, their occurence will be x2, i.e. there will be 2x chances to pick them
    minimize_metric : type
        if true, hardest examples are associated with lowest metric value

    Returns
    -------
    ndarray of in, rank 1
        shuffled indices with enriched hard exmaples to set with method set_allowed_indexes

    """
    assert evaluation_result.shape[1]==2 or evaluation_result.shape[1] == len(metrics_weights)+1
    if evaluation_result.shape[1]>2:
        eval = np.zeros(shape=(evaluation_result.shape[0], 2), dtype=evaluation_result.dtype)
        eval[:,0] = evaluation_result[:,0]
        for midx in range(1, evaluation_result.shape[1]):
            if metrics_weights[midx-1]!=0:
                eval[:, 1] += evaluation_result[:, midx] * metrics_weights[midx-1]
        eval[:, 1]/=np.sum(metrics_weights)
    else:
        eval = evaluation_result
    sorted_indices = eval.T[0][np.argsort(eval.T[-1])]
    n = int(hardest_example_percentage * evaluation_result.shape[0] + 0.5)
    hardest_indices = (sorted_indices[:n] if minimize_metric else sorted_indices[-n:]).astype(np.int)
    other_indices = (sorted_indices[n:] if minimize_metric else sorted_indices[:-n]).astype(np.int)
    rep = int(enrichment_factor)
    remain = int((enrichment_factor-rep) * n + 0.5)
    if rep>1 and remain>0:
        indices = np.concatenate([other_indices, np.repeat(hardest_indices, rep), (hardest_indices[:remain] if minimize_metric else hardest_indices[-remain:])])
    elif remain>0:
        indices = np.concatenate([other_indices, hardest_indices, (hardest_indices[:remain] if minimize_metric else hardest_indices[-remain:])])
    elif rep>1 and remain==0:
        indices = np.concatenate([other_indices, hardest_indices])
    else:
        indices = sorted_indices
        warning.warn("no enrichment in hardest indices")
    np.random.shuffle(indices)
    return indices
