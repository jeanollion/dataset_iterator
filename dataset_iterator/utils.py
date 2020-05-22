import numpy as np

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
