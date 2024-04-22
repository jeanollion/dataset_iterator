import tensorflow as tf
import numpy as np
from math import ceil

def get_tf_version():
    return tuple(map(int, (tf.__version__.split("."))))

def replace_last(s, old, new):
    # return (s[::-1].replace(old[::-1], new[::-1], 1))[::-1]
    return new.join(s.rsplit(old, 1))

def remove_duplicates(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def ensure_size(array, size:int, shuffle:bool = False):
    rep = ceil(size / len(array))
    if rep > 1:
        array = np.tile(array, rep)
    if shuffle:
        array = np.random.permutation(array)
    return array[:size]

def ensure_same_shape(array_list):
    rank = len(array_list[0].shape)
    assert np.all([len(im.shape) == rank for im in array_list]), f"at least one array array have rank that differ from : {rank}"
    shape = [np.max([im.shape[ax] for im in array_list]) for ax in range(rank)]
    for i in range(len(array_list)):
        if np.any(array_list[i].shape != shape):
            array_list[i] = pad_to_shape(array_list[i], shape)
def pad_to_shape(array, shape):
    pad = [ (0, max(0, s - array.shape[i]))  for (i, s) in enumerate(shape)]
    return np.pad(array, tuple(pad))

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

def is_list(l):
    return isinstance(l, (list, tuple, np.ndarray))

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

def transpose_list(l):
    if len(l)==0:
        return l
    n_inner = len(l[0])
    return [[l[i][j] for i in range(len(l))] for j in range(n_inner)]

def pick_from_array(array, proportion, p=None):
    if proportion<=0:
        return []
    elif proportion<1:
        return np.random.choice(array, replace=False, size=int(len(array)*proportion+0.5), p=p)
    elif proportion==1:
        return array
    elif p is None: # do not use choice here to ensure more diversity
        rep = int(proportion)
        return np.concatenate( [array]*rep + [pick_from_array(array, proportion - rep) ]).astype(np.int, copy=False)
    else:
        return np.random.choice(array, replace=True, size=int(len(array) * proportion + 0.5), p=p)

def is_null(param, null_value):
    if param is None:
        return True
    if is_list(param):
        for p in param:
            if p != null_value:
                return False
        return True
    else:
        return param == null_value
