import os
import tensorflow as tf
import numpy as np
from math import ceil


def get_tf_version():
    version = tf.__version__
    if "-rc" in version:
        version = version.split("-rc")[0]
    return tuple(map(int, (version.split("."))))

def is_keras_3():
    if get_tf_version() < (2, 16, 0):
        return False
    if "TF_USE_LEGACY_KERAS" in os.environ:
        return not (os.environ["TF_USE_LEGACY_KERAS"] == "1" or os.environ["TF_USE_LEGACY_KERAS"] == 1 or os.environ["TF_USE_LEGACY_KERAS"] == True or os.environ["TF_USE_LEGACY_KERAS"] == "True")
    else:
        return True

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

def is_dict(l):
    return isinstance(l, dict)

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


def random_choice_multidim(a, size=1, replace=True, p=None, axis=0):
    """
    Multidimensional version of np.random.choice.

    Parameters:
    - a: Multidimensional array to choose from (shape: (N, ...)).
    - size: Number of elements to choose.
    - replace: Whether to sample with replacement.
    - p: Probabilities for each element along axis (shape: (N,)).
    - axis: Axis along which to choose (default: 0).

    Returns:
    - Array of chosen elements (shape: (size, ...)).
    """
    if p is not None:
        p = np.asarray(p)
        p = p / np.sum(p)  # Normalize
        assert a.shape[axis] == p.shape[0], f"{a.shape[axis]} probabilities expected, got {p.shape[0]} instead"

    indices = np.random.choice(a.shape[axis], size=size, replace=replace, p=p)
    if axis == 0:
        return a[indices, ...]
    else:
        return np.take(a, indices, axis=axis)

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


def tf_to_np(tensor):
    if not tf.is_tensor(tensor):
        tensor = tf.convert_to_tensor(tensor, dtype=tf.float32)
    return tensor.numpy()
