import itertools
from math import ceil
import numpy as np
from numpy.random import randint
from .utils import ensure_multiplicity

OVERLAP_MODE = ["NO_OVERLAP", "ALLOW", "FORCE"]

def extract_tile_function(tile_shape, perform_augmentation=True, overlap_mode=OVERLAP_MODE[1], min_overlap=1, random_stride=False, scaling_function=None):
    def func(batch):
        tiles = extract_tiles(batch, tile_shape=tile_shape, overlap_mode=overlap_mode, min_overlap=min_overlap, random_stride=random_stride, return_coords=False)
        if perform_augmentation:
            tiles = augment_tiles_inplace(tiles, all([s==tile_shape[0] for s in tile_shape]), len(tile_shape))
        if scaling_function is not None:
            tiles = scaling_function(tiles)
        return tiles
    return func

def extract_tiles(batch, tile_shape, overlap_mode=OVERLAP_MODE[1], min_overlap=1, random_stride=False, return_coords=False):
    """Extract tiles.

    Parameters
    ----------
    batch : numpy array
        dimensions BYXC or BZYXC (B = batch)
    tile_shape : tuple
        tile shape, dimensions YX or ZYX. Z,Y,X,must be inferior or equal to batch dimensions
    overlap_mode : string
        one of ["NO_OVERLAP", "ALLOW", "FORCE"]
        "NO_OVERLAP" maximum number of tiles so that they do not overlap
        "ALLOW" maximum number of tiles that fit in the image, allowing overlap
        "FORCE"  maximum number of tiles that fit in the image with a minimum overlap defined by min_overlap
    min_overlap : integer or tuple
        min overlap along each spatial dimension. only used in mode "FORCE"
    random_stride : type
        whether tile coordinates should be randomized, within the gap / overlap zone
    return_coords : type
        whether tile coodinates should be returned

    Returns
    -------
    numpy array, ([numpy array])
        tiles concatenated along first axis, (tiles coordinates)

    """
    tile_coords = _get_tile_coords(batch.shape[1:-1], tile_shape, overlap_mode, min_overlap, random_stride)
    if len(tile_shape)==2:
        tiles = np.concatenate([batch[:, tile_coords[0][i]:tile_coords[0][i] + tile_shape[0], tile_coords[1][i]:tile_coords[1][i] + tile_shape[1]] for i in range(len(tile_coords[0]))])
    else:
        tiles = np.concatenate([batch[:, tile_coords[0][i]:tile_coords[0][i] + tile_shape[0], tile_coords[1][i]:tile_coords[1][i] + tile_shape[1], tile_coords[2][i]:tile_coords[2][i] + tile_shape[2]] for i in range(len(tile_coords[0]))])
    if return_coords:
        return tiles, tile_coords
    else:
        return tiles

def _get_tile_coords(image_shape, tile_shape, overlap_mode=OVERLAP_MODE[1], min_overlap=1, random_stride=False):
    n_dims = len(image_shape)
    min_overlap = ensure_multiplicity(n_dims, min_overlap)
    assert n_dims == len(tile_shape), "tile shape should be equal to image shape"
    tile_coords_by_axis = [_get_tile_coords_axis(image_shape[i], tile_shape[i], overlap_mode, min_overlap[i], random_stride) for i in range(n_dims)]
    return [a.flatten() for a in np.meshgrid(*tile_coords_by_axis, sparse=False, indexing='ij')]

def _get_tile_coords_axis(size, tile_size, overlap_mode=OVERLAP_MODE[1], min_overlap=1, random_stride=False):
    if tile_size==size:
        return [0]
    assert tile_size<size, "tile size must be inferior or equal to size"
    o_mode = OVERLAP_MODE.index(overlap_mode)
    assert o_mode>=0 and o_mode<=2, "invalid overlap mode"
    if o_mode==0:
        n_tiles = int(size/tile_size)
    elif o_mode==1:
        n_tiles = ceil(size/tile_size)
    elif o_mode==2:
        assert min_overlap<tile_size, "invalid min_overlap: value: {} should be <{}".format(min_overlap, tile_size)
        n_tiles = ceil((size - min_overlap)/(tile_size - min_overlap))
    if n_tiles==2:
        if o_mode==2 or (o_mode==1 and not random_stride):
            return [0, size-tile_size]

    sum_stride = np.abs(n_tiles * tile_size - size)
    stride = np.array([0]+[sum_stride//(n_tiles-1)]*(n_tiles-1), dtype=int)
    remains = sum_stride%(n_tiles-1)
    stride[1:remains+1] += 1
    if o_mode!=0:
        stride=-stride
    stride = np.cumsum(stride)
    coords = np.array([tile_size*idx + stride[idx] for idx in range(n_tiles)])
    if random_stride:
        half_mean_stride = np.abs(ceil(0.5 * sum_stride/(n_tiles-1)))
        coords += randint(-half_mean_stride, half_mean_stride, size=n_tiles)
        coords[0] = max(coords[0], 0)
        coords[-1] = min(coords[-1], size-tile_size)
    return coords

def augment_tiles(tiles, rotate, n_dims=2):
    flip_axis = [1, 2, (1,2)] if n_dims==2 else [2, 3, (2,3)]
    flips = [np.flip(tiles, axis=ax) for ax in flip_axis]
    augmented = np.concatenate([tiles]+flips, axis=0)
    if rotate:
        rot_axis = (1, 2) if n_dims==2 else (2, 3)
        augmented = np.concatenate((augmented, np.rot90(augmented, k=1, axes=rot_axis)))
    return augmented

AUG_FUN_2D = [
    lambda img : img,
    lambda img : np.flip(img, axis=0),
    lambda img : np.flip(img, axis=1),
    lambda img : np.flip(img, axis=(0, 1)),
    lambda img : np.rot90(img, k=1, axes=(0,1)),
    lambda img : np.rot90(img, k=3, axes=(0,1)), # rot + flip0
    lambda img : np.rot90(np.flip(img, axis=1), k=1, axes=(0,1)),
    lambda img : np.rot90(np.flip(img, axis=(0, 1)), k=1, axes=(0,1))
]
AUG_FUN_3D = [
    lambda img : img,
    lambda img : np.flip(img, axis=1),
    lambda img : np.flip(img, axis=2),
    lambda img : np.flip(img, axis=(1, 2)),
    lambda img : np.rot90(img, k=1, axes=(1,2)),
    lambda img : np.rot90(img, k=3, axes=(1,2)), # rot + flip0
    lambda img : np.rot90(np.flip(img, axis=2), k=1, axes=(1,2)),
    lambda img : np.rot90(np.flip(img, axis=(1, 2)), k=1, axes=(1,2))
]

def augment_tiles_inplace(tiles, rotate, n_dims=2):
    aug_fun = AUG_FUN_2D if n_dims==2 else AUG_FUN_3D
    aug = randint(0, len(aug_fun) if rotate else len(aug_fun)/2, size=tiles.shape[0])
    for b in range(tiles.shape[0]):
        if aug[b]>0: # 0 is identity
            tiles[b] = aug_fun[aug[b]](tiles[b])
    return tiles
