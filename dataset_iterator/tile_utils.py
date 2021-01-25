import itertools
from math import ceil, floor
import numpy as np
from numpy.random import randint
from .utils import ensure_multiplicity

OVERLAP_MODE = ["NO_OVERLAP", "ALLOW", "FORCE"]

def extract_tile_function(tile_shape, perform_augmentation=True, overlap_mode=OVERLAP_MODE[1], min_overlap=1, n_tiles=None, random_stride=False, scaling_function=None, augmentation_rotate=True):
    def func(batch):
        tiles = extract_tiles(batch, tile_shape=tile_shape, overlap_mode=overlap_mode, min_overlap=min_overlap, n_tiles=n_tiles, random_stride=random_stride, return_coords=False)
        if perform_augmentation:
            tiles = augment_tiles_inplace(tiles, rotate = augmentation_rotate and all([s==tile_shape[0] for s in tile_shape]), n_dims=len(tile_shape))
        if scaling_function is not None:
            tiles = scaling_function(tiles)
        return tiles
    return func

def extract_tiles(batch, tile_shape, overlap_mode=OVERLAP_MODE[1], min_overlap=1, n_tiles=None, random_stride=False, return_coords=False):
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
        "FORCE"  maximum number of tiles that fit in the image while enforcing a minimum overlap defined by min_overlap. If min_overlap is less than zero, it enforces a distance between tiles
    min_overlap : integer or tuple
        min overlap along each spatial dimension. only used in mode "FORCE"
    n_tiles : int
        if provided overlap_mode and min_overlap are ignored
    random_stride : type
        whether tile coordinates should be randomized, within the gap / overlap zone
    return_coords : type
        whether tile coodinates should be returned

    Returns
    -------
    numpy array, ([numpy array])
        tiles concatenated along first axis, (tiles coordinates)

    """
    tile_shape = ensure_multiplicity(len(batch.shape[1:-1]), tile_shape)
    if n_tiles is None:
        tile_coords = _get_tile_coords_overlap(batch.shape[1:-1], tile_shape, overlap_mode, min_overlap, random_stride)
    else:
        assert len(batch.shape[1:-1])==2, "only 2d images supported when specifying n_tiles"
        _, n_tiles_yx = get_stride_2d(batch.shape[1:-1], tile_shape, n_tiles)
        tile_coords = _get_tile_coords(batch.shape[1:-1], tile_shape, n_tiles_yx, random_stride)
    if len(batch.shape[1:-1])==2:
        tiles = np.concatenate([batch[:, tile_coords[0][i]:tile_coords[0][i] + tile_shape[0], tile_coords[1][i]:tile_coords[1][i] + tile_shape[1]] for i in range(len(tile_coords[0]))])
    else:
        tiles = np.concatenate([batch[:, tile_coords[0][i]:tile_coords[0][i] + tile_shape[0], tile_coords[1][i]:tile_coords[1][i] + tile_shape[1], tile_coords[2][i]:tile_coords[2][i] + tile_shape[2]] for i in range(len(tile_coords[0]))])
    if return_coords:
        return tiles, tile_coords
    else:
        return tiles

def get_stride_2d(image_shape, tile_shape, n_tiles):
    if n_tiles == 1:
        return (image_shape[0], image_shape[1]), (1, 1)
    assert len(image_shape)==2, "only available for 2d images"
    tile_shape = ensure_multiplicity(2, tile_shape)
    Sy = image_shape[0] - tile_shape[0]
    Sx = image_shape[1] - tile_shape[1]
    assert Sy>=0, "tile size is too high on first axis"
    assert Sx>=0, "tile size is too high on second axis"
    a = - n_tiles + 1
    b = Sy + Sx
    c = Sx*Sy
    d = b**2 - 4*a*c
    d = np.sqrt(d)
    r1 = (-b+d)/(2*a)
    r2 = (-b-d)/(2*a)
    stride = r1 if r1>r2 else r2
    n_tiles_x = (Sx / stride) + 1
    n_tiles_y = (Sy / stride) + 1
    n_tiles_x_i = round(n_tiles_x)
    n_tiles_y_i = round(n_tiles_y)
    if abs(n_tiles_x_i-n_tiles_x)<abs(n_tiles_y_i-n_tiles_y):
        n_tiles_x = n_tiles_x_i
        n_tiles_y = n_tiles // n_tiles_x
    else:
        n_tiles_y = n_tiles_y_i
        n_tiles_x = n_tiles // n_tiles_y
    stride_x = Sx // (n_tiles_x - 1) if n_tiles_x > 1 else image_shape[1]
    stride_y = Sy // (n_tiles_y - 1) if n_tiles_y > 1 else image_shape[0]
    return (stride_y, stride_x), (n_tiles_y, n_tiles_x)

def _get_tile_coords(image_shape, tile_shape, n_tiles, random_stride=False):
    n_dims = len(image_shape)
    assert n_dims == len(tile_shape), "tile rank should be equal to image rank"
    assert n_dims == len(n_tiles), "n_tiles should have same rank as image"
    tile_coords_by_axis = [_get_tile_coords_axis(image_shape[i], tile_shape[i], n_tiles[i], random_stride=random_stride) for i in range(n_dims)]
    return [a.flatten() for a in np.meshgrid(*tile_coords_by_axis, sparse=False, indexing='ij')]

def _get_tile_coords_overlap(image_shape, tile_shape, overlap_mode=OVERLAP_MODE[1], min_overlap=1, random_stride=False):
    n_dims = len(image_shape)
    min_overlap = ensure_multiplicity(n_dims, min_overlap)
    assert n_dims == len(tile_shape), "tile shape should be equal to image shape"
    tile_coords_by_axis = [_get_tile_coords_axis_overlap(image_shape[i], tile_shape[i], overlap_mode, min_overlap[i], random_stride) for i in range(n_dims)]
    return [a.flatten() for a in np.meshgrid(*tile_coords_by_axis, sparse=False, indexing='ij')]

def _get_tile_coords_axis_overlap(size, tile_size, overlap_mode=OVERLAP_MODE[1], min_overlap=1, random_stride=False):
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
        if min_overlap>=0:
            n_tiles = 1 + ceil((size - tile_size)/(tile_size - min_overlap)) # size = tile_size + (n-1) * (tile_size - min_overlap)
        else:
            n_tiles = floor((size - min_overlap)/(tile_size - min_overlap)) # n-1 gaps and n tiles: size = n * tile_size + (n-1)*-min_overlap
    return _get_tile_coords_axis(size, tile_size, n_tiles, random_stride)

def _get_tile_coords_axis(size, tile_size, n_tiles, random_stride=False):
    if n_tiles==1:
        coords = [(size - tile_size)//2]
        if random_stride and coords[0]>0:
            coords += randint(-coords[0], size-(coords[0]+tile_size), size=1)
        return coords
    if n_tiles==2:
        coords = [0, size-tile_size]
        if random_stride:
            gap = size - 2 * tile_size
            if gap>1:
                delta = randint(0, gap//2, size=2)
                coords[0] += delta[0]
                coords[1] -= delta[1]
        return coords

    sum_stride = np.abs(n_tiles * tile_size - size)
    stride = np.array([0]+[sum_stride//(n_tiles-1)]*(n_tiles-1), dtype=int)
    remains = sum_stride%(n_tiles-1)
    stride[1:remains+1] += 1
    if np.sign(n_tiles * tile_size - size)>0:
        stride=-stride
    stride = np.cumsum(stride)
    coords = np.array([tile_size*idx + stride[idx] for idx in range(n_tiles)])
    # print("before random: n_tiles: {}, tile_size: {} size: {}, stride: {}, coords: {}".format(n_tiles, tile_size, size, stride, coords))
    if random_stride:
        spacing = (size-tile_size)//(n_tiles-1)
        if spacing >= tile_size: # no overlap
            half_mean_gap = floor(0.5 * (spacing-tile_size) )
        else: # overlap
            half_mean_gap = ceil(0.5 * spacing )
        coords += randint(-half_mean_gap, half_mean_gap+1, size=n_tiles)
        coords[0] = max(coords[0], 0)
        coords[-1] = min(coords[-1], size-tile_size)
        # print("after random: spacing: {}, gap: {}, coords: {}".format(spacing, half_mean_gap, coords))
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
