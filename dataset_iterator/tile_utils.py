from math import ceil, floor
import numpy as np
from numpy.random import randint, random
from .utils import ensure_multiplicity, is_null, random_choice_multidim
from scipy.ndimage import zoom

OVERLAP_MODE = ["NO_OVERLAP", "ALLOW", "FORCE"]


def extract_single_tile(tensor_slice=None, tile_shape=None, contraction_factor=None, return_none_for_identity:bool=True):
    if tensor_slice is not None:
        def extract_tiles_fun(batch, is_mask: bool, allow_random: bool = False, tile_probabilities=None):
            if isinstance(batch, (list, tuple)):
                return [b[tensor_slice] for b in batch]
            else:
                return batch[tensor_slice]
    elif tile_shape is not None: # fixed patch shape
        def extract_tiles_fun(batch, is_mask: bool, allow_random: bool = False, tile_probabilities=None):
            if isinstance(batch, (list, tuple)):
                return [crop(b, tile_shape) for b in batch]
            else:
                return crop(batch, tile_shape)
    elif contraction_factor is not None: # honor contraction factor
        assert contraction_factor is not None, "either tensor_slice, tile_shape or contraction factor must be provided"
        def extract_tiles_fun(batch, is_mask:bool, allow_random:bool=False, tile_probabilities=None):
            if isinstance(batch, (list, tuple)):
                return [crop_to_contraction(b, contraction_factor) for b in batch]
            else:
                return crop_to_contraction(batch, contraction_factor)
    else: # identity
        if return_none_for_identity:
            return None
        def extract_tiles_fun(batch, is_mask: bool, allow_random: bool = False, tile_probabilities=None):
            return batch
    return extract_tiles_fun

def extract_tile_function(tile_shape, perform_augmentation=True, overlap_mode=OVERLAP_MODE[1], min_overlap=1, n_tiles=None, random_stride=False, augmentation_rotate=True, anchor_point_mask_idx=None):
    return extract_tile_random_zoom_function(tile_shape, perform_augmentation=perform_augmentation, overlap_mode=overlap_mode, min_overlap=min_overlap, n_tiles=n_tiles, random_stride=random_stride, augmentation_rotate=augmentation_rotate, zoom_range=[1, 1], aspect_ratio_range=[1, 1], zoom_probability=0, random_channel_jitter_shape=None, anchor_point_mask_idx=anchor_point_mask_idx)

def extract_tile_random_zoom_function(tile_shape, perform_augmentation=True, overlap_mode=OVERLAP_MODE[1], min_overlap=1, n_tiles=None, random_stride=True, augmentation_rotate=True, zoom_range=[0.9, 1.1], aspect_ratio_range=[0.9, 1.1], zoom_probability:float=0.5, interpolation_order=1, random_channel_jitter_shape=None, anchor_point_mask_idx=None):
    def func(batch, is_mask:bool, allow_random:bool=True, tile_probabilities=None):
        if (is_null(zoom_range, 1) or is_null(zoom_range, 0) or zoom_probability==0) and is_null(random_channel_jitter_shape, 0):
            order = None
        else:
            if isinstance(batch, (list, tuple)):
                is_mask = ensure_multiplicity(len(batch), is_mask)
                order = [0 if m else interpolation_order for m in is_mask]
            else:
                order = 0 if is_mask else interpolation_order
        tiles = extract_tiles(batch, tile_shape=tile_shape, overlap_mode=overlap_mode, min_overlap=min_overlap, n_tiles=n_tiles, random_stride=random_stride if allow_random else False, zoom_range=zoom_range if allow_random else [1., 1.], aspect_ratio_range=aspect_ratio_range if allow_random else [1., 1.], zoom_probability=zoom_probability if allow_random else 0., interpolation_order=order, random_channel_jitter_shape=random_channel_jitter_shape if allow_random else None, tile_probabilities=tile_probabilities if allow_random else None, anchor_point_mask_idx=anchor_point_mask_idx)
        if perform_augmentation and allow_random:
            tiles = augment_tiles_inplace(tiles, rotate=augmentation_rotate and all([s==tile_shape[0] for s in tile_shape]), n_dims=len(tile_shape))
        return tiles
    return func

def extract_tiles(batch, tile_shape, overlap_mode=OVERLAP_MODE[1], min_overlap=1, n_tiles=None, random_stride=False, zoom_range=[0.9, 1.1], aspect_ratio_range=[0.9, 1.1], zoom_probability:float=0.5, interpolation_order=1, random_channel_jitter_shape=None, tile_probabilities = None, anchor_point_mask_idx:int=None):
    """Extract tiles with random zoom.

    Parameters
    ----------
    batch : numpy array or list of numpy arrays
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
    random_stride : bool
        whether tile coordinates should be randomized, within the gap / overlap zone
    zoom_range : list
        [min zoom ratio, max zoom ratio]
    aspect_ratio_range : list
        aspect ratio relative to the first axis.
        [min aspect ratio, max aspect ratio]
    zoom_probability : float in [0,1] probability to perform random zoom
    interpolation_order : int
        The order of the spline interpolation passed to scipy.ndimage.zoom, range 0-5
    random_channel_jitter_shape : list / tuple of ints or int
        if not None: tile coordinates are translated of a random value in this range. The range can be either the same for all dimensions (random_channel_jitter_range should be an integer) or distinct (random_channel_jitter_range should be a list or tuple of ints of length equal to the number of spatial dimensions of the batch)
    tile_probabilities: np.array (B, N_tiles)

    anchor_point_mask_idx: int or None
        if not None, index of mask that defines the anchor point, which is a point always included in all tiles.
        Anchor point defined as the mean coordinate of the mask along each axis.
        if input is a list of batches then anchor_mask_center_mask_idx is the index of the batch in this list, and middle channel is used. Otherwise it is the channel index of the batch.

    Returns
    -------
    numpy array
        tiles concatenated along first axis

    """
    multiple_inputs = isinstance(batch, (list, tuple))
    image_shape = batch[0].shape[1:-1] if multiple_inputs else batch.shape[1:-1]
    rank = len(image_shape)
    nchan_max = np.max([b.shape[-1] for b in batch]) if multiple_inputs else batch.shape[-1]
    batch_size = batch[0].shape[0] if multiple_inputs else batch.shape[0]
    assert rank in [2, 3], "only 2D or 3D images are supported"
    aspect_ratio_range = ensure_multiplicity(2, aspect_ratio_range)
    assert aspect_ratio_range[0]<=aspect_ratio_range[1], "invalid aspect_ratio_range"
    aspect_ratio_range = [1./aspect_ratio_range[1], 1./aspect_ratio_range[0]]
    zoom_range = ensure_multiplicity(2, zoom_range)
    assert zoom_range[0]<=zoom_range[1], "invalid zoom range"
    tile_shape = ensure_multiplicity(len(image_shape), tile_shape)
    for t, (t_s, i_s) in enumerate(zip(tile_shape, image_shape)):
        assert t_s <= i_s, f"invalid tile shape ({t_s}) at axis: {t}: must be lower than image shape ({i_s})"
    assert anchor_point_mask_idx is None or tile_probabilities is None, "anchor_point mode is incompatible with tile_probability mode"

    if tile_probabilities is not None: # choose tiles according to tile probability
        assert tile_probabilities.shape[0] == batch_size, f"invalid tile_probability array: first axis should be of same size as of batch size={batch_size} but is {tile_probabilities.shape[0]}"
        all_tile_coords = _get_tile_coords_overlap(image_shape, tile_shape, overlap_mode=OVERLAP_MODE[1], min_overlap=0, random_stride=False) # A, N
        assert all_tile_coords[0].shape[0] == tile_probabilities.shape[1], f"invalid tile_probability array: second axis should be of same size as of tile number={all_tile_coords[0].shape[0]} (all possibly overlapping tiles for an images shape: {image_shape} tile shape: {tile_shape}) but is {tile_probabilities.shape[1]}"
        # tile center = anchor points.
        anchor_point_list = np.array(all_tile_coords, dtype=np.float32) + np.array(tile_shape, dtype=np.float32)[:, np.newaxis] / 2 # anchor points = middle of tiles
        anchor_point_list = np.transpose(anchor_point_list, (1, 0)) # N, A
        if n_tiles is None:  # infer n_tiles from overlap
            n_tiles = _get_tile_coords_overlap(image_shape, tile_shape, overlap_mode, min_overlap, random_stride=False)[0].shape[0]
        anchor_point_list = np.stack([ random_choice_multidim(anchor_point_list, size=n_tiles, replace=True, p=tile_probabilities[b] ) for b in range(batch_size)])
        tile_coords_bat = [[_get_tile_coords(image_shape, tile_shape, (1, 1), random_stride, anchor_point=anchor_point_list[b][t])[0] for t in range(n_tiles)] for b in range(batch_size) ]
        anchor_intervals = None
    else:
        if anchor_point_mask_idx is not None:  # single anchor point per batch item, n_tiles around this anchor point
            if multiple_inputs:
                assert anchor_point_mask_idx < len(batch), f"invalid anchor mask idx: {anchor_point_mask_idx} > {len(batch)}"
                anchor_intervals = compute_anchor_interval(batch[anchor_point_mask_idx], jitter_shape=random_channel_jitter_shape)
            else:
                assert anchor_point_mask_idx < nchan_max, f"invalid anchor mask idx: {anchor_point_mask_idx} > {nchan_max}"
                anchor_intervals = compute_anchor_interval(batch[..., anchor_point_mask_idx:anchor_point_mask_idx+1], jitter_shape=random_channel_jitter_shape)
        else:
            anchor_intervals = None
        if n_tiles is None and anchor_intervals is None: # overlap mode
            tile_coords_bat = [_get_tile_coords_overlap(image_shape, tile_shape, overlap_mode, min_overlap, random_stride)  for _ in range(batch_size)]
        else:
            if n_tiles is None:  # infer n_tiles from overlap
                n_tiles = _get_tile_coords_overlap(image_shape, tile_shape, overlap_mode, min_overlap, random_stride)[0].shape[0]
            assert len(image_shape) == 2, "only 2d images supported when specifying n_tiles"
            tile_coords_bat = []
            for b in range(batch_size):
                _, n_tiles_yx = get_stride_2d(image_shape, tile_shape, n_tiles, anchor_interval=anchor_intervals[b] if anchor_intervals is not None else None)
                all_tile_coords = _get_tile_coords(image_shape, tile_shape, n_tiles_yx, random_stride,  anchor_interval=anchor_intervals[b] if anchor_intervals is not None else None)
                tile_coords_bat.append(_ensure_n_tiles(all_tile_coords, n_tiles, random=random_stride))

    n_t = tile_coords_bat[0][0].shape[0]
    if (is_null(zoom_range, 1) or is_null(zoom_range, 0) or zoom_probability == 0) and is_null( random_channel_jitter_shape, 0):
        def tile_fun(b):
            tiles = []
            for i in range(n_t):
                tile_coords_ba = [[tile_coords_bat[bidx][ax][i] for ax in range(rank)] for bidx in range(batch_size)]
                tiles.append(_subset_by_b(b, tile_coords_ba, tile_shape))
            return np.concatenate(tiles)
        if multiple_inputs:
            tiles = [tile_fun(b) for b in batch]
        else:
            tiles = tile_fun(batch)
        return tiles
    else:
        zoom_range_corrected = [1./np.max(zoom_range), np.min([ min(1./np.min(zoom_range), float(image_shape[ax])/float(tile_shape[ax])) for ax in range(rank) ])]
        zoom = random(n_t) * (zoom_range_corrected[1] - zoom_range_corrected[0]) + zoom_range_corrected[0]
        aspect_ratio_fun = lambda ax : random(n_t) * (np.minimum(image_shape[ax] / (zoom * tile_shape[ax]), aspect_ratio_range[1]) - aspect_ratio_range[0]) + aspect_ratio_range[0]
        aspect_ratio = [ aspect_ratio_fun(ax) for ax in range(1, rank) ]
        if zoom_probability < 1: # some tiles are not zoomed
            no_zoom = random(n_t) >= zoom_probability
            for t in range(n_t):
                if no_zoom[t]:
                    zoom[t] = 1
                    for ax in range(0, rank-1):
                        aspect_ratio[ax][t] = 1
        tile_size_fun = lambda ax : np.rint(zoom * tile_shape[ax]).astype(int) if ax==0 else np.rint(zoom * aspect_ratio[ax-1] * tile_shape[ax]).astype(int)
        r_tile_shape = [tile_size_fun(ax) for ax in range(rank)]
        for b in range(batch_size):
            for t in range(n_t): # translate coords if necessary so that tile is valid
                for ax in range(rank):
                    delta = tile_coords_bat[b][ax][t] + r_tile_shape[ax][t] - image_shape[ax]
                    if delta>0:
                        tile_coords_bat[b][ax][t] -= delta
                    if anchor_intervals is not None and r_tile_shape[ax][t] < tile_shape[ax]: # honor anchor interval when zooming in
                        if tile_coords_bat[b][ax][t] > anchor_intervals[b, ax, 0]:
                            tile_coords_bat[b][ax][t] = anchor_intervals[b, ax, 0]
                        elif tile_coords_bat[b][ax][t] + r_tile_shape[ax][t] < anchor_intervals[b, ax, 1]:
                            tile_coords_bat[b][ax][t] = anchor_intervals[b, ax, 1] - r_tile_shape[ax][t]

        def tile_fun_no_jitter(b, o):
            tiles = []
            for i in range(n_t):
                tile_shape_ =  [r_tile_shape[ax][i] for ax in range(rank)]
                tile_coords_ba = [[tile_coords_bat[bidx][ax][i] for ax in range(rank)] for bidx in range(batch_size)]
                tiles_ = _subset_by_b(b, tile_coords_ba, tile_shape_)
                tiles.append(_zoom(tiles_, tile_shape, o))
            return np.concatenate(tiles)

        if random_channel_jitter_shape is not None and nchan_max>1:
            random_channel_jitter_shape = ensure_multiplicity(rank, random_channel_jitter_shape)
            def r_channel_jitter_fun(b, ax):
                min_a = np.maximum(0, tile_coords_bat[b][ax]-random_channel_jitter_shape[ax] )
                max_a = np.minimum(tile_coords_bat[b][ax] + random_channel_jitter_shape[ax], image_shape[ax]-r_tile_shape[ax])
                return randint(min_a, max_a+1, size=n_t)
            tile_coords_cbat = [ [ [r_channel_jitter_fun(b, ax) for ax in range(rank)] for b in range(batch_size) ] for c in range(nchan_max) ]
            def tile_fun_jitter(b, o):
                if b.shape[-1]==1:
                    return tile_fun_no_jitter(b, o)
                tiles = []
                for i in range(n_t):
                    tile_shape_ = [r_tile_shape[ax][i] for ax in range(rank)]
                    tile_coords_cba = [[[tile_coords_cbat[c][bidx][ax][i] for ax in range(rank)] for bidx in range(batch_size)] for c in range(b.shape[-1])]
                    tiles_ = _subset_by_cb(b, tile_coords_cba, tile_shape_)
                    tiles.append(_zoom(tiles_, tile_shape, o))
                return np.concatenate(tiles)
            tile_fun = tile_fun_jitter
        else:
            tile_fun = tile_fun_no_jitter
    if multiple_inputs: # multi-array case (batch is actually a list of batches)
        interpolation_order= ensure_multiplicity(len(batch), interpolation_order)
        return [tile_fun(b, interpolation_order[i]) for i, b in enumerate(batch)]
    else:
        return tile_fun(batch, interpolation_order)

def _subset(batch, tile_coords_a, tile_shape):
    if len(tile_coords_a)==2:
        return batch[:, tile_coords_a[0]:tile_coords_a[0] + tile_shape[0], tile_coords_a[1]:tile_coords_a[1] + tile_shape[1]]
    else:
        return batch[:, tile_coords_a[0]:tile_coords_a[0] + tile_shape[0], tile_coords_a[1]:tile_coords_a[1] + tile_shape[1], tile_coords_a[2]:tile_coords_a[2] + tile_shape[2]]

def _subset_by_b(batch, tile_coords_ba, tile_shape):
    bsize = batch.shape[0]
    subsets = [_subset(batch[b:b+1], tile_coords_ba[b], tile_shape) for b in range(bsize)]
    return np.concatenate(subsets)

def _subset_by_cb(batch, tile_coords_cba, tile_shape):
    nchan = batch.shape[-1]
    subsets = [_subset_by_b(batch[..., c:c + 1], tile_coords_cba[c], tile_shape) for c in range(nchan)]
    return np.concatenate(subsets, axis=-1)

def _zoom(batch, target_shape, order):
    ratio = [float(i) / float(j) for i, j in zip(target_shape, batch.shape[1:-1])]
    if np.all(ratio == 1):
        return batch
    else:
        return zoom(batch, zoom = [1] + ratio + [1], order=order, grid_mode=False, mode="reflect")

def get_stride_2d(image_shape, tile_shape, n_tiles, anchor_interval=None):
    if n_tiles == 1:
        return (image_shape[0], image_shape[1]), (1, 1)
    assert len(image_shape)==2, "only available for 2d images"
    tile_shape = ensure_multiplicity(2, tile_shape)
    Sy = image_shape[0] - tile_shape[0]
    Sx = image_shape[1] - tile_shape[1]
    assert Sy>=0, f"tile size is too high on first axis: image size: {image_shape[0]} tile size: {tile_shape[0]}"
    assert Sx>=0, f"tile size is too high on second axis: image size: {image_shape[1]} tile size: {tile_shape[1]}"
    if anchor_interval is not None: # anchor interval limits the possible range of stride
        anchor_interval = np.asarray(anchor_interval)
        assert tuple(anchor_interval.shape) == (2,2), "anchor_interval must be a 2D interval shape (2, 2)"
        Dy = min(anchor_interval[0, 0], image_shape[0] - anchor_interval[0, 1])
        Dx = min(anchor_interval[1, 0], image_shape[1] - anchor_interval[1, 1])
        Sy = min(Sy, Dy)
        Sx = min(Sx, Dx)
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
    if abs(n_tiles_x_i-n_tiles_x) < abs(n_tiles_y_i-n_tiles_y):
        n_tiles_y_i = n_tiles // n_tiles_x_i
    else:
        n_tiles_x_i = n_tiles // n_tiles_y_i
    if n_tiles_x_i * n_tiles_y_i < n_tiles:
        if abs(n_tiles_x_i + 1 - n_tiles_x) < abs(n_tiles_y_i + 1 - n_tiles_y):
            n_tiles_x_i += 1
        else:
            n_tiles_y_i += 1
    n_tiles_x = n_tiles_x_i
    n_tiles_y = n_tiles_y_i
    stride_x = Sx // (n_tiles_x - 1) if n_tiles_x > 1 else image_shape[1]
    stride_y = Sy // (n_tiles_y - 1) if n_tiles_y > 1 else image_shape[0]
    return (stride_y, stride_x), (n_tiles_y, n_tiles_x)

def _ensure_n_tiles(tile_coords, n_tiles, random:bool=True):
    if tile_coords[0].shape[0] == n_tiles:
        return tile_coords
    elif tile_coords[0].shape[0] > n_tiles:
        if random:
            idxs = np.random.permutation(tile_coords[0].shape[0])[:n_tiles]
            return [tile_coords[i][idxs] for i in range(len(tile_coords))]
        else:
            return [tile_coords[i][:n_tiles] for i in range(len(tile_coords))]
    else:
        raise ValueError(f"Too few tiles: expected={n_tiles} got={tile_coords[0].shape[0]}")

def _get_tile_coords(image_shape, tile_shape, n_tiles, random_stride=False, anchor_interval=None, anchor_point=None):
    n_dims = len(image_shape)
    assert n_dims == len(tile_shape), "tile rank should be equal to image rank"
    assert n_dims == len(n_tiles), "n_tiles should have same rank as image"
    if anchor_point is not None and anchor_interval is None: # compute interval = point interval
        anchor_interval = np.stack([np.asarray(anchor_point), np.asarray(anchor_point)], -1) # (Ndims, 2)
    if anchor_interval is None:
        tile_coords_by_axis = [_get_tile_coords_axis(image_shape[ax], tile_shape[ax], n_tiles[ax], random_stride=random_stride) for ax in range(n_dims)]
    else:
        assert n_dims == len(anchor_interval), f"anchor_interval should have same rank as image: but is {anchor_interval} and rank={n_dims}"
        tile_coords_by_axis = [ _get_tile_coords_anchor_axis(image_shape[ax], tile_shape[ax], n_tiles[ax], anchor_interval=anchor_interval[ax], random_stride=random_stride) for ax in range(n_dims)]
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
    else:
        raise ValueError(f"Invalid overlap_mode = {overlap_mode} must be in {OVERLAP_MODE}")
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

def _get_tile_coords_anchor_axis(size, tile_size, n_tiles, anchor_interval, random_stride=False):
    if anchor_interval is None:
        raise ValueError("anchor_interval must be provided for this function.")

    # Calculate the minimum and maximum possible starting positions for each tile
    # such that the anchor_interval is always included
    right_bound = max(anchor_interval[1] - tile_size + 1, 0)
    left_bound = min(anchor_interval[0], size - tile_size)
    min_start = min(left_bound, right_bound)
    max_start = max(left_bound, right_bound)

    # If the tile_size is larger than the size, only one tile is possible
    if tile_size >= size:
        return [0] * n_tiles

    # Calculate the base coordinates such that all tiles include the anchor_interval
    if n_tiles == 1:
        base_coords = np.array([(min_start + max_start) // 2], dtype=np.int32)
    else:
        # Distribute tiles evenly around the anchor_interval
        base_coords = np.linspace(min_start, max_start, n_tiles, dtype=np.int32)

    # Apply random perturbation if required
    if random_stride:
        spacing = (max_start - min_start) / (n_tiles + 1)
        half_mean_gap = ceil(spacing / 2) if n_tiles > 1 else spacing
        perturbed_coords = base_coords + randint(-half_mean_gap, half_mean_gap + 1, size=n_tiles)
        if n_tiles>=2: # extra perturbation @ edges
            perturbed_coords[0] = max(min_start, perturbed_coords[0] - randint(0, half_mean_gap))
            perturbed_coords[-1] = min(max_start, perturbed_coords[0] + randint(0, half_mean_gap))
        return perturbed_coords
    else:
        return base_coords

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
    lambda img : np.rot90(img, k=3, axes=(0,1)), # rot + flip01
    lambda img : np.rot90(np.flip(img, axis=1), k=1, axes=(0,1)),
    lambda img : np.rot90(np.flip(img, axis=0), k=3, axes=(0,1))
]
AUG_FUN_3D = [
    lambda img : img,
    lambda img : np.flip(img, axis=1),
    lambda img : np.flip(img, axis=2),
    lambda img : np.flip(img, axis=(1, 2)),
    lambda img : np.rot90(img, k=1, axes=(1,2)),
    lambda img : np.rot90(img, k=3, axes=(1,2)), # rot + flip01
    lambda img : np.rot90(np.flip(img, axis=2), k=1, axes=(1,2)),
    lambda img : np.rot90(np.flip(img, axis=1), k=1, axes=(1,2))
]

def augment_tiles_inplace(tiles, rotate, n_dims=2):
    aug_fun = AUG_FUN_2D if n_dims==2 else AUG_FUN_3D
    n_tiles = tiles[0].shape[0] if isinstance(tiles, (tuple, list)) else tiles.shape[0]
    aug = randint(0, len(aug_fun) if rotate else len(aug_fun)/2, size=n_tiles)
    if isinstance(tiles, (tuple, list)):
        for bidx in range(len(tiles)):
            for b in range(n_tiles):
                if aug[b]>0: # 0 is identity
                    tiles[bidx][b] = aug_fun[aug[b]](tiles[bidx][b])
    else:
        for b in range(n_tiles):
            if aug[b]>0: # 0 is identity
                tiles[b] = aug_fun[aug[b]](tiles[b])
    return tiles

def crop_to_contraction(batch, contraction_factor):
    if len(batch.shape) == 4:
        contraction_factor = ensure_multiplicity(2, contraction_factor)
        _, Y, X, _ = batch.shape
        newY = contraction_factor[0] * int(Y / contraction_factor[0])
        newX = contraction_factor[1] * int(X / contraction_factor[1])
        return crop(batch, (newY, newX))
    elif len(batch.shape) == 5:
        contraction_factor = ensure_multiplicity(3, contraction_factor)
        _, Z, Y, X, _ = batch.shape
        newZ = contraction_factor[0] * int(Z / contraction_factor[0])
        newY = contraction_factor[1] * int(Y / contraction_factor[1])
        newX = contraction_factor[2] * int(X / contraction_factor[2])
        return crop(batch, (newZ, newY, newX))

def crop(batch, target_shape, start_point=None):
    if len(batch.shape) == 4:
        target_shape = ensure_multiplicity(2, target_shape)
        start_point = ensure_multiplicity(2, start_point) if start_point is not None else None
        _, Y, X, _ = batch.shape
        assert target_shape[0] <= Y and target_shape[1] <= X, "target shape is larger than tensor to crop"
        if start_point is None or np.all(start_point==0):
            return batch if target_shape[0]==Y and target_shape[1]==X else batch[:, :target_shape[0], :target_shape[1]]
        else:
            start_point[0] = min(start_point[0], Y - target_shape[0])
            start_point[1] = min(start_point[1], X - target_shape[1])
            return batch[:, start_point[0]:start_point[0]+target_shape[0], start_point[1]:start_point[1]+target_shape[1]]
    elif len(batch.shape) == 5:
        target_shape = ensure_multiplicity(3, target_shape)
        start_point = ensure_multiplicity(3, start_point) if start_point is not None else None
        _, Z, Y, X, _ = batch.shape
        assert target_shape[0] <= Z and target_shape[1] <= Y and target_shape[0] <= X, "target shape is larger than tensor to crop"
        if start_point is None or np.all(start_point == 0):
            return batch if target_shape[0] == Z and target_shape[1] == Y and target_shape[2] == X else batch[:, :target_shape[0], :target_shape[1], :target_shape[2]]
        else:
            start_point[0] = min(start_point[0], Z - target_shape[0])
            start_point[0] = min(start_point[1], Y - target_shape[1])
            start_point[1] = min(start_point[2], X - target_shape[2])
            return batch[:, start_point[0]:start_point[0]+target_shape[0], start_point[1]:start_point[1]+target_shape[1], start_point[2]:start_point[2]+target_shape[2]]

def compute_middle_points(tensor): # (B, Y, X) -> (B, 2)
    B = tensor.shape[0]
    ndim = tensor.ndim - 1  # exclude batch dimension
    middle_points = np.zeros((B, ndim))
    for axis in range(ndim):
        # Sum over all spatial axes except the current one
        axes = tuple(a for a in range(1, ndim + 1) if a != axis + 1)
        counts = np.sum(tensor > 0, axis=axes, keepdims=False)
        indices = np.arange(tensor.shape[axis + 1])[None, ]
        middle_points[:, axis] = np.sum(counts * indices, axis=1) / np.sum(counts, axis=1)
    return middle_points

def compute_anchor_interval(tensor, jitter_shape=None): # [ (B, Y, X, C), (2) ] -> (B, A=2 (Y, X), I=2 (min/max) )
    anchor_points = [ compute_middle_points(tensor[..., c]) for c in range(tensor.shape[-1]) ] # (C, B, 2)
    anchor_interval = np.stack([ np.min(anchor_points, axis=0), np.max(anchor_points, axis=0) ], -1) # (B, 2, 2)
    if jitter_shape is not None:
        jitter_shape = np.asarray(jitter_shape)[np.newaxis] # (1, 2)
        anchor_interval[..., 0] = np.maximum( anchor_interval[..., 0] - jitter_shape, 0 )
        max = np.array(tensor.shape[1:-1])[np.newaxis] # (1, 2)
        anchor_interval[..., 1] = np.minimum( anchor_interval[..., 1] + jitter_shape, max )
    return anchor_interval