from math import ceil
import numpy as np
from .multichannel_iterator import MultiChannelIterator
from scipy.ndimage import gaussian_filter

def get_optimal_tiling(dataset, channel:str, target_batch_size:int, tile_shape:tuple, group_keyword:str=None, tile_overlap_fraction:float=1./3):
    """
    Parameters
    ----------
    dataset : datasetIO or dataset path (string)
    channel : channel to inspect within dataset
    target_batch_size : batch size after tiling
    tile_shape: shape of tiles (Y, X)
    tile_overlap_fraction : hint on volume fraction of tiles that can overlap

    Returns
    -------
    batch_size:int, n_tiles:int so that target_batch_size = batch_size x n_tiles
    """
    it = MultiChannelIterator(dataset, channel_keywords=[channel], input_channels=[0], output_channels=[], group_keyword=group_keyword, incomplete_last_batch_mode=0)
    assert it.consistent_image_shape, "dataset contains sub-datasets with different images shapes"
    assert tile_overlap_fraction < 1, "invalid argument: tile_overlap_fraction must be <1"
    image_shape = it.channel_image_shapes[0][:it.n_spatial_dims]
    n_tiles = np.prod(image_shape) / ((1 - tile_overlap_fraction) * np.prod(tile_shape))
    assert n_tiles >= 1, f"tile volume is higher than image volume: image shape: {image_shape} tile shape: {tile_shape}"
    n_tile_int = int(n_tiles + 0.5)
    n_tile_candidates = [n_tile_int, int(n_tiles+1), int(n_tiles)]
    for i in range(1, n_tile_int):
        n_tile_candidates.append(n_tile_int + i)
        n_tile_candidates.append(n_tile_int - i)
    for n_t in n_tile_candidates:
        if target_batch_size % n_t == 0:
            return target_batch_size//n_t, n_t
    return target_batch_size, 1
def open_channel(dataset, channel_keyword:str, group_keyword:str=None, size=None):
    iterator = MultiChannelIterator(dataset=dataset, channel_keywords=[channel_keyword], group_keyword=group_keyword, input_channels=[0], output_channels=[], batch_size=1 if size is None else size, incomplete_last_batch_mode=0, shuffle=False)
    if size is None:
        iterator.batch_size=len(iterator)
    data = iterator[0]
    iterator._close_datasetIO()
    return data

def get_min_and_max(dataset, channel_keyword:str, group_keyword:str=None, batch_size=1):
    iterator = MultiChannelIterator(dataset=dataset, channel_keywords=[channel_keyword], group_keyword=group_keyword, input_channels=[0], output_channels=[], batch_size=batch_size, incomplete_last_batch_mode=0)
    vmin = float('inf')
    vmax = float('-inf')
    for i in range(len(iterator)):
        batch = iterator[i]
        vmin = min(batch.min(), vmin)
        vmax = max(batch.max(), vmax)
    iterator._close_datasetIO()
    return vmin, vmax

def get_min_max_range(dataset, channel_keyword:str = "/raw", group_keyword:str=None, min_centile_range:list=[0.01, 5.], max_centile_range:list=[95., 99.9], verbose:bool=False):
    """Computes a range for min and max value for random intensity normalization during data augmentation.
    Image can then be normalized using a random min and max value that will be mapped to [0, 1]

    Parameters
    ----------
    dataset : datasetIO/path(str) OR list/tuple of datasetIO/path(str)
    channel_keyword : str
        name of the dataset
    min_centile_range : list
        interval for min range in centiles
    min_centile_range : float
        interval for max range in centiles

    Returns
    -------
    min_range (list(2)) , max_range (list(2))

    """
    if isinstance(min_centile_range, float):
        min_centile_range = [min_centile_range, min_centile_range]
    if isinstance(max_centile_range, float):
        max_centile_range = [max_centile_range, max_centile_range]
    assert min_centile_range[0]<=min_centile_range[1], "invalid min range"
    assert max_centile_range[0]<=max_centile_range[1], "invalid max range"
    assert min_centile_range[0]<max_centile_range[1], "invalid min and max range"
    if isinstance(dataset, (list, tuple)):
        min_range, max_range = [], []
        for ds in dataset:
            min_r, max_r = get_min_max_range(ds, channel_keyword, min_centile_range, max_centile_range)
            min_range.append(min_r)
            max_range.append(max_r)
        if len(dataset)==1:
            return min_range[0], max_range[0]
        return min_range, max_range

    bins = get_histogram_bins_IPR(*get_histogram(dataset, channel_keyword, group_keyword=group_keyword, bins=1000), n_bins=256, percentiles=[0, 95], verbose=True)
    histogram, _ = get_histogram(dataset, channel_keyword, bins=bins)

    values = get_percentile(histogram, bins, min_centile_range + max_centile_range)
    min_range = [values[0], values[1]]
    max_range = [values[2], values[3]]
    if verbose:
        print(f"normalization: min_range: [{min_range[0]}; {min_range[1]}] max_range: [{max_range[0]}; {max_range[1]}]")
    return min_range, max_range

def get_histogram(dataset, channel_keyword:str, bins, bin_size=None, sum_to_one:bool=False, group_keyword:str=None, batch_size:int=1, return_min_and_bin_size:bool=False, smooth_scale:float = 0., smooth_scale_in_bin_unit:bool=True):
    iterator = MultiChannelIterator(dataset=dataset, channel_keywords=[channel_keyword], group_keyword=group_keyword, input_channels=[0], output_channels=[], batch_size=batch_size, incomplete_last_batch_mode=0)
    if bins is None:
        assert bin_size is not None
        vmin, vmax = get_min_and_max(dataset, channel_keyword, batch_size=batch_size)
        n_bins = round( (vmax - vmin) / bin_size )
        bin_size = (vmax - vmin) / n_bins
        bins = np.linspace(vmin, vmax, num=n_bins+1)
    if isinstance(bins, int):
        vmin, vmax = get_min_and_max(dataset, channel_keyword, batch_size=batch_size)
        bin_size = (vmax - vmin)/bins
        bins = np.linspace(vmin, vmax, num=bins+1)
    histogram = None
    for i in range(len(iterator)):
        batch = iterator[i]
        histo, _ = np.histogram(batch, bins)
        if histogram is None:
            histogram = histo
        else:
            histogram += histo
    iterator._close_datasetIO()
    if smooth_scale>0:
        if not smooth_scale_in_bin_unit:
            smooth_scale /= bin_size
        gaussian_filter(histogram, sigma = smooth_scale, mode="nearest", output=histogram)
    if sum_to_one:
        histogram=histogram/np.sum(histogram)
    if return_min_and_bin_size:
        return histogram, vmin, bin_size
    else:
        return histogram, bins

def get_histogram_bins_IPR(histogram, bins, n_bins, percentiles=[25, 75], min_bin_size=None, bin_range_percentiles=[0, 100], verbose = False):
    if isinstance(percentiles, (list, tuple)):
        assert len(percentiles)==2, "if list or tuple, percentiles should have length 2"
        assert percentiles[0]<percentiles[1] and percentiles[1]<=100 and percentiles[0]>=0, "invalid percentile values"
    else:
        assert percentiles>=0 and percentiles<=100, "invalid percentile valud"
        p2 = 100 - percentiles
        percentiles = [min(p2, percentiles), max(p2, percentiles)]
    if isinstance(bin_range_percentiles, (list, tuple)):
        assert len(bin_range_percentiles)==2, "if list or tuple, bin_range_percentiles should have length 2"
        assert bin_range_percentiles[0]<bin_range_percentiles[1] and bin_range_percentiles[1]<=100 and bin_range_percentiles[0]>=0, "invalid percentile values"
    else:
        assert bin_range_percentiles>=0 and bin_range_percentiles<=100, "invalid percentile valud"
        p2 = 100 - bin_range_percentiles
        bin_range_percentiles = [min(p2, bin_range_percentiles), max(p2, bin_range_percentiles)]
    pmin, pmax = get_percentile(histogram, bins, percentiles)
    bin_size = (pmax - pmin) / n_bins
    if min_bin_size is not None and min_bin_size>0:
        bin_size = max(min_bin_size, bin_size)
    if bin_range_percentiles[0]==0 and bin_range_percentiles[1]==100:
        bin_range_percentiles=[0, 100]
        vmin, vmax = bins[0], bins[-1]
    else:
        vmin, vmax = get_percentile(histogram, bins, bin_range_percentiles)
    n_bins = round( (vmax - vmin) / bin_size )
    if verbose:
        print("histo IPR: percentiles: [{}%={}, {}%={}], final range:[{}%={}, {}%={}], binsize: {}, nbins: {}".format(percentiles[0], pmin, percentiles[1], pmax, bin_range_percentiles[0], vmin, bin_range_percentiles[1], vmax, bin_size, n_bins))
    return np.linspace(vmin, vmax, n_bins+1)

def get_percentile(histogram, bins, percentile):
    assert np.shape(histogram)[0] == np.shape(bins)[0]-1, "invalid edges"
    cs = np.cumsum(histogram)
    if isinstance(percentile, (list, tuple)):
        percentile = np.array(percentile, dtype = "float64")
    percentile = percentile * cs[-1] / 100
    bin_centers = ( bins[1:] + bins[:-1] ) / 2
    return np.interp(percentile, cs, bin_centers)

def get_percentile_from_value(histogram, bins, value):
    assert np.shape(histogram)[0] == np.shape(bins)[0]-1, "invalid edges"
    cs = np.cumsum(histogram)
    cs = cs / cs[-1]
    if isinstance(value, (list, tuple)):
        value = np.array(value, dtype = "float64")
    bin_centers = ( bins[1:] + bins[:-1] ) / 2
    return np.interp(value, bin_centers, cs) * 100

def get_modal_value(histogram, bins, return_bin = False):
    bin_centers = ( bins[1:] + bins[:-1] ) / 2
    bin = np.argmax(histogram)
    if return_bin:
        return bin_centers[bin], bin
    else:
        return bin_centers[bin]

def get_mean_sd(dataset, channel_keyword:str, group_keyword:str=None, per_channel:bool=True): # TODO TEST
  params = dict(dataset=dataset,
              channel_keywords=[channel_keyword],
              group_keyword=group_keyword,
              input_channels = [0],
              output_channels=[],
              perform_data_augmentation=False,
              batch_size=1,
              incomplete_last_batch_mode=0,
              shuffle=False)
  it = MultiChannelIterator(**params)
  shape = it[0].shape
  ds_size = len(it)
  n_channels = shape[-1]
  sum_im = np.zeros(shape=(ds_size, n_channels), dtype=np.float64)
  sum2_im = np.zeros(shape=(ds_size, n_channels), dtype=np.float64)
  for i in range(ds_size):
    image = it[i]
    for c in range(n_channels):
      sum_im[i,c] = np.sum(image[...,c])
      sum2_im[i,c] = np.sum(image[...,c]*image[...,c])
  it._close_datasetIO()
  size = np.prod(shape[1:-1]) * ds_size * (1 if per_channel else shape[-1])
  sum_im /= size
  sum2_im /= size
  axis = 0 if per_channel else (0, 1)
  mean_ = np.sum(sum_im, axis=axis)
  sd_ = np.sqrt(np.sum(sum2_im, axis=axis) - mean_ * mean_)
  return mean_, sd_

def distribution_summary(dataset, channel_keyword:str, bins, group_keyword:str=None, percentiles = [5, 50, 95]):
    histogram, bins = get_histogram(dataset, channel_keyword, bins, group_keyword=group_keyword)
    mode = get_modal_value(histogram, bins)
    percentiles_values = get_percentile(histogram, bins, percentiles)
    percentiles = {p:v for p,v in zip(percentiles, percentiles_values)}
    mean, sd = get_mean_sd(dataset, channel_keyword, group_keyword)
    vmin, vmax = get_min_and_max(dataset, channel_keyword, group_keyword)
    print("range:[{:.5g}; {:.5g}] mode: {:.5g} mean: {}, sd: {}, percentiles: {}".format(vmin, vmax, mode,  "; ".join("{:.5g}".format(m) for m in mean), "; ".join("{:.5g}".format(s) for s in sd), "; ".join("{}%:{:.4g}".format(k,v) for k,v in percentiles.items())))
    return vmin, vmax, mode, mean, sd, percentiles
