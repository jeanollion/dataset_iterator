import numpy as np
from .multichannel_iterator import MultiChannelIterator
from scipy.ndimage import gaussian_filter

def open_channel(dataset, channel_keyword, group_keyword=None, size=None):
    iterator = MultiChannelIterator(dataset = dataset, channel_keywords=[channel_keyword], group_keyword=group_keyword, input_channels=list(np.arange(len(channel_keyword))) if isinstance(channel_keyword, (list, tuple)) else [0], output_channels=[], batch_size=1 if size is None else size, shuffle=False)
    if size is None:
        iterator.batch_size=len(iterator)
    data = iterator[0]
    iterator._close_datasetIO()
    return data

def get_min_and_max(dataset, channel_keyword, group_keyword=None, batch_size=1):
    iterator = MultiChannelIterator(dataset = dataset, channel_keywords=[channel_keyword], group_keyword=group_keyword, output_channels=[], batch_size=batch_size)
    vmin = float('inf')
    vmax = float('-inf')
    for i in range(len(iterator)):
        batch = iterator[i]
        vmin = min(batch.min(), vmin)
        vmax = max(batch.max(), vmax)
    iterator._close_datasetIO()
    return vmin, vmax

def get_histogram(dataset, channel_keyword, bins, bin_size=None, sum_to_one=False, group_keyword=None, batch_size=1, return_min_and_bin_size=False, smooth_scale = 0, smooth_scale_in_bin_unit=True):
    iterator = MultiChannelIterator(dataset = dataset, channel_keywords=[channel_keyword], group_keyword=group_keyword, output_channels=[], batch_size=batch_size)
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

def get_mean_sd(dataset, channel_keyword, group_keyword=None, per_channel=True): # TODO TEST
  params = dict(dataset=dataset,
              channel_keywords=[channel_keyword],
              group_keyword=group_keyword,
              output_channels=[],
              perform_data_augmentation=False,
              batch_size=1,
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

def distribution_summary(dataset, channel_keyword, bins, group_keyword=None, percentiles = [5, 50, 95]):
    histogram, bins = get_histogram(dataset, channel_keyword, bins, group_keyword=group_keyword)
    mode = get_modal_value(histogram, bins)
    percentiles_values = get_percentile(histogram, bins, percentiles)
    percentiles = {p:v for p,v in zip(percentiles, percentiles_values)}
    mean, sd = get_mean_sd(dataset, channel_keyword, group_keyword)
    vmin, vmax = get_min_and_max(dataset, channel_keyword, group_keyword)
    print("range:[{:.5g}; {:.5g}] mode: {:.5g} mean: {}, sd: {}, percentiles: {}".format(vmin, vmax, mode,  "; ".join("{:.5g}".format(m) for m in mean), "; ".join("{:.5g}".format(s) for s in sd), "; ".join("{}%:{:.4g}".format(k,v) for k,v in percentiles.items())))
    return vmin, vmax, mode, mean, sd, percentiles
