import numpy as np
from .multichannel_iterator import MultiChannelIterator

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

def get_histogram(dataset, channel_keyword, bins, bin_size=None, sum_to_one=False, group_keyword=None, batch_size=1, return_min_and_bin_size=False):
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
    if sum_to_one:
        histogram=histogram/np.sum(histogram)
    if return_min_and_bin_size:
        return histogram, vmin, bin_size
    else:
        return histogram, bins

def get_percentile(histogram, bins, percentile):
    cs = np.cumsum(histogram)
    if isinstance(percentile, (list, tuple)):
        percentile = np.array(percentile)
    percentile = percentile * cs[-1] / 100
    bin_centers = ( bins[1:] + bins[:-1] ) / 2
    return np.interp(percentile, cs, bin_centers)

def get_modal_value(histogram, bins):
    bin_centers = ( bins[1:] + bins[:-1] ) / 2
    return bin_centers[np.argmax(histogram)]

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
    #print("computing mean / sd : image: {}/{}".format(i, DS_SIZE[dataset_idx]))
    image = it[i]
    for c in range(n_channels):
      sum_im[i,c] = np.sum(image[...,c])
      sum2_im[i,c] = np.sum(image[...,c]*image[...,c])
  it._close_datasetIO()
  size = np.prod(shape[1:-1]) * ds_size
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
