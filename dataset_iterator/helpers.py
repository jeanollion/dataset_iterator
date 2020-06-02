import numpy as np
from .multichannel_iterator import MultiChannelIterator

def open_channel(dataset, channel_keyword, size=None):
    iterator = MultiChannelIterator(dataset = dataset, channel_keywords=[channel_keyword], input_channels=list(np.arange(len(channel_keyword))) if isinstance(channel_keyword, (list, tuple)) else [0], output_channels=[], batch_size=1 if size is None else size, shuffle=False)
    if size is None:
        iterator.batch_size=len(iterator)
    return iterator[0]

def get_min_and_max(dataset, channel_keyword, batch_size=1):
    iterator = MultiChannelIterator(dataset = dataset, channel_keywords=[channel_keyword], output_channels=[], batch_size=batch_size)
    vmin = float('inf')
    vmax = float('-inf')
    for i in range(len(iterator)):
        batch = iterator[i]
        vmin = min(batch.min(), vmin)
        vmax = max(batch.max(), vmax)
    return vmin, vmax

def get_histogram(dataset, channel_keyword, bins, sum_to_one=False, batch_size=1):
    iterator = MultiChannelIterator(dataset = dataset, channel_keywords=[channel_keyword], output_channels=[], batch_size=batch_size)
    if isinstance(bins, int):
        vmin, vmax = get_min_and_max(dataset, channel_keyword, batch_size=batch_size)
        bins = np.linspace(vmin, vmax + (vmax - vmin)/bins, num=bins+1)
    histogram = None
    for i in range(len(iterator)):
        batch = iterator[i]
        histo, _ = np.histogram(batch, bins)
        if histogram is None:
            histogram = histo
        else:
            histogram += histo
    if sum_to_one:
        histogram=histogram/np.sum(histogram)
    return histogram, bins

def get_percentile(histogram, bins, percentile):
    cs = np.cumsum(histo)
    bin_idx = np.searchsorted(cs, np.percentile(cs, percentile)) # TODO linear interpolation for more precision within bin
    bin_centers = ( bins[1:] + bins[:-1] ) / 2
    return bin_centers[bin_idx]

def get_modal_value(histogram, bins):
    bin_centers = ( bins[1:] + bins[:-1] ) / 2
    return bin_centers[np.argmax(histogram)]

def get_mean_sd(dataset, channel_keyword, per_channel): # TODO TEST
  params = dict(dataset=dataset,
              channel_keywords=[channel_keyword],
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
  for i in range(range(ds_size)):
    #print("computing mean / sd : image: {}/{}".format(i, DS_SIZE[dataset_idx]))
    image = it[i]
    for c in range(n_channels):
      sum_im[i,c] = np.sum(image[...,c])
      sum2_im[i,c] = np.sum(image[...,c]*image[...,c])
  size = np.prod(shape[1:-1]) * ds_size
  sum_im /= size
  sum2_im /= size
  axis = 0 if per_channel else (0, 1)
  mean_ = np.sum(sum_im, axis=axis)
  sd_ = np.sqrt(np.sum(sum2_im, axis=axis) - mean_ * mean_)
  return [mean_, sd_]
