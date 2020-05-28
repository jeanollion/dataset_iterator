import numpy as np
from .multichannel_iterator import MultiChannelIterator

def open_channel(dataset, channel_keyword, size=None):
    iterator = MultiChannelIterator(dataset = dataset, channel_keywords=[channel_keyword], output_channels=[], batch_size=1 if size is None else size)
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
