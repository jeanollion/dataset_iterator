import numpy as np
from random import uniform, random
from scipy.ndimage.filters import gaussian_filter
from .utils import ensure_multiplicity, is_list
import helpers as dih

# image scaling
SCALING_MODES = ["RANDOM_CENTILES", "RANDOM_MIN_MAX", "FLUORESCENCE", "TRANSMITTED_LIGHT"]
def get_random_scaling_function(mode="RANDOM_CENTILES", dataset=None, channel_name:str=None, **kwargs):
    assert mode in SCALING_MODES, f"invalid mode, should be in {SCALING_MODES}"
    if mode == "RANDOM_CENTILES":
        min_centile_range = kwargs.get("min_centile_range", [0.1, 5])
        max_centile_range = kwargs.get("max_centile_range", [95, 99.9])
        assert min_centile_range[0] <= min_centile_range[1], "invalid min range"
        assert max_centile_range[0] <= max_centile_range[1], "invalid max range"
        assert min_centile_range[0] < max_centile_range[1], "invalid min and max range"
        saturate = kwargs.get("saturate", True)

        def fun(img):
            min0, min1, max0, max1 = np.percentile(img, min_centile_range + max_centile_range)
            cmin = uniform(min0, min1)
            cmax = uniform(max0, max1)
            while cmin >= cmax:
                cmin = uniform(min0, min1)
                cmax = uniform(max0, max1)
            if saturate:
                img = adjust_histogram_range(img, min=0, max=1, initial_range=[cmin, cmax])  # will saturate values under cmin or over cmax, as in real life.
            else:
                scale = 1. / (cmax - cmin)
                img = (img - cmin) * scale
            return img

        return fun
    elif mode == "RANDOM_MIN_MAX":
        return lambda img:random_histogram_range(img, **kwargs)
    elif mode == "FLUORESCENCE" or mode == "TRANSMITTED_LIGHT":
        fluo = mode == "FLUORESCENCE"
        if dataset is None:
            assert "scale_range" in kwargs and "center_range" in kwargs, "if no dataset is provided, scale_range and center_range must be provided"
            scale_range = kwargs["scale_range"]
            center_range = kwargs["center_range"]
        else :
            center_range, scale_range = get_center_scale_range(dataset, channel_name=channel_name, fluorescence=fluo, **kwargs)
        if not fluo and not kwargs.get("transmitted_light_per_image_mode", False):
            def fun(img):
                center = uniform(center_range[0], center_range[1])
                scale = uniform(scale_range[0], scale_range[1])
                mean = np.mean(img)
                sd = np.std(img)
                return (img - (center + mean)) / (scale * sd)
        else:
            def fun(img):
                center = uniform(center_range[0], center_range[1])
                scale = uniform(scale_range[0], scale_range[1])
                return (img - center) / scale
        return fun

def adjust_histogram_range(img, min=0, max=1, initial_range=None):
    if initial_range is None:
        initial_range=[img.min(), img.max()]
    return np.interp(img, initial_range, (min, max))

def compute_histogram_range(min_range, range=[0, 1]):
    if range[1]-range[0]<min_range:
        raise ValueError("Range must be greater than min_range")
    vmin = uniform(range[0], range[1]-min_range)
    vmax = uniform(vmin+min_range, range[1])
    return vmin, vmax

def random_histogram_range(img, min_range=0.1, range=[0,1]):
    min, max = compute_histogram_range(min_range, range)
    return adjust_histogram_range(img, min, max)

def get_histogram_normalization_center_scale_ranges(histogram, bins, center_percentile_extent, scale_percentile_range, verbose=False):
    assert dih is not None, "dataset_iterator package is required for this method"
    mode_value = dih.get_modal_value(histogram, bins)
    mode_percentile = dih.get_percentile_from_value(histogram, bins, mode_value)
    print("mode value={}, mode percentile={}".format(mode_value, mode_percentile))
    assert mode_percentile<scale_percentile_range[0], "mode percentile is {} and must be lower than lower bound of scale_percentile_range={}".format(mode_percentile, scale_percentile_range)
    if is_list(center_percentile_extent):
        assert len(center_percentile_extent) == 2
    else:
        center_percentile_extent = [center_percentile_extent, center_percentile_extent]
    percentiles = [max(0, mode_percentile-center_percentile_extent[0]), min(100, mode_percentile+center_percentile_extent[1])]
    scale_percentile_range = ensure_multiplicity(2, scale_percentile_range)
    if isinstance(scale_percentile_range, tuple):
        scale_percentile_range = list(scale_percentile_range)
    percentiles = percentiles + scale_percentile_range
    values = dih.get_percentile(histogram, bins, percentiles)
    mode_range = [values[0], values[1] ]
    scale_range = [values[2] - mode_value, values[3] - mode_value]
    if verbose:
        print("normalization_center_scale: modal value: {}, center_range: [{}; {}] scale_range: [{}; {}]".format(mode_value, mode_range[0], mode_range[1], scale_range[0], scale_range[1]))
    return mode_range, scale_range

def get_center_scale_range(dataset, channel_name:str = "/raw", fluorescence:bool = False, tl_sd_factor:float=3., fluo_scale_centile_range:list=[75, 99.9], fluo_center_centile_extent:list=[20, 30], transmitted_light_per_image_mode:bool=True):
    """Computes a range for center and for scale factor for data augmentation.
    Image can then be normalized using a random center C in the center range and a random scaling factor in the scale range: I -> (I - C) / S

    Parameters
    ----------
    dataset : datasetIO/path(str) OR list/tuple of datasetIO/path(str)
    channel_name : str
        name of the dataset
    fluorescence : bool
        in fluoresence mode:
            mode M is computed, corresponding to the Mp centile: M = centile(Mp). center_range = [centile(Mp-fluo_center_centile_extent), centile(Mp+fluo_center_centile_extent)]
            scale_range = [centile(fluo_scale_centile_range[0]) - M, centile(fluo_scale_centile_range[0]) + M ]
        in transmitted light mode: with transmitted_light_per_image_mode=True center_range = [mean - tl_sd_factor*sd, mean + tl_sd_factor*sd]; scale_range = [sd/tl_sd_factor, sd*tl_sd_factor]
        in transmitted light mode: with transmitted_light_per_image_mode=False: center_range = [-tl_sd_factor*sd, tl_sd_factor*sd]; scale_range = [1/tl_sd_factor, tl_sd_factor]
    tl_sd_factor : float
        use in the computation of transmitted light ranges cf description of fluorescence parameter
    fluo_scale_centile_range : list
        in fluoresence mode, interval for scale range in centiles
    fluo_center_centile_extent : float
        in fluoresence mode, extent for center range in centiles

    Returns
    -------
    scale_range (list(2)) , center_range (list(2))

    """
    if isinstance(dataset, (list, tuple)):
        scale_range, center_range = [], []
        for ds in dataset:
            sr, cr = get_center_scale_range(ds, channel_name, fluorescence, tl_sd_factor, fluo_scale_centile_range, fluo_center_centile_extent)
            scale_range.append(sr)
            center_range.append(cr)
        if len(dataset)==1:
            return scale_range[0], center_range[0]
        return scale_range, center_range
    if fluorescence:
        bins = dih.get_histogram_bins_IPR(*dih.get_histogram(dataset, channel_name, bins=1000), n_bins=256, percentiles=[0, 95], verbose=True)
        histo, _ = dih.get_histogram(dataset, channel_name, bins=bins)
        center_range, scale_range = get_histogram_normalization_center_scale_ranges(histo, bins, fluo_center_centile_extent, fluo_scale_centile_range, verbose=True)
        print("center: [{}; {}] / scale: [{}; {}]".format(center_range[0], center_range[1], scale_range[0], scale_range[1]))
        return center_range, scale_range
    else:
        mean, sd = dih.get_mean_sd(dataset, channel_name, per_channel=True)
        mean, sd = np.mean(mean), np.mean(sd)
        print("mean: {} sd: {}".format(mean, sd))
        if transmitted_light_per_image_mode:
            center_range, scale_range = [- tl_sd_factor*sd, tl_sd_factor*sd], [1./tl_sd_factor, tl_sd_factor]
            print("center: [{}; {}] / scale: [{}; {}]".format(center_range[0], center_range[1], scale_range[0], scale_range[1]))
        else:
            center_range, scale_range = [mean - tl_sd_factor*sd, mean + tl_sd_factor*sd], [sd/tl_sd_factor, sd*tl_sd_factor]
            print("center: [{}; {}] / scale: [{}; {}]".format(center_range[0], center_range[1], scale_range[0], scale_range[1]))
        return center_range, scale_range


# bluring, noise
def random_gaussian_blur(img, sigma=(1, 2)):
    if is_list(sigma):
        assert len(sigma)==2 and sigma[0]<=sigma[1], "sigma should be a range"
        sig = uniform(sigma[0], sigma[1])
    else:
        sig = sigma
    return gaussian_blur(img, sig)

def add_gaussian_noise(img, sigma=(0, 0.1), scale_sigma_to_image_range=False):
    if is_list(sigma):
        if len(sigma)==2:
            sigma = uniform(sigma[0], sigma[1])
        else:
            raise ValueError("Sigma  should be either a list/tuple of length 2 or a scalar")
    if scale_sigma_to_image_range:
        sigma *= (img.max() - img.min())
    gauss = np.random.normal(0,sigma,img.shape)
    return img + gauss
def gaussian_blur(img, sig):
    if len(img.shape)>2 and img.shape[-1]==1:
        return np.expand_dims(gaussian_filter(img.squeeze(-1), sig), -1)
    else:
        return gaussian_filter(img, sig)

# other helper functions
def sometimes(func, prob=0.5):
    return lambda im:func(im) if random()<prob else im

def apply_successively(*functions):
    if len(functions)==0:
        return lambda img:img
    def func(img):
        for f in functions:
            img = f(img)
        return img
    return func