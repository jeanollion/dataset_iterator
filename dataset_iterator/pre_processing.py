import copy
import numpy as np
from random import uniform, random
from scipy.ndimage.filters import gaussian_filter
from scipy import interpolate
from .utils import ensure_multiplicity, is_list
from .helpers import get_modal_value, get_percentile, get_percentile_from_value, get_histogram, get_histogram_bins_IPR, get_mean_sd

def adjust_histogram_range(img, min=0, max=1, initial_range=None):
    if initial_range is None:
        initial_range=[img.min(), img.max()]
    return np.interp(img, initial_range, (min, max))

def compute_histogram_range(min_range, range=[0, 1]):
    if range[1]-range[0]<min_range:
        raise ValueError("Range must be greater than min_range")
    elif range[1]-range[0]==min_range:
        return range[0], range[1]
    vmin = uniform(range[0], range[1]-min_range)
    vmax = uniform(vmin+min_range, range[1])
    return vmin, vmax

def random_histogram_range(img, min_range=0.1, range=[0,1]):
    min, max = compute_histogram_range(min_range, range)
    return adjust_histogram_range(img, min, max)

def get_histogram_normalization_center_scale_ranges(histogram, bins, center_percentile_extent, scale_percentile_range, verbose=False):
    mode_value = get_modal_value(histogram, bins)
    mode_percentile = get_percentile_from_value(histogram, bins, mode_value)
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
    values = get_percentile(histogram, bins, percentiles)
    mode_range = [values[0], values[1] ]
    scale_range = [values[2] - mode_value, values[3] - mode_value]
    if verbose:
        print("normalization_center_scale: modal value: {}, center_range: [{}; {}] scale_range: [{}; {}]".format(mode_value, mode_range[0], mode_range[1], scale_range[0], scale_range[1]))
    return mode_range, scale_range

def get_center_scale_range(dataset, channel_name:str = "/raw", fluorescence:bool = False, bf_sd_factor:float=3., fluo_scale_centile_range:list=[75, 99.9], fluo_center_centile_extent:list=[20, 30], per_image:bool=True, verbose:bool=True):
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
        in bright field mode: with per_image=True center_range = [mean - bf_sd_factor*sd, mean + bf_sd_factor*sd]; scale_range = [sd/bf_sd_factor, sd*bf_sd_factor]
        in bright field mode: with per_image=False: center_range = [-bf_sd_factor*sd, bf_sd_factor*sd]; scale_range = [1/bf_sd_factor, bf_sd_factor]
    bf_sd_factor : float
        use in the computation of bright field ranges cf description of fluorescence parameter
    fluo_scale_centile_range : list
        in fluorescence mode, interval for scale range in centiles
    fluo_center_centile_extent : float
        in fluorescence mode, extent for center range in centiles

    Returns
    -------
    scale_range (list(2)) , center_range (list(2))

    """
    if isinstance(dataset, (list, tuple)):
        scale_range, center_range = [], []
        for ds in dataset:
            sr, cr = get_center_scale_range(ds, channel_name, fluorescence, bf_sd_factor, fluo_scale_centile_range, fluo_center_centile_extent)
            scale_range.append(sr)
            center_range.append(cr)
        if len(dataset)==1:
            return scale_range[0], center_range[0]
        return scale_range, center_range
    if fluorescence:
        if per_image:
            center_range, scale_range = [0, 1], [0, 1]
        else:
            bins = get_histogram_bins_IPR(*get_histogram(dataset, channel_name, bins=1000), n_bins=256, percentiles=[0, 95], verbose=verbose)
            histo, _ = get_histogram(dataset, channel_name, bins=bins)
            center_range, scale_range = get_histogram_normalization_center_scale_ranges(histo, bins, fluo_center_centile_extent, fluo_scale_centile_range, verbose=True)
            if verbose:
                print("center: [{}; {}] / scale: [{}; {}]".format(center_range[0], center_range[1], scale_range[0], scale_range[1]))
        return center_range, scale_range
    else:
        if per_image:
            center_range, scale_range = [- bf_sd_factor, bf_sd_factor], [1. / bf_sd_factor, bf_sd_factor]
        else:
            mean, sd = get_mean_sd(dataset, channel_name, per_channel=True)
            mean, sd = np.mean(mean), np.mean(sd)
            if verbose:
                print("mean: {} sd: {}".format(mean, sd))
            center_range, scale_range = [mean - bf_sd_factor * sd, mean + bf_sd_factor * sd], [sd / bf_sd_factor, sd * bf_sd_factor]
            if verbose:
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

def gaussian_blur(img, sig):
    if len(img.shape)>2 and img.shape[-1]==1:
        return np.expand_dims(gaussian_filter(img.squeeze(-1), sig), -1)
    else:
        return gaussian_filter(img, sig)

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

def add_poisson_noise(img, noise_intensity=[0, 0.1], adjust_intensity=True):
    if is_list(noise_intensity):
        if len(noise_intensity)==2:
            noise_intensity = uniform(noise_intensity[0], noise_intensity[1])
        else:
            raise ValueError("noise_intensity should be either a list/tuple of lenth 2 or a scalar")
    if adjust_intensity:
        noise_intensity /= 10.0 # so that intensity is comparable to gaussian sigma
    min = img.min()
    max = img.max()
    img = (img - min) / (max - min)
    output = np.random.poisson(img / noise_intensity) * noise_intensity
    return output * (max - min) + min

def add_speckle_noise(img, sigma=[0, 0.1]):
    if is_list(sigma):
        if len(sigma)==2:
            sigma = uniform(sigma[0], sigma[1])
        else:
            raise ValueError("noise_intensity  should be either a list/tuple of length 2 or a scalar")
    min = img.min()
    gauss = np.random.normal(1, sigma, img.shape)
    return (img - min) * gauss + min

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

# functions
def histogram_elasticdeform(image, num_control_points=5, intensity=0.5, target_point_delta = None, return_mapping = False):
    '''
    Adapted from delta software: https://gitlab.com/dunloplab/delta/blob/master/data.py
    It performs an elastic deformation on the image histogram to simulate
    changes in illumination
    '''
    assert intensity > 0 and intensity < 1, "Intensity should be in range ]0, 1["
    if target_point_delta is not None:
        assert len(target_point_delta)== num_control_points + 2, "invalid target point delta number"

    min = image.min()
    max = image.max()
    control_points = np.linspace(min, max, num=num_control_points + 2)
    if target_point_delta is None:
        target_point_delta = get_histogram_elasticdeform_target_points_delta(num_control_points + 2)
    target_points = control_points + target_point_delta * intensity * (max - min) / float(num_control_points + 1)
    if target_points[0] != min or target_points[-1] != max:
        target_points[0] = min
        target_points[-1] = max
    mapping = interpolate.PchipInterpolator(control_points, target_points)
    newimage = mapping(image)
    if return_mapping:
        return newimage, mapping
    else:
        return newimage

def get_histogram_elasticdeform_target_points_delta(n_points):
    assert n_points>2, "n_point must be > 2"
    deltas = np.random.uniform(low=-1, high=1, size = n_points)
    deltas[0] = 0
    deltas[-1] = 0
    return deltas

def illumination_variation(image, num_control_points_y=5, num_control_points_x=5, intensity=0.8, target_points = None, perform_2d:bool = False):
    '''
    Adapted from delta software: https://gitlab.com/dunloplab/delta/blob/master/data.py
    It simulates a variation in illumination along the length of the chamber
    '''
    assert intensity > 0 and intensity < 1, "Intensity should be in range ]0, 1["

    min = image.min()
    max = image.max()
    if num_control_points_y > 0 and num_control_points_x > 0 and perform_2d: # this is much slower than 2 times 1D grid
        num_control_points = num_control_points_x * num_control_points_y
        if target_points is not None:
            assert len(target_points) == num_control_points, "invalid target point number"
        else:
            target_points = get_illumination_variation_target_points(num_control_points, intensity)
        target_points = np.reshape(target_points, (num_control_points_y, num_control_points_x))
        cp_y = np.linspace(0, image.shape[0] - 1, num=num_control_points_y)
        cp_x = np.linspace(0, image.shape[1] - 1, num=num_control_points_x)
        mapping = interpolate.RegularGridInterpolator((cp_y, cp_x), target_points)
        y = np.linspace(0, image.shape[0] - 1, image.shape[0])
        x = np.linspace(0, image.shape[1] - 1, image.shape[1])
        Y, X = np.meshgrid(y, x, indexing='ij')
        curve_im = mapping((Y, X))
        if len(image.shape) == 3:
            curve_im = np.expand_dims(curve_im, -1)
        image = np.multiply(image - min, curve_im)
    elif num_control_points_y > 0 and num_control_points_x > 0 :
        target_points = np.sqrt(target_points) # each pixel will be multiplied by Y and X transform
    if num_control_points_y>0 and not perform_2d:
        # Create a random curve along y:
        if target_points is not None:
            assert len(target_points) == num_control_points_x + num_control_points_y, f"invalid target point number for y axis expected = {num_control_points_x + num_control_points_y} actual = {len(target_points)}"
            target_points_ = target_points[:num_control_points_y]
        else:
            target_points_ = get_illumination_variation_target_points(num_control_points_y, intensity)
        control_points = np.linspace(0, image.shape[0] - 1, num=num_control_points_y)
        mapping = interpolate.PchipInterpolator(control_points, target_points_)
        curve = mapping(np.linspace(0,image.shape[0]-1,image.shape[0]))
        newshape = [curve.shape[0], 1]
        if len(np.shape(image)) == 3:
            newshape += [1]
        curve_im_y = np.reshape(curve, newshape)
        image = np.multiply(image - min, curve_im_y)
    if num_control_points_x>0 and not perform_2d:
        # Create a random curve along y:
        if target_points is not None:
            assert len(target_points) == num_control_points_x + num_control_points_y, "invalid target point number for x axis"
            target_points_ = target_points[num_control_points_y:]
        else :
            target_points_ = get_illumination_variation_target_points(num_control_points_x, intensity)
        control_points = np.linspace(0, image.shape[1] - 1, num=num_control_points_x)
        mapping = interpolate.PchipInterpolator(control_points, target_points_)
        curve = mapping(np.linspace(0, image.shape[1]-1, image.shape[1]))
        newshape = [1, curve.shape[0]]
        if len(np.shape(image))==3:
            newshape += [1]
        curve_im_x = np.reshape(curve, newshape)
        min_ = 0 if num_control_points_y>0 else min
        image = np.multiply(image - min_, curve_im_x)
    # Rescale values to original range:
    return np.interp(image, (image.min(), image.max()), (min, max))

def get_illumination_variation_target_points(num_control_points, intensity):
    assert intensity >= 0 and intensity <= 1, "Intensity should be in range [0, 1]"
    return np.random.uniform(low=(1 - intensity) / 2.0, high=(1 + intensity) / 2.0, size=num_control_points)

