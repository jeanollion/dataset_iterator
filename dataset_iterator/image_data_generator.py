import tensorflow as tf
import numpy as np
from random import uniform, random, getrandbits
from .utils import is_list, ensure_multiplicity
from .pre_processing import get_center_scale_range, compute_histogram_range, adjust_histogram_range, add_poisson_noise, add_speckle_noise, add_gaussian_noise, gaussian_blur, get_histogram_elasticdeform_target_points_delta, histogram_elasticdeform, get_illumination_variation_target_points, illumination_variation

def get_image_data_generator(scaling_parameters=None, illumination_parameters=None, affine_transform_parameters=None):
    generators = []
    if scaling_parameters is not None:
        if isinstance(scaling_parameters, dict):
            generators.append(ScalingImageGenerator(**scaling_parameters))
        else:
            generators.append(None)
    if illumination_parameters is not None:
        if isinstance(illumination_parameters, dict):
            generators.append(IlluminationImageGenerator(**illumination_parameters))
        else:
            generators.append(None)
    if affine_transform_parameters is not None:
        if isinstance(affine_transform_parameters, dict):
            generators.append(KerasImageDataGenerator(**affine_transform_parameters))
        else:
            generators.append(None)
    if len(generators) == 1:
        return generators[0]
    else:
        return ImageGeneratorList(generators)

def data_generator_to_channel_postprocessing_fun(image_data_generator, channels):
    def pp_fun(batch_by_channel):
        if not is_list(image_data_generator):
            generator_list = [image_data_generator]
            channel_list = [channels]
        else:
            generator_list = image_data_generator
            channel_list = channels
            assert len(generator_list) == len(channel_list), "as many generators as channel list should be provided"
        for gen, channels_ in zip(generator_list, channel_list):
            for c in channels_:
                batch = batch_by_channel[c]
                for b in range(batch.shape[0]):
                    params = gen.get_random_transform(batch.shape[1:-1])
                    for c in range(batch.shape[-1]):
                        batch[b,...,c] = gen.apply_transform(batch[b,...,c], params)
    return pp_fun

class ImageGeneratorList():
    """Chain several ImageGenerators

        Parameters
        ----------
        generators : list of generators

        Attributes
        ----------
        generators

        """

    def __init__(self, generators:list):
        assert is_list(generators), "generator must be a list"
        self.generators = generators

    def get_random_transform(self, image_shape:tuple):
        all_params = {}
        for i, g in enumerate(self.generators):
            if g is not None:
                try:
                    params = g.get_random_transform(image_shape)
                    if params is not None:
                        all_params[i] = params
                except AttributeError:
                    pass
        return all_params

    def transfer_parameters(self, source:dict, destination:dict):
        for i, g in enumerate(self.generators):
            if g is not None and i in source and i in destination:
                try:
                    g.transfer_parameters(source[i], destination[i])
                except AttributeError:
                    pass

    def adjust_augmentation_param_from_mask(self, parameters:dict, mask):
        for i, g in enumerate(self.generators):
            if g is not None:
                try:
                    g.adjust_augmentation_param_from_mask(parameters[i], mask)
                except AttributeError:
                    pass

    def apply_transform(self, img, aug_params:dict):
        for i, g in enumerate(self.generators):
            if g is not None:
                try:
                    im2 = g.apply_transform(img, aug_params[i])
                    if im2 is not None:
                        img = im2
                except AttributeError:
                    pass
        return img

    def standardize(self, img):
        for g in self.generators:
            if g is not None:
                try:
                    im2 = g.standardize(img)
                    if im2 is not None:
                        img = im2
                except AttributeError:
                    pass
        return img

# image scaling
SCALING_MODES = ["RANDOM_CENTILES", "RANDOM_MIN_MAX", "FLUORESCENCE", "BRIGHT_FIELD"]
def get_random_scaling_function(mode="RANDOM_CENTILES", dataset=None, channel_name:str=None, **kwargs):
    data_gen = ScalingImageGenerator(mode, dataset, channel_name, **kwargs)
    def fun(img):
        params = data_gen.get_random_transform(img.shape)
        return data_gen.apply_transform(img, params)
    return fun

class ScalingImageGenerator():
    def __init__(self, mode="RANDOM_CENTILES", dataset=None, channel_name: str = None, **kwargs):
        assert mode in SCALING_MODES, f"invalid mode={mode}, should be in {SCALING_MODES}"
        self.mode = mode
        if mode == "RANDOM_CENTILES":
            self.min_centile_range = kwargs.get("min_centile_range", [0.1, 5])
            self.max_centile_range = kwargs.get("max_centile_range", [95, 99.9])
            assert self.min_centile_range[0] <= self.min_centile_range[1], "invalid min range"
            assert self.max_centile_range[0] <= self.max_centile_range[1], "invalid max range"
            assert self.min_centile_range[0] < self.max_centile_range[1], "invalid min and max range"
            self.saturate = kwargs.get("saturate", True)
        elif mode == "RANDOM_MIN_MAX":
            self.min_range = kwargs.get("min_range", 0.1)
            self.range = kwargs.get("range", [0, 1])
        elif mode == "FLUORESCENCE" or mode == "BRIGHT_FIELD":
            fluo = mode == "FLUORESCENCE"
            if "per_image" not in kwargs:
                kwargs["per_image"] = dataset is None
            self.per_image = kwargs.get("per_image", True)
            if not self.per_image and dataset is None:
                assert "scale_range" in kwargs and "center_range" in kwargs, "if no dataset is provided, scale_range and center_range must be provided"
                self.scale_range = kwargs["scale_range"]
                self.center_range = kwargs["center_range"]
            else:
                center_range, scale_range = get_center_scale_range(dataset, channel_name=channel_name, fluorescence=fluo, **kwargs)
                self.scale_range = scale_range
                self.center_range = center_range
    def get_random_transform(self, image_shape):
        params = {}
        if self.mode == "RANDOM_CENTILES":
            pmin = random()
            pmax = random()
            cmin = self.min_centile_range[0] + (self.min_centile_range[1] - self.min_centile_range[0]) * pmin
            cmax = self.max_centile_range[0] + (self.max_centile_range[1] - self.max_centile_range[0]) * pmax
            while cmax <= cmin:
                pmin = random()
                pmax = random()
                cmin = self.min_centile_range[0] + (self.min_centile_range[1] - self.min_centile_range[0]) * pmin
                cmax = self.max_centile_range[0] + (self.max_centile_range[1] - self.max_centile_range[0]) * pmax
            params["cmin"] = pmin
            params["cmax"] = pmax
        elif self.mode == "RANDOM_MIN_MAX":
            pmin, pmax = compute_histogram_range(self.min_range, self.range)
            params["vmin"] = pmin
            params["vmax"] = pmax
        elif self.mode == "FLUORESCENCE" or self.mode == "BRIGHT_FIELD":
            params["center"] = uniform(self.center_range[0], self.center_range[1])
            params["scale"] = uniform(self.scale_range[0], self.scale_range[1])
        return params

    def transfer_parameters(self, source, destination):
        if self.mode == "RANDOM_CENTILES":
            destination["cmin"] = source.get("cmin", 0)
            destination["cmax"] = source.get("cmax", 1)
        elif self.mode == "RANDOM_MIN_MAX":
            destination["vmin"] = source.get("vmin", 0)
            destination["vmax"] = source.get("vmax", 1)
        elif self.mode == "FLUORESCENCE" or self.mode == "BRIGHT_FIELD":
            destination["center"] = source["center"]
            destination["scale"] = source["scale"]

    def apply_transform(self, img, aug_params):
        if self.mode == "RANDOM_CENTILES":
            min0, min1, max0, max1 = np.percentile(img, self.min_centile_range + self.max_centile_range)
            cmin = min0 + (min1 - min0) * aug_params["cmin"]
            cmax = max0 + (max1 - max0) * aug_params["cmax"]
            if self.saturate:
                img = adjust_histogram_range(img, min=0, max=1, initial_range=[cmin,  cmax])  # will saturate values under cmin or over cmax, as in real life.
            else:
                scale = 1. / (cmax - cmin)
                img = (img - cmin) * scale
            return img
        elif self.mode == "RANDOM_MIN_MAX":
            return adjust_histogram_range(img, aug_params["vmin"], aug_params["vmax"])
        elif self.mode == "FLUORESCENCE" or self.mode == "BRIGHT_FIELD":
            center = aug_params["center"]
            scale = aug_params["scale"]
            if self.mode == "BRIGHT_FIELD" and self.per_image:
                mean = np.mean(img)
                sd = np.std(img)
                center = center * sd + mean
                scale = scale * sd
            elif self.mode == "FLUORESCENCE" and self.per_image:
                raise NotImplementedError("FLUORESCENCE per image is not implemented yet")
            return (img - center) / scale

    def standardize(self, img):
        return img


class IlluminationImageGenerator():
    def __init__(self, gaussian_blur_range:list=[1, 2], noise_intensity:float = 0.1, gaussian_noise:bool = True, poisson_noise:bool=True, speckle_noise:bool=False, histogram_elasticdeform_n_points:int=5, histogram_elasticdeform_intensity:float=0.5, illumination_variation_n_points:list=[0, 0], illumination_variation_intensity:float=0.6, illumination_variation_2d:bool = False):
        self.gaussian_blur_range = ensure_multiplicity(2, gaussian_blur_range)
        self.noise_intensity = noise_intensity
        self.gaussian_noise = gaussian_noise
        self.poisson_noise = poisson_noise
        self.speckle_noise = speckle_noise
        self.histogram_elasticdeform_n_points = histogram_elasticdeform_n_points
        assert histogram_elasticdeform_intensity <= 1, "histogram_elasticdeform_intensity should be in range [0, 1]"
        self.histogram_elasticdeform_intensity = histogram_elasticdeform_intensity
        self.illumination_variation_n_points = ensure_multiplicity(2, illumination_variation_n_points)
        assert illumination_variation_intensity <= 1, "illumination_variation_intensity should be in range [0, 1]"
        self.illumination_variation_intensity = illumination_variation_intensity
        self.illumination_variation_2d = illumination_variation_2d

    def get_random_transform(self, image_shape):
        params = {}
        params["gaussian_blur"] = uniform(self.gaussian_blur_range[0], self.gaussian_blur_range[1])
        gaussian = self.gaussian_noise and not getrandbits(1)
        speckle = self.speckle_noise and not getrandbits(1)
        poisson = self.poisson_noise and not getrandbits(1)
        ni = self.noise_intensity / float(1.5 ** (sum([gaussian, speckle, poisson]) - 1))
        if gaussian:
            params["gaussian_noise"] = uniform(0, ni * 0.7)
        if speckle:
            params["speckle_noise"] = uniform(0, ni)
        if poisson:
            params["poisson_noise"] = uniform(0, ni)
        
        if self.histogram_elasticdeform_n_points > 0 and self.histogram_elasticdeform_intensity > 0 : #and not getrandbits(1):
            # draw target point displacement  
            params["histogram_elasticdeform_target_points_delta"] = get_histogram_elasticdeform_target_points_delta(self.histogram_elasticdeform_n_points + 2) # +2 = edges
        elif "histogram_elasticdeform_target_points_delta" in params:
            del params["histogram_elasticdeform_target_points_delta"]

        if self.illumination_variation_n_points[0] == 0 and self.illumination_variation_n_points[0] == 0:
            n_points = 0
        else:
            if self.illumination_variation_2d:
                n_points = max(1, self.illumination_variation_n_points[0]) * max(1, self.illumination_variation_n_points[1])
            else:
                n_points = self.illumination_variation_n_points[0] + self.illumination_variation_n_points[1]
        if self.illumination_variation_intensity > 0 and n_points > 0: # and not getrandbits(1):
            params["illumination_variation_target_points"] = get_illumination_variation_target_points(n_points, self.illumination_variation_intensity)
        elif "illumination_variation_target_points" in params:
            del params["illumination_variation_target_points"]
        return params

    def transfer_parameters(self, source, destination):
        # do not transfer gaussian blur as focus may vary from one frame to the other
        if "poisson_noise" in source:
            destination["poisson_noise"] = source.get("poisson_noise", 0)
        elif "poisson_noise" in destination:
            del destination["poisson_noise"]
        if "speckle_noise" in source:
            destination["speckle_noise"] = source.get("speckle_noise", 0)
        elif "speckle_noise" in destination:
            del destination["speckle_noise"]
        if "gaussian_noise" in source:
            destination["gaussian_noise"] = source.get("gaussian_noise", 0)
        elif "gaussian_noise" in destination:
            del destination["gaussian_noise"]
        if "gaussian_blur" in source:
            destination["gaussian_blur"] = source.get("gaussian_blur", 0)
        elif "gaussian_blur" in destination:
            del destination["gaussian_blur"]
        if "histogram_elasticdeform_target_points_delta" in source:
            destination["histogram_elasticdeform_target_points_delta"] = source["histogram_elasticdeform_target_points_delta"]
        elif "histogram_elasticdeform_target_points_delta" in destination:
            del destination["histogram_elasticdeform_target_points_delta"]
        if "illumination_variation_target_points" in source:
            destination["illumination_variation_target_points"] = source["illumination_variation_target_points"]
        elif "illumination_variation_target_points" in destination:
            del destination["illumination_variation_target_points"]

    def apply_transform(self, img, aug_params):
        if "histogram_elasticdeform_target_points_delta" in aug_params:
            img = histogram_elasticdeform(img, self.histogram_elasticdeform_n_points, self.histogram_elasticdeform_intensity, target_point_delta=aug_params["histogram_elasticdeform_target_points_delta"])
        if "illumination_variation_target_points" in aug_params:
            target_points = aug_params.get("illumination_variation_target_points", None)
            img = illumination_variation(img, num_control_points_y=self.illumination_variation_n_points[0], num_control_points_x=self.illumination_variation_n_points[1], intensity=self.illumination_variation_intensity, target_points=target_points, perform_2d=self.illumination_variation_2d)
        if aug_params.get("gaussian_blur", 0) > 0:
            img = gaussian_blur(img, aug_params["gaussian_blur"])
        gaussian_noise_intensity = aug_params.get("gaussian_noise", 0)
        poisson_noise_intensity = aug_params.get("poisson_noise", 0)
        speckle_noise_intensity = aug_params.get("speckle_noise", 0)
        if gaussian_noise_intensity > 0 :
            img = add_gaussian_noise(img, sigma=gaussian_noise_intensity)
        if poisson_noise_intensity > 0 :
            img = add_poisson_noise(img, noise_intensity=poisson_noise_intensity)
        if speckle_noise_intensity > 0 :
            img = add_speckle_noise(img, sigma=speckle_noise_intensity)
        return img

    def standardize(self, img):
        return img

class KerasImageDataGenerator(tf.keras.preprocessing.image.ImageDataGenerator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def transfer_parameters(self, source, destination):
        destination['flip_vertical'] = source.get('flip_vertical', False)
        destination['flip_horizontal'] = source.get('flip_horizontal', False)
        destination['zy'] = source.get('zy', 1)
        destination['zx'] = source.get('zx', 1)
        destination['shear'] = source.get('shear', 0)
        if 'brightness' in source:
            destination['brightness'] = source['brightness']
        elif 'brightness' in destination:
            del destination['brightness']
        destination['theta'] = source.get('theta', 0)
        destination['tx'] = source.get('tx', 0)
        destination['ty'] = source.get('ty', 0)

class PreProcessingImageGenerator():
    """Simple data generator that only applies a custom pre-processing function to each image.
    To use as an element of the image_data_generators array in MultiChannelIterator

    Parameters
    ----------
    preprocessing_fun : function
        this function inputs a ndarray and return a ndarry of the same type

    Attributes
    ----------
    preprocessing_fun

    """

    def __init__(self, preprocessing_fun):
        assert callable(preprocessing_fun), "preprocessing_fun must be callable"
        self.preprocessing_fun = preprocessing_fun

    def get_random_transform(self, image_shape):
        return None

    def transfer_parameters(self, source, destination):
        pass

    def apply_transform(self, img, aug_params):
        return img

    def standardize(self, img):
        return self.preprocessing_fun(img)
