from .datasetIO import DatasetIO
import threading
import os
from os import listdir
from os.path import isfile, join
from keras_preprocessing import image
import keras_preprocessing.image.utils as im_utils

try:
    from PIL import Image as pil_image
except ImportError:
    pil_image = None


class MultipleFileIO(DatasetIO):
    """Allows to iterate an image dataset that contains several image files compatible with PILLOW

    Parameters
    ----------
    directory : string
        Each subdirectory in this directory will be considered to contain images from one channel.
    n_image_per_file : int
        how many images contains each file. set zero if each file contains several images but this number is unknown
    target_shape : tuple
        tuple of integers, dimensions to resize input images to. if None all image must have same dimension. no check is done at initialization
    channel_map_interpolation : dict
        interpolation scheme for each channel. Key is the channel directory
        value is the interpolation method used to resample the image if the
          target size is different from that of the loaded image.
          Supported methods are "nearest", "bilinear", and "bicubic".
          If PIL version 1.1.3 or newer is installed, "lanczos" is also
          supported. If PIL version 3.4.0 or newer is installed, "box" and
          "hamming" are also supported.
          if None: no resampling is performed else target_shape must be provided. if not none, an iterpolation for each channel that will be used must be provided
    data_format : type
        one of `channels_first`, `channels_last`.
    data_type : string
        type to use for generated arrays.
    supported_image_fun : type
        function that takes a file name as argument and return whether the file is a supported image type

    Attributes
    ----------
    image_shape : tuple
        target image shape
    n_image_per_file
    supported_image_fun
    channel_map_interpolation : interpolation function for each channel, or None
    data_format
    data_type

    """
    def __init__(self, directory, n_image_per_file, target_shape=None, channel_map_interpolation=None , data_format='channels_last', data_type='float32', supported_image_fun = lambda f : f.endswith(('.png', '.tif', '.tiff'))):
        super().__init__()
        self.path = directory
        if pil_image is None:
            raise ImportError('Could not import PIL.Image. The use of `MultipleFilesIO` requires PIL.')
        self.n_image_per_file = n_image_per_file
        self.image_shape = target_shape
        self.supported_image_fun = supported_image_fun
        self.channel_map_interpolation = {c:get_interpolation_function(target_shape, i) for c,i in channel_map_interpolation.items()} if channel_map_interpolation is not None else None
        self.channel_map_nn_interpolation = {c:i=='nearest' for c,i in channel_map_interpolation.items() } if channel_map_interpolation is not None else None
        self.data_format = data_format
        self.dtype = data_type

    def close(self):
        pass

    def get_dataset_paths(self, channel_keyword, group_keyword):
        # currently no group supported
        if self.n_image_per_file==1:
            return [join(self.path, channel_keyword)]
        else:
            return self.get_images(join(self.path, channel_keyword))

    def get_dataset(self, path):
        if self.n_image_per_file==1:
            channel_keyword = os.path.basename(os.path.normpath(path))
            return ImageListWrapper(path, self, channel_keyword)
        else:
            channel_keyword = os.path.basename(self.get_parent_path(path))
            return ImageWrapper(path, self, channel_keyword)

    def get_attribute(self, path, attribute_name):
        return None

    def create_dataset(self, path, **create_dataset_kwargs):
        raise NotImplementedError("Not implemented yet")

    def __contains__(self, key):
        if self.n_image_per_file: # datasets are channel folders
            for root, dirs, files in os.walk(self.path):
                if key in dirs:
                    return True
            return False
        else: # datasets are channel files
            for root, dirs, files in os.walk(self.path):
                if key in files:
                    return True
            return False

    def write_direct(self, path, data, source_sel, dest_sel):
        raise NotImplementedError("Not implemented yet")

    def get_images(self, path):
        return [join(path, f) for f in listdir(path) if self.supported_image_fun(f)]

    def get_parent_path(self, path):
        return os.path.dirname(os.path.normpath(path))

# adapted from keras_preprocessing
def get_interpolation_function(target_shape, interpolation):
    target_size = target_shape[::-1]
    if interpolation not in im_utils._PIL_INTERPOLATION_METHODS:
        raise ValueError('Invalid interpolation method {} specified. Supported methods are {}'.format(interpolation, ", ".join(im_utils._PIL_INTERPOLATION_METHODS.keys())))
    resample = im_utils._PIL_INTERPOLATION_METHODS[interpolation]
    def fun(img):
        if img.size != target_size:
            return img.resize(target_size, resample)
        else:
            return img
    return fun

# one file with multiple images
class ImageWrapper():
    def __init__(self, path, mfileIO, channel_keyword):
        self.path = path
        self.mfileIO=mfileIO
        if mfileIO.n_image_per_file==0 or mfileIO.image_shape is None:
            self.image = pil_image.open(self.path)
            self.shape = (self.image.n_frames,) + (mfileIO.image_shape if mfileIO.image_shape is not None else self.image.size[::-1])
        else:
            self.shape = (mfileIO.n_image_per_file,)+mfileIO.image_shape
        self.image = None
        self.interpolator = self.mfileIO.channel_map_interpolation[channel_keyword] if self.mfileIO.channel_map_interpolation is not None else None
        self.convert = not self.mfileIO.channel_map_nn_interpolation[channel_keyword] if self.mfileIO.channel_map_nn_interpolation is not None else False
        self.__lock__ = threading.Lock()

    def __getitem__(self, idx):
        with self.__lock__:
            if self.image is None:
                self.image = pil_image.open(self.path)
            assert idx<self.shape[0], "invalid index"
            self.image.seek(idx)
            if self.interpolator is not None:
                pil_img = self.image.convert("F") if self.convert else self.image
                pil_img = self.interpolator(pil_img)
            else:
                pil_img = self.image
            return image.img_to_array(pil_img, data_format=self.mfileIO.data_format, dtype=self.mfileIO.dtype)

# several files with one single image
class ImageListWrapper():
    def __init__(self, directory, mfileIO, channel_keyword):
        self.path = directory
        self.mfileIO=mfileIO
        self.image_paths = mfileIO.get_images(directory)
        if len(self.image_paths)==0:
            raise ValueError("No supported image found in dir: {}".format(directory))
        if mfileIO.image_shape is None:
            pil_img = pil_image.open(self.image_paths[0])
            self.shape = (len(self.image_paths),) + pil_img.size[::-1]
        else:
            self.shape = (len(self.image_paths),) + mfileIO.image_shape
        self.interpolator = self.mfileIO.channel_map_interpolation[channel_keyword] if self.mfileIO.channel_map_interpolation is not None else None
        self.convert = not self.mfileIO.channel_map_nn_interpolation[channel_keyword] if self.mfileIO.channel_map_nn_interpolation is not None else False

    def __getitem__(self, idx):
        #pil_img = im_utils.load_img(self.image_paths[idx])
        pil_img = pil_image.open(self.image_paths[idx])
        if self.interpolator is not None:
            if self.convert:
                pil_img = pil_img.convert("F")
            pil_img = self.interpolator(pil_img)
        return image.img_to_array(pil_img, data_format=self.mfileIO.data_format, dtype=self.mfileIO.dtype)
