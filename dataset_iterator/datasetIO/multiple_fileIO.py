from .datasetIO import DatasetIO
import threading
import os
from os import listdir
from os.path import isfile, join
from keras_preprocessing import image
import keras_preprocessing.image.utils as im_utils
import numpy as np
try:
    from PIL import Image as pil_image
except ImportError:
    pil_image = None


class MultipleFileIO(DatasetIO):
    """Allows to iterate an image dataset that contains several image files compatible with PILLOW

    Parameters
    ----------
    directory : string
        Each subdirectory in this directory will be considered to contain images from one channel with the name of the channel corresponding to the name of the subdirectory
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
          if None: crop to target_shape is performed. If not none, an iterpolation for each channel that will be used must be provided
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
        if channel_map_interpolation is not None:
            assert target_shape is not None, "target_shape must be provided if channel_map_interpolation is provided"
        self.n_image_per_file = n_image_per_file
        self.image_shape = target_shape
        self.supported_image_fun = supported_image_fun
        self.channel_map_interpolation = {c:get_interpolation_function(target_shape, i) for c,i in channel_map_interpolation.items()} if channel_map_interpolation is not None else None
        self.channel_map_nn_interpolation = {c:i=='nearest' for c,i in channel_map_interpolation.items() } if channel_map_interpolation is not None else None # flag channels that have nearest neighbor interpolation to avoid converting them to float
        self.data_format = data_format
        self.dtype = data_type
        self.crop_function = get_crop_function(target_shape) if target_shape is not None else None

    def close(self):
        pass

    def get_dataset_paths(self, channel_keyword, group_keyword=None):
        channel_keyword = fix_keyword(channel_keyword)
        all_dirs = scandir(self.path)
        if self.n_image_per_file==1:
            return [d for d in all_dirs if os.path.basename(d) == channel_keyword and (group_keyword is None or group_keyword in d) ]
        else:
            filtered_dirs = all_dirs if group_keyword is None else [d for d in all_dirs if group_keyword in d]
            all_imgs = []
            for d in filtered_dirs:
                all_imgs.extend(self.get_images(d, name = channel_keyword))
                all_imgs.extend(self.get_images(d, name = channel_keyword, npy=True))
            return all_imgs

    def __getitem__(self, path):
        return self.get_dataset(path)

    def get_dataset(self, path):
        if self.n_image_per_file==1:
            channel_keyword = os.path.basename(os.path.normpath(path))
            return ImageListWrapper(path, self, channel_keyword)
        else:
            channel_keyword = os.path.splitext(os.path.basename(path))[0]
            return ImageWrapper(path, self, channel_keyword)

    def get_attribute(self, path, attribute_name):
        return None

    def create_dataset(self, path, **create_dataset_kwargs):
        pass

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

    def write_direct(self, path, data, source_sel=None, dest_sel=None):
        if source_sel is not None or dest_sel is not None:
            assert self.n_image_per_file == 1, "index selection is only supported if n_image_per_file==1"
        if self.n_image_per_file == 1:
            if not os.path.isdir(path):
                os.makedirs(path)
            n_zeros = int(math.log10( len(data) )) + 1
            if dest_sel is None:
                dest_sel = source_sel
            for i, j in zip(source_sel, dest_sel):
                np.save(os.path.join(path, str(j).zfill(n_zeros)), data[i], allow_pickle=True)
        else:
            if not os.path.isdir(get_parent_path(path)):
                os.makedirs(get_parent_path(path))
            np.save(path, data)

    def get_images(self, path, name = None, npy=False):
        if npy:
            return [join(path, f) for f in listdir(path) if f.lower().endswith('.npy') and (name is None or os.path.splitext(f)[0] == name)]
        else:
            return [join(path, f) for f in listdir(path) if self.supported_image_fun(f.lower()) and (name is None or os.path.splitext(f)[0] == name)]

def fix_keyword(keyword):
    if keyword is None:
        return None
    if keyword[0]=='/':
        keyword = keyword[1:]
    return keyword

def get_parent_path(path):
    return os.path.dirname(os.path.normpath(path))

def get_crop_function(target_shape):
    target_size = target_shape[::-1]
    width, height = target_size
    def fun(img):
        if img.size != target_size:
            return img.crop((0, 0, width, height))
        else:
            return img
    return fun

def scandir(dirname):
    subfolders= [f.path for f in os.scandir(dirname) if f.is_dir()]
    for dirname in list(subfolders):
        subfolders.extend(scandir(dirname))
    return subfolders

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
        self.npy = path.endswith('.npy')
        if mfileIO.n_image_per_file==0 or mfileIO.image_shape is None: # get shape from image file
            if self.npy:
                img = np.load(self.path, mmap_mode='r')
                self.shape = img.shape
            else:
                self.image = pil_image.open(self.path)
                self.shape = (self.image.n_frames,) + (mfileIO.image_shape if mfileIO.image_shape is not None else self.image.size[::-1])
        else:
            self.shape = (mfileIO.n_image_per_file,)+mfileIO.image_shape
        self.image = None
        self.interpolator = self.mfileIO.channel_map_interpolation[channel_keyword] if self.mfileIO.channel_map_interpolation is not None else None
        if self.interpolator is None and self.mfileIO.crop_function is not None:
            self.interpolator = self.mfileIO.crop_function
        self.convert = not self.mfileIO.channel_map_nn_interpolation[channel_keyword] if self.mfileIO.channel_map_nn_interpolation is not None else False
        if self.npy:
            assert self.interpolator is None, "interpolation not supported (yet) with npy files"
        self.__lock__ = None if self.npy else threading.Lock()

    def __getitem__(self, idx):
        if self.npy:
            return np.load(self.path, mmap_mode='r')[idx]
        else:
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

    def __len__(self):
        return self.shape[0]

# several files with one single image
class ImageListWrapper():
    def __init__(self, directory, mfileIO, channel_keyword):
        self.path = directory
        self.mfileIO=mfileIO
        self.image_paths = mfileIO.get_images(directory)
        if len(self.image_paths)==0: # try with npy images
            self.image_paths = mfileIO.get_images(directory, npy=True)
            self.npy = True
            if len(self.image_paths)==0:
                raise ValueError("No supported image found in dir: {}".format(directory))
        else:
            self.npy = False
        if mfileIO.image_shape is None:
            if self.npy:
                img = np.load(self.image_paths[0], mmap_mode='r')
                self.shape = (len(self.image_paths),) + img.shape
            else:
                pil_img = pil_image.open(self.image_paths[0])
                self.shape = (len(self.image_paths),) + pil_img.size[::-1]
        else:
            self.shape = (len(self.image_paths),) + mfileIO.image_shape
        self.interpolator = self.mfileIO.channel_map_interpolation[channel_keyword] if self.mfileIO.channel_map_interpolation is not None else None
        if self.interpolator is None and self.mfileIO.crop_function is not None:
            self.interpolator = self.mfileIO.crop_function
        self.convert = not self.mfileIO.channel_map_nn_interpolation[channel_keyword] if self.mfileIO.channel_map_nn_interpolation is not None else False

        if self.npy:
            assert self.interpolator is None, "interpolation not supported (yet) with npy files"

    def __getitem__(self, idx):
        if self.npy:
            return np.load(self.image_paths[idx], mmap_mode='r')
        else:
            pil_img = pil_image.open(self.image_paths[idx])
            if self.interpolator is not None:
                if self.convert:
                    pil_img = pil_img.convert("F")
                pil_img = self.interpolator(pil_img)
            return image.img_to_array(pil_img, data_format=self.mfileIO.data_format, dtype=self.mfileIO.dtype)

    def __len__(self):
        return len(self.image_paths)
