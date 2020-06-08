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

    def apply_transform(self, img, aug_params):
        return img

    def standardize(self, img):
        return self.preprocessing_fun(img)
