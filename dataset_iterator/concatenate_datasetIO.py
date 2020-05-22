from .datasetIO import DatasetIO, get_datasetIO
import threading

class ConcatenateDatasetIO(DatasetIO):
    """DatasetIO resulting from the concatenation of several datasetIO. Dataset paths must be unique among all datasetIO.

    Parameters
    ----------
    dataset_io_list : list of datasetIO (or path that DatasetIO.get_datasetIO method can transform in a datasetIO instance)

    Attributes
    ----------
    dsio_list : list of datasetIO
    path_map_dsio : dict
        mapping dataset path -> datasetIO
    __lock__ : threading.Lock()

    """
    def __init__(self, dataset_io_list):
        self.dsio_list= [get_datasetIO(dsio) for dsio in dataset_io_list]
        self.path_map_dsio = dict()
        self.__lock__ = threading.Lock()

    def close(self):
        for dsio in self.dsio_list:
            dsio.close()

    def get_dataset_paths(self, channel_keyword, group_keyword):
        paths_map_dsio=dict()
        for dsio in self.dsio_list:
            paths = dsio.get_dataset_paths(channel_keyword, group_keyword)
            paths_map_dsio.update({p:dsio for p in paths})
        with self.__lock__: # store computed path for faster retrieving of datasetIO
            self.path_map_dsio.update(paths_map_dsio)
        return list(paths_map_dsio.keys())

    def get_dataset(self, path):
        return self._get_dsio(path).get_dataset(path)

    def _get_dsio(self, path):
        try:
            ds = self.path_map_dsio[path]
        except KeyError: # look for path in all dsio
            with self.__lock__:
                if path not in self.path_map_dsio:
                    for dsio in self.dsio_list:
                        if path in dsio:
                            self.path_map_dsio[path] = dsio
                            break
            ds = self.path_map_dsio[path]
        return ds

    def get_attribute(self, path, attribute_name):
        return self._get_dsio(path).get_attribute(path, attribute_name)

    def create_dataset(self, path, **create_dataset_kwargs):
        self._get_dsio(path).create_dataset(path, **create_dataset_kwargs)

    def write_direct(self, path, data, source_sel, dest_sel):
        self._get_dsio(path).write_direct(path, data, source_sel, dest_sel)

    def __contains__(self, key):
        for dsio in self.dsio_list:
            if key in dsio:
                return True
        return False

    def get_parent_path(self, path):
        self._get_dsio(path).get_parent_path(path)
