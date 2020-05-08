from .datasetIO import DatasetIO, get_datasetIO
import threading

class MultipleDatasetIO(DatasetIO):
    def __init__(self, dataset_map_channel_keywords):
        self.dataset_map_channel_keywords= dict()
        for ds, keys in dataset_map_channel_keywords.items():
            ds = get_datasetIO(ds)
            if not isinstance(keys, (list, tuple)):
                keys = [keys]
            self.dataset_map_channel_keywords[ds] = keys
        self.channel_keywords_map_dataset = dict()
        for ds, keys in self.dataset_map_channel_keywords.items():
            for k in keys:
                assert k not in self.channel_keywords_map_dataset, "Error: duplicate channel_keyword: {}".format(k)
                self.channel_keywords_map_dataset[k] = ds
        self.path_map_dataset = dict()
        self.__lock__ = threading.Lock()

    def close(self):
        for ds in self.dataset_map_channel_keywords.keys():
            ds.close()

    def get_dataset_paths(self, channel_keyword, group_keyword):
        paths = self.channel_keywords_map_dataset[channel_keyword].get_dataset_paths(channel_keyword, group_keyword)
        with self.__lock__: # store computed path for faster retrieving of datasetIO
            for path in paths:
                if path not in self.path_map_dataset:
                    self.path_map_dataset[path] = self.channel_keywords_map_dataset[channel_keyword]
        return paths

    def _get_dsio(self, path):
        try:
            ds = self.path_map_dataset[path]
        except KeyError: # TODO enable use of different type of datasetIO : do not find path by replacing channelkeyword in MultiChannelIterator
            with self.__lock__:
                if path not in self.path_map_dataset:
                    for channel_keyword in self.channel_keywords_map_dataset.keys():
                        if channel_keyword in path:
                            self.path_map_dataset[path]  = self.channel_keywords_map_dataset[channel_keyword]
                            break
            ds = self.path_map_dataset[path]
            #raise ValueError("Unknown path: "+path)
        return ds

    def get_dataset(self, path):
        return self._get_dsio(path).get_dataset(path)

    def get_attribute(self, path, attribute_name):
        return self._get_dsio(path).get_attribute(path, attribute_name)

    def create_dataset(self, path, **create_dataset_kwargs):
        self._get_dsio(path).create_dataset(path, **create_dataset_kwargs)

    def write_direct(self, path, data, source_sel, dest_sel):
        self._get_dsio(path).write_direct(path, data, source_sel, dest_sel)

    def __contains__(self, key):
        for dsio in self.dataset_map_channel_keywords:
            if key in dsio:
                return True
        return False

    def get_parent_path(self, path):
        self._get_dsio(path).get_parent_path(path)
