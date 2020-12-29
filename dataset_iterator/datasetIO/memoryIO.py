from .datasetIO import DatasetIO
import threading

class MemoryIO(DatasetIO):
    def __init__(self, datasetIO:DatasetIO):
        super().__init__()
        self.datasetIO = datasetIO
        self.__lock__ = threading.Lock()
        self.datasets=dict()

    def close(self):
        self.datasets.clear()
        self.datasetIO.close()

    def get_dataset_paths(self, channel_keyword, group_keyword):
        return self.datasetIO.get_dataset_paths(channel_keyword, group_keyword)

    def get_dataset(self, path):
        if path not in self.datasets:
            with self.__lock__:
                if path not in self.datasets:
                    self.datasets[path] = self.datasetIO.get_dataset(path)[:] # load into memory
        return self.datasets[path]

    def get_attribute(self, path, attribute_name):
        return self.datasetIO.get_attribute(path, attribute_name)

    def create_dataset(self, path, **create_dataset_kwargs):
        self.datasetIO.create_dataset(path, **create_dataset_kwargs)

    def write_direct(self, path, data, source_sel, dest_sel):
        self.datasetIO.write_direct(path, data, source_sel, dest_sel)

    def __contains__(self, key):
        self.datasetIO.__contains__(key)

    def get_parent_path(self, path):
        self.datasetIO.get_parent_path(path)
