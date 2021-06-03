from .datasetIO import DatasetIO, get_datasetIO
import itertools

class GroupDatasetIO(DatasetIO):
    """DatasetIO that wraps several dataset IO associated, each to one group.

    Parameters
    ----------
    dataset_map_group_keywords : dict
        keys: datasetIO (or path that DatasetIO.get_datasetIO method can transform in a datasetIO instance)
        values: group keyword

        all datasetIO should contain the same channels
    Attributes
    ----------
    group_keywords_map_dataset : dict
        mapping group_keyword -> datasetIO
    path_map_group_keyword : dict
        mapping of dataset path > dataset
    __lock__ : threading.Lock()
    dataset_map_group_keyword : dict

    """
    def __init__(self, dataset_map_group_keyword, channel_keywords):
        super().__init__()
        self.channel_keywords=channel_keywords
        self.dataset_map_group_keyword= dict()
        self.group_keywords_map_dataset = dict()
        for ds, grp in dataset_map_group_keyword.items():
            ds = get_datasetIO(ds)
            self.dataset_map_group_keyword[ds] = grp
            assert grp not in self.group_keywords_map_dataset, "duplicated group: {}".format(grp)
            self.group_keywords_map_dataset[grp] = ds

        # check that all channels are contained in all datasetIO and populate path_map_group_keyword
        self.path_map_group_keyword = dict()
        for channel_keyword, group_keyword in itertools.product(channel_keywords, dataset_map_group_keyword.values()):
            self.get_dataset_paths(channel_keyword, group_keyword)

    def close(self):
        for ds in self.dataset_map_group_keyword.keys():
            ds.close()

    def get_dataset_paths(self, channel_keyword, group_keyword):
        if group_keyword not in self.group_keywords_map_dataset:
            return []
        paths = self.group_keywords_map_dataset[group_keyword].get_dataset_paths(channel_keyword, None)
        for path in paths:
            if path not in self.path_map_group_keyword:
                self.path_map_group_keyword[path] = group_keyword
        return paths

    def _get_dsio(self, path):
        try:
            grp = self.path_map_group_keyword[path]
        except KeyError:
            print("path not found : {}".format(path))
            raise
        return self.group_keywords_map_dataset[grp]

    def get_dataset(self, path):
        return self._get_dsio(path).get_dataset(path)

    def get_attribute(self, path, attribute_name):
        return self._get_dsio(path).get_attribute(path, attribute_name)

    def create_dataset(self, path, **create_dataset_kwargs):
        self._get_dsio(path).create_dataset(path, **create_dataset_kwargs)

    def write_direct(self, path, data, source_sel, dest_sel):
        self._get_dsio(path).write_direct(path, data, source_sel, dest_sel)

    def __contains__(self, key):
        return key in self.path_map_group_keyword

    def get_parent_path(self, path):
        self._get_dsio(path).get_parent_path(path)
