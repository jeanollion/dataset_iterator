import numpy as np
from .datasetIO import DatasetIO


class DictDatasetIO(DatasetIO):
    """In-memory DatasetIO backed by a dict mapping `path -> numpy array`.

    Useful for tests and small synthetic datasets where no file should be
    written to disk. The data dict is shallow-copied at construction so the
    caller's dict is not aliased.

    Example
    -------
    >>> ds = DictDatasetIO({
    ...     "/posA/raw": np.zeros((5, 32, 32), dtype=np.float32),
    ...     "/posA/regionLabels": np.zeros((5, 32, 32), dtype=np.int32),
    ... })
    """
    def __init__(self, data: dict):
        super().__init__()
        self.data = dict(data)

    def close(self):
        self.data.clear()

    def get_dataset_paths(self, channel_keyword, group_keyword):
        return [p for p in self.data
                if p.endswith(channel_keyword)
                and (group_keyword is None or group_keyword in p)]

    def get_dataset(self, path):
        return self.data[path]

    def get_attribute(self, path, attribute_name):
        return None

    def create_dataset(self, path, **kwargs):
        if "data" in kwargs:
            self.data[path] = np.asarray(kwargs["data"])
        elif "shape" in kwargs:
            self.data[path] = np.zeros(kwargs["shape"], dtype=kwargs.get("dtype", np.float32))

    def write_direct(self, path, data, source_sel, dest_sel):
        self.data[path][dest_sel] = data[source_sel]

    def __contains__(self, key):
        return key in self.data

    def get_parent_path(self, path):
        idx = path.rfind('/')
        return path[:idx] if idx > 0 else None
