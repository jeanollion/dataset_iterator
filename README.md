# Dataset Iterator
This repo contains keras iterator classes for multi-channel (time-lapse) images contained in dataset files such as hdf5.

## Dataset structure:
One dataset file can contain several sub-datasets (dataset_name0, dataset_name1, etc...), the iterator will iterate through all of them as if they were concatenated.

    .
    ├── ...
    ├── dataset_name0                    
    │   ├── channel0          
    │   └── channel1   
    │   └── ...
    ├── dataset_name0                    
    │   ├── channel0          
    │   └── channel1   
    │   └── ...
    └── ...

Each dataset contain channels (channel0, channel1 ...) that must have same shape. All datasets must have the same number of channels, and shape (except batch size) must be equal among datasets.

## Groups

There can be more folder level, for instance to have train and test sets in the same file:

    .
    ├── ...
    ├── experiment1                    
    │   ├── train          
    │   │   ├── raw
    │   │   └── labels
    │   └── test   
    │       ├── raw
    │       └── labels
    ├── experiment2                    
    │   ├── train          
    │   │   ├── raw
    │   │   └── labels
    │   └── test   
    │       ├── raw
    │       └── labels
    └── ...
```python
train_it = MultiChannelIterator(dataset_file_path = file_path, channel_keywords = ["/raw", "/labels"], group_keyword="train")
test_it = MultiChannelIterator(dataset_file_path = file_path, channel_keywords = ["/raw", "/labels"], group_keyword="test")
```
# Image formats
- Those iterators are using an object of class `DatasetIO` to access the data.
- There is currently an implementation of DatasetIO for .h5 files (`H5pyIO`), as well as dataset composed of multiple images files supported by PILLOW (`MultipleFileIO`).
- one can also concatenate datasets from different files:
  - if a dataset is split into several files that contain the same channels: use `ConcatenateDatasetIO`
  - if a dataset contains channels in different files, use: `MultipleDatasetIO`

# Demo
See this notebook for a demo: [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1J-UPivwyNTpyLhOMfzhfG0pIl6gDD9I5)
