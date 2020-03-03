# Dataset Iterator
This repo contains
- a keras iterator class for multi-channel images contained in dataset files such as hdf5
- a data generator class with transformation specific to mother machine data.

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

Such datasets can be generated directly from [BACMMAN software](https://github.com/jeanollion/bacmman).
