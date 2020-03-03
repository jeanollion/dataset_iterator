# Dataset Iterator
This repo contains
- a keras iterator class for multi-channel images contained in dataset files such as hdf5
- a data generator class with transformation specific to mother machine data.

dataset structure is:

+dataset_name0
--+channel0
--+channel1
--+channel2
+dataset_name1
--+channel0
--+channel1
--+channel2

One dataset file can contain several sub-datasets (dataset_name0, dataset_name1, etc...), the iterator will iterate through all of them as if they were concatenated.

Each dataset contain channels that must have same shape. All datasets must have the same number of channels, and shape (except batch size) must be equal among datasets.

There can be more folder level, for instance to have train and test sets in the same file:

+experiment1
--+train
----+raw
----+labels
--+test
----+raw
----+labels

+experiment2
--+train
----+raw
----+labels
--+test
----+raw
----+labels

train_it = MultiChannelIterator(dataset_file_path = file_path, channel_keywords = ["/raw", "/labels"], group_keyword="train")
test_it = MultiChannelIterator(dataset_file_path = file_path, channel_keywords = ["/raw", "/labels"], group_keyword="test")

Such datasets can be generated using BACMMAN software (java)
