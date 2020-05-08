import numpy as np
from dataset_iterator import MultiChannelIterator

class AutoencoderIterator(MultiChannelIterator):
    def __init__(self,
            dataset,
            channel_keywords=['/raw'],
            **kwargs):
        assert len(channel_keywords)==1, "Only one channel must be provided"
        super().__init__(dataset=dataset,
                         channel_keywords=channel_keywords,
                         input_channels=[0],
                         output_channels=[0],
                         **kwargs)
        
    # we allow the output_postprocessing_function to return the modified input
    def _get_batches_of_transformed_samples(self, index_array):
        batch_by_channel, aug_param_array, ref_chan_idx = self._get_batch_by_channel(index_array, self.perform_data_augmentation)
        #if self.extract_tile_function is not None: # if several channels -> save numpy state, and reset it before each tile computation
        #    batch_by_channel[ref_chan_idx] = self.extract_tile_function(batch_by_channel[ref_chan_idx])
        output = self._get_output_batch(batch_by_channel, ref_chan_idx, None)
        if isinstance(output, list):
            [input, output] = output
        else:
            input = self._get_input_batch(batch_by_channel, ref_chan_idx, None)
        if self.output_multiplicity>1:
            if not isinstance(output, list):
                output = [output] * self.output_multiplicity
            else:
                output = output * self.output_multiplicity
        return (input, output)

    def _apply_postprocessing_and_concat_weight_map(self, batch, output_chan_idx):
        if self.weight_map_functions is not None and self.weight_map_functions[output_chan_idx] is not None:
            wm = self.weight_map_functions[output_chan_idx](batch)
        else:
            wm = None
            batch_i = None
        if self.output_postprocessing_functions is not None and self.output_postprocessing_functions[output_chan_idx] is not None:
            batch = self.output_postprocessing_functions[output_chan_idx](batch)
        if isinstance(batch, tuple) or isinstance(batch, list):
            assert len(batch)==2, "if output_postprocessing_function must return either output batch either input & output batches"
            batch_i, batch = batch
        if wm is not None:
            batch = np.concatenate([batch, wm], -1)
        if batch_i is not None:
            return [batch_i, batch]
        else:
            return batch
