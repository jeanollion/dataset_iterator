import numpy as np
from dataset_iterator import MultiChannelIterator

class DenoisingIterator(MultiChannelIterator):
	def __init__(self,
				dataset_file_path,
				channel_keywords=['/raw'],
				weight_map_functions=None,
				output_postprocessing_functions=None, # we accept a function that returns both input and output
				output_multiplicity = 1,
				channel_scaling_param=None, #[{'level':1, 'qmin':5, 'qmax':95}],
				group_keyword=None,
				image_data_generators=None,
				batch_size=32,
				shuffle=True,
				perform_data_augmentation=True,
				seed=None,
				dtype='float32'):
		assert len(channel_keywords)==1, "Only one channel must be provided"
		super().__init__(dataset_file_path, channel_keywords, [0], [0], weight_map_functions, output_postprocessing_functions, None, output_multiplicity, channel_scaling_param, group_keyword, image_data_generators, batch_size, shuffle, perform_data_augmentation, seed)

	def _get_batches_of_transformed_samples(self, index_array):
		batch_by_channel, aug_param_array, ref_chan_idx = self._get_batch_by_channel(index_array, self.perform_data_augmentation)
		input = self._get_input_batch(batch_by_channel, ref_chan_idx, aug_param_array)
		output = self._get_output_batch(batch_by_channel, ref_chan_idx, aug_param_array)
		if isinstance(output, list):
			[input, output] = output
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
