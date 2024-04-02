import math
from math import ceil, sqrt, log
import os
import numpy as np
import tensorflow as tf
from .index_array_iterator import IndexArrayIterator, INCOMPLETE_LAST_BATCH_MODE
from .pre_processing import adjust_histogram_range

class SampleProbabilityCallback(tf.keras.callbacks.Callback):
    def __init__(self, iterator, predict_fun, loss_fun, period:int=10, start_epoch:int=0, skip_first:bool=False, enrich_factor:float=10., quantile_max:float=0.99, quantile_min:float=None, disable_channel_postprocessing:bool=False, iterator_modifier:callable=None, workers=None, verbose:bool=True):
        super().__init__()
        self.period = period
        self.start_epoch = start_epoch
        self.skip_first=skip_first
        self.iterator = iterator
        self.predict_fun = predict_fun
        self.loss_fun = loss_fun
        self.enrich_factor = enrich_factor
        self.quantile_max = quantile_max
        self.quantile_min = quantile_min
        self.disable_channel_postprocessing = disable_channel_postprocessing
        self.iterator_modifier = iterator_modifier
        self.workers = workers
        self.verbose = verbose
        self.loss_idx = -1
        self.proba_per_loss = None
        self.n_losses = 0
    def on_epoch_begin(self, epoch, logs=None):
        if (epoch + self.start_epoch) % self.period == 0:
            if epoch == 0 and self.skip_first:
                return
            if self.iterator_modifier is not None:
                self.iterator_modifier(self.iterator, True)
            loss = compute_loss(self.iterator, self.predict_fun, self.loss_fun, disable_augmentation=True, disable_channel_postprocessing=self.disable_channel_postprocessing, workers=self.workers, verbose=self.verbose)
            if self.iterator_modifier is not None:
                self.iterator_modifier(self.iterator, False)
            self.proba_per_loss = get_index_probability(loss, enrich_factor=self.enrich_factor, quantile_max=self.quantile_max, quantile_min=self.quantile_min, verbose=self.verbose)
            self.n_losses = self.proba_per_loss.shape[0] if len(self.proba_per_loss.shape) == 2 else 1
        if self.proba_per_loss is not None:
            if len(self.proba_per_loss.shape) == 2:
                self.loss_idx = (self.loss_idx + 1) % self.n_losses
                proba = self.proba_per_loss[self.loss_idx]
            else:
                proba = self.proba_per_loss
            self.iterator.index_probability = proba # in case of multiprocessing iwth OrderedEnqueeur this will be taken into account only a next epoch has iterator has already been sent to processes at this stage


def get_index_probability_1d(loss, enrich_factor:float=10., quantile_max:float=0.99, quantile_min:float=None, max_power:int=10, verbose:int=1):
    assert 1 >= quantile_max > 0.5, "invalid max quantile"
    if quantile_min is None:
        quantile_min = 1 - quantile_max
    assert 0.5 > quantile_min >= 0, f"invalid min quantile: {quantile_min}"
    if 1. / enrich_factor < (1 - quantile_max): # incompatible enrich factor and quantile
        quantile_max = 1. - 1 / enrich_factor
        #print(f"modified quantile_max to : {quantile_max}")
    loss_quantiles = np.quantile(loss, [quantile_min, quantile_max])

    # TODO compute drop factor and enrich factor to get a constant probability factor in the end.

    Nh = loss[loss>=loss_quantiles[1]].shape[0]
    Ne = loss[loss <= loss_quantiles[0]].shape[0]
    loss_sub = loss[(loss<loss_quantiles[1]) & (loss>loss_quantiles[0])]
    Nm = loss_sub.shape[0]
    S = np.sum( ((loss_sub - loss_quantiles[0]) / (loss_quantiles[1] - loss_quantiles[0])) )
    p_max = enrich_factor / loss.shape[0]
    p_min = (1 - p_max * (Nh + S)) / (Nm + Ne - S) if (Nm + Ne - S)!=0 else -1
    if p_min<0:
        p_min = 0.
        target = 1./p_max - Nh
        if target <= 0: # cannot reach enrich factor: too many hard examples
            power = max_power
        else:
            fun = lambda power_: np.sum(((loss_sub - loss_quantiles[0]) / (loss_quantiles[1] - loss_quantiles[0])) ** power_)
            power = 1
            power_inc = 0.25
            Sn = S
            while power < max_power and Sn > target:
                power += power_inc
                Sn = fun(power)
            if power > 1 and Sn < target:
                power -= power_inc
    else:
        power = 1
    #print(f"p_min {p_min} ({(1 - p_max * (Nh + S)) / (Nm + Ne - S)}) Nh: {Nh} nE: {Ne} Nm: {Nm} S: {S} pmax: {p_max} power: {power}")
    # drop factor at min quantile, enrich factor at max quantile, interpolation in between
    def get_proba(value):
        if value <= loss_quantiles[0]:
            return p_min
        elif value >= loss_quantiles[1]:
            return p_max
        else:
            return p_min + (p_max - p_min) * ((value - loss_quantiles[0]) / (loss_quantiles[1] - loss_quantiles[0]))**power

    vget_proba = np.vectorize(get_proba)
    proba = vget_proba(loss)
    p_sum = float(np.sum(proba))
    proba /= p_sum
    if verbose > 1:
        print(f"loss proba: [{np.min(proba) * loss.shape[0]}, {np.max(proba) * loss.shape[0]}] pmin: {p_min} pmax: {p_max} power: {power} sum: {p_sum} quantile_max: {quantile_max}", flush=True)
    return proba

def get_index_probability(loss, enrich_factor:float=10., quantile_max:float=0.99, quantile_min:float=None, verbose:int=1):
    if len(loss.shape) == 1:
        return get_index_probability_1d(loss, enrich_factor=enrich_factor, quantile_max=quantile_max, quantile_min=quantile_min, verbose=verbose)
    probas_per_loss = [get_index_probability_1d(loss[:, i], enrich_factor=enrich_factor, quantile_max=quantile_max, quantile_min=quantile_min, verbose=verbose) for i in range(loss.shape[1])]
    probas_per_loss = np.stack(probas_per_loss, axis=0)
    #proba = np.max(probas_per_loss, axis=0)
    #proba /= np.sum(proba)
    return probas_per_loss

def compute_loss(iterator, predict_function, loss_function, disable_augmentation:bool=True, disable_channel_postprocessing:bool=False, workers:int=None, verbose:int=1):
    data_aug_param = iterator.disable_random_transforms(disable_augmentation, disable_channel_postprocessing)

    simple_iterator = SimpleIterator(iterator)
    batch_size = simple_iterator.batch_size
    n_batches = len(simple_iterator)
    @tf.function
    def compute_loss(x, y_true):
        y_pred = predict_function(x)
        n_samples = tf.shape(x)[0]
        def loss_fun(j): # compute loss per batch item in case loss is reduced along batch size
            y_t = [y_true[k][j:j+1] for k in range(len(y_true))]
            y_p = [y_pred[k][j:j+1] for k in range(len(y_pred))]
            return tf.stack(loss_function(y_t, y_p), 0)
        return tf.map_fn(loss_fun, elems=tf.range(n_samples), fn_output_signature=tf.float32, parallel_iterations=batch_size)

    losses = []
    if verbose>=1:
        print(f"Active learning: computing loss...")
        progbar = tf.keras.utils.Progbar(n_batches)
    if workers is None:
        workers = os.cpu_count()
    enq = tf.keras.utils.OrderedEnqueuer(simple_iterator, use_multiprocessing=True, shuffle=False)
    enq.start(workers=workers, max_queue_size=max(3, min(n_batches, int(workers * 1.5))))
    gen = enq.get()
    n_tiles = None
    for i in range(n_batches):
        x, y_true = next(gen)
        batch_loss = compute_loss(x, y_true)
        if x.shape[0] > batch_size or n_tiles is not None: # tiling: keep hardest tile per sample
            if n_tiles is None: # record n_tile which is constant but last batch may have fewer elements
                n_tiles = x.shape[0] // batch_size
            batch_loss = tf.reshape(batch_loss, shape=(n_tiles, -1, batch_loss.shape[1]))
            batch_loss = tf.reduce_max(batch_loss, axis=0, keepdims=False) # record hardest tile per sample
        losses.append(batch_loss)
        if verbose >= 1:
            progbar.update(i+1)
    enq.stop()
    if data_aug_param is not None:
        iterator.enable_random_transforms(data_aug_param)
    return tf.concat(losses, axis=0).numpy()

class SimpleIterator(IndexArrayIterator):
    def __init__(self, iterator, input_scaling_function=None):
        index_array = iterator._get_index_array(choice=False)
        super().__init__(len(index_array), iterator.batch_size, False, 0, incomplete_last_batch_mode=INCOMPLETE_LAST_BATCH_MODE[0], step_number = 0)
        self.set_allowed_indexes(index_array)
        self.iterator = iterator
        self.input_scaling_function = batchwise_inplace(input_scaling_function) if input_scaling_function is not None else None

    def _get_batches_of_transformed_samples(self, index_array):
        batch = self.iterator._get_batches_of_transformed_samples(index_array)
        if self.input_scaling_function is not None:
            if isinstance(batch, (list, tuple)):
                x, y = batch
                x = self.input_scaling_function(x)
                batch = x, y
            else:
                batch = self.input_scaling_function(batch)
        return batch
def batchwise_inplace(function):
    def fun(batch):
        for i in range(batch.shape[0]):
            batch[i] = function(batch[i])
        return batch
    return fun
