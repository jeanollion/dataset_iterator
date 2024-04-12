import warnings
import os
import numpy as np
import tensorflow as tf
from .index_array_iterator import IndexArrayIterator, INCOMPLETE_LAST_BATCH_MODE

class HardSampleMiningCallback(tf.keras.callbacks.Callback):
    def __init__(self, iterator, target_iterator, predict_fun, metrics_fun, period:int=10, start_epoch:int=0, skip_first:bool=False, enrich_factor:float=10., quantile_max:float=0.99, quantile_min:float=None, disable_channel_postprocessing:bool=False, workers=None, verbose:int=1):
        super().__init__()
        self.period = period
        self.start_epoch = start_epoch
        self.skip_first = skip_first
        self.iterator = iterator
        self.target_iterator = target_iterator
        self.predict_fun = predict_fun
        self.metrics_fun = metrics_fun
        self.enrich_factor = enrich_factor
        self.quantile_max = quantile_max
        self.quantile_min = quantile_min
        self.disable_channel_postprocessing = disable_channel_postprocessing
        self.workers = workers
        self.verbose = verbose
        self.metric_idx = -1
        self.proba_per_metric = None
        self.n_metrics = 0
        self.data_aug_param = self.iterator.disable_random_transforms(True, self.disable_channel_postprocessing)
        simple_iterator = SimpleIterator(self.iterator)
        self.batch_size = self.iterator.get_batch_size()
        self.n_batches = len(simple_iterator)
        self.enq = tf.keras.utils.OrderedEnqueuer(simple_iterator, use_multiprocessing=True, shuffle=False)

    def close(self):
        self.enq.stop()
        if self.data_aug_param is not None:
            self.iterator.enable_random_transforms(self.data_aug_param)
        self.iterator.close()

    def on_epoch_begin(self, epoch, logs=None):
        if self.period==1 or (epoch + self.start_epoch) % self.period == 0:
            if epoch == 0 and self.skip_first:
                return
            metrics = self.compute_metrics()
            first = self.proba_per_metric is None
            self.proba_per_metric = get_index_probability(metrics, enrich_factor=self.enrich_factor, quantile_max=self.quantile_max, quantile_min=self.quantile_min, verbose=self.verbose)
            self.n_metrics = self.proba_per_metric.shape[0] if len(self.proba_per_metric.shape) == 2 else 1
            if first and self.n_metrics > self.period:
                warnings.warn(f"Hard sample mining period = {self.period} should be greater than metric number = {self.n_metrics}")
        if self.proba_per_metric is not None:
            if len(self.proba_per_metric.shape) == 2:
                self.metric_idx = (self.metric_idx + 1) % self.n_metrics
                proba = self.proba_per_metric[self.metric_idx]
            else:
                proba = self.proba_per_metric
            # set probability to iterator in case of multiprocessing iwth OrderedEnqueeur this will be taken into account only a next epoch has iterator has already been sent to processes at this stage
            self.target_iterator.set_index_probability(proba)

    def on_train_end(self, logs=None):
        self.close()

    def compute_metrics(self):
        workers = os.cpu_count() if self.workers is None else self.workers
        self.enq.start(workers=workers, max_queue_size=max(3, min(self.n_batches, workers)))
        gen = self.enq.get()
        compute_metrics_fun = get_compute_metrics_fun(self.predict_fun, self.metrics_fun, self.batch_size)
        metrics = compute_metrics_loop(compute_metrics_fun, gen, self.batch_size, self.n_batches, self.verbose)
        self.enq.stop()
        return metrics


def get_index_probability_1d(metric, enrich_factor:float=10., quantile_min:float=0.01, quantile_max:float=None, max_power:int=10, power_accuracy:float=0.1, verbose:int=1):
    assert 0.5 > quantile_min >= 0, f"invalid min quantile: {quantile_min}"
    if quantile_max is None:
        quantile_max = 1 - quantile_min
    assert 1 >= quantile_max > 0.5, f"invalid max quantile: {quantile_max}"
    if 1. / enrich_factor < (1 - quantile_max): # incompatible enrich factor and quantile
        quantile_max = 1. - 1 / enrich_factor
        #print(f"modified quantile_max to : {quantile_max}")
    metric_quantiles = np.quantile(metric, [quantile_min, quantile_max])

    Nh = metric[metric <= metric_quantiles[0]].shape[0] # hard examples (low metric)
    Ne = metric[metric >= metric_quantiles[1]].shape[0] # easy examples (high metric)
    metric_sub = metric[(metric < metric_quantiles[1]) & (metric > metric_quantiles[0])]
    Nm = metric_sub.shape[0]
    S = np.sum( ((metric_sub - metric_quantiles[1]) / (metric_quantiles[0] - metric_quantiles[1])) )
    p_max = enrich_factor / metric.shape[0]
    p_min = (1 - p_max * (Nh + S)) / (Nm + Ne - S) if (Nm + Ne - S)!=0 else -1
    if p_min<0:
        p_min = 0.
        target = 1./p_max - Nh
        if target <= 0: # cannot reach enrich factor: too many hard examples
            power = max_power
        else:
            fun = lambda power_: np.sum(((metric_sub - metric_quantiles[1]) / (metric_quantiles[0] - metric_quantiles[1])) ** power_)
            power = 1
            Sn = S
            while power < max_power and Sn > target:
                power += power_accuracy
                Sn = fun(power)
            if power > 1 and Sn < target:
                power -= power_accuracy
    else:
        power = 1
    #print(f"p_min {p_min} ({(1 - p_max * (Nh + S)) / (Nm + Ne - S)}) Nh: {Nh} nE: {Ne} Nm: {Nm} S: {S} pmax: {p_max} power: {power}")
    # drop factor at min quantile, enrich factor at max quantile, interpolation in between
    def get_proba(value):
        if value <= metric_quantiles[0]:
            return p_max
        elif value >= metric_quantiles[1]:
            return p_min
        else:
            return p_min + (p_max - p_min) * ((value - metric_quantiles[1]) / (metric_quantiles[0] - metric_quantiles[1]))**power

    vget_proba = np.vectorize(get_proba)
    proba = vget_proba(metric)
    p_sum = float(np.sum(proba))
    proba /= p_sum
    if verbose > 1:
        print(f"metric proba range: [{np.min(proba) * metric.shape[0]}, {np.max(proba) * metric.shape[0]}] (target range: [{p_min}; {p_max}]) power: {power} sum: {p_sum} quantiles: [{quantile_min}; {quantile_max}]", flush=True)
    return proba

def get_index_probability(metrics, enrich_factor:float=10., quantile_max:float=0.99, quantile_min:float=None, verbose:int=1):
    if len(metrics.shape) == 1:
        return get_index_probability_1d(metrics, enrich_factor=enrich_factor, quantile_max=quantile_max, quantile_min=quantile_min, verbose=verbose)
    probas_per_metric = [get_index_probability_1d(metrics[:, i], enrich_factor=enrich_factor, quantile_max=quantile_max, quantile_min=quantile_min, verbose=verbose) for i in range(metrics.shape[1])]
    probas_per_metric = np.stack(probas_per_metric, axis=0)
    #proba = np.max(probas_per_metric, axis=0)
    #proba /= np.sum(proba)
    return probas_per_metric

def compute_metrics(iterator, predict_function, metrics_function, disable_augmentation:bool=True, disable_channel_postprocessing:bool=False, workers:int=None, verbose:int=1):
    data_aug_param = iterator.disable_random_transforms(disable_augmentation, disable_channel_postprocessing)
    simple_iterator = SimpleIterator(iterator)
    batch_size = iterator.get_batch_size()
    n_batches = len(simple_iterator)

    compute_metrics_fun = get_compute_metrics_fun(predict_function, metrics_function, batch_size)

    if workers is None:
        workers = os.cpu_count()
    enq = tf.keras.utils.OrderedEnqueuer(simple_iterator, use_multiprocessing=True, shuffle=False)
    enq.start(workers=workers, max_queue_size=max(3, min(n_batches, workers)))
    gen = enq.get()
    metrics = compute_metrics_loop(compute_metrics_fun, gen, batch_size, n_batches, verbose)
    enq.stop()
    if data_aug_param is not None:
        iterator.enable_random_transforms(data_aug_param)
    iterator.close()
    return metrics

def compute_metrics_loop(compute_metrics_fun, gen, batch_size, n_batches, verbose):
    metrics = []
    indices = []
    if verbose>=1:
        print(f"Hard Sample Mining: computing metrics...")
        progbar = tf.keras.utils.Progbar(n_batches)
    n_tiles = None
    for i in range(n_batches):
        current_indices, x, y_true = next(gen)
        batch_metrics = compute_metrics_fun(x, y_true)
        if x.shape[0] > batch_size or n_tiles is not None:  # tiling: keep hardest tile per sample
            if n_tiles is None:  # record n_tile which is constant but last batch may have fewer elements
                n_tiles = x.shape[0] // batch_size
            batch_metrics = tf.reshape(batch_metrics, shape=(n_tiles, -1, batch_metrics.shape[1]))
            batch_metrics = tf.reduce_max(batch_metrics, axis=0, keepdims=False)  # record hardest tile per sample
        metrics.append(batch_metrics)
        indices.append(current_indices)
        if verbose >= 1:
            progbar.update(i + 1)
    if verbose >= 1:
        print("metrics computed", flush=True)
    indices = tf.concat(indices, axis=0).numpy()
    metrics = tf.concat(metrics, axis=0).numpy()
    if len(metrics.shape) == 1:
        metrics = indices[:, np.newaxis]
    # metrics = tf.concat([indices[:, np.newaxis].astype(metrics.dtype), metrics], axis=1)
    if not np.all(indices[1:] - indices[:-1] == 1):
        indices = np.argsort(indices)
        metrics = metrics[indices, :]
        if verbose >= 1:
            print("some indices where not in order!!", flush=True)
    return metrics
def get_compute_metrics_fun(predict_function, metrics_function, batch_size):
    @tf.function
    def compute_metrics(x, y_true):
        y_pred = predict_function(x)
        n_samples = tf.shape(x)[0]
        def metrics_fun(j): # compute metric per batch item in case metric is reduced along batch size
            y_t = [y_true[k][j:j+1] for k in range(len(y_true))]
            y_p = [y_pred[k][j:j+1] for k in range(len(y_pred))]
            metrics = metrics_function(y_t, y_p)
            return tf.stack(metrics, 0)
        return tf.map_fn(metrics_fun, elems=tf.range(n_samples), fn_output_signature=tf.float32, parallel_iterations=batch_size)

    return compute_metrics

class SimpleIterator(IndexArrayIterator):
    def __init__(self, iterator, input_scaling_function=None):
        index_array = iterator._get_index_array(choice=False)
        super().__init__(len(index_array), iterator.batch_size, False, 0, incomplete_last_batch_mode=INCOMPLETE_LAST_BATCH_MODE[0], step_number = 0)
        self.set_allowed_indexes(index_array)
        self.iterator = iterator
        self.input_scaling_function = batchwise_inplace(input_scaling_function) if input_scaling_function is not None else None

    def _get_batches_of_transformed_samples(self, index_array):
        batch = self.iterator._get_batches_of_transformed_samples(index_array)
        if isinstance(batch, (list, tuple)):
            x, y = batch
            batch = index_array, x, y
        else:
            if self.input_scaling_function is not None:
                x = self.input_scaling_function(batch)
            batch = index_array, x
        return batch

def batchwise_inplace(function):
    def fun(batch):
        for i in range(batch.shape[0]):
            batch[i] = function(batch[i])
        return batch
    return fun
