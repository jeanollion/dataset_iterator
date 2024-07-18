import gc
import time
import warnings
import os
import numpy as np
import tensorflow as tf
from .index_array_iterator import IndexArrayIterator, INCOMPLETE_LAST_BATCH_MODE
from .concat_iterator import ConcatIterator
from dataset_iterator.ordered_enqueuer_cf import OrderedEnqueuerCF
import threading

class HardSampleMiningCallback(tf.keras.callbacks.Callback):
    def __init__(self, iterator, target_iterator, predict_fun, metrics_fun, period:int=10, start_epoch:int=0, start_from_epoch:int=0, enrich_factor:float=10., quantile_max:float=0.99, quantile_min:float=None, disable_channel_postprocessing:bool=False, verbose:int=1):
        super().__init__()
        self.period = period
        self.start_epoch = start_epoch
        self.start_from_epoch = start_from_epoch
        self.iterator = iterator
        self.target_iterator = target_iterator
        self.predict_fun = predict_fun
        self.metrics_fun = metrics_fun
        self.enrich_factor = enrich_factor
        self.quantile_max = quantile_max
        self.quantile_min = quantile_min
        self.disable_channel_postprocessing = disable_channel_postprocessing
        self.verbose = verbose
        self.metric_idx = -1
        self.proba_per_metric = None
        self.n_metrics = 0
        self.data_aug_param = self.iterator.disable_random_transforms(True, self.disable_channel_postprocessing)
        try:
            self.iterator_params = self.iterator.enqueuer_init()
        except AttributeError:
            self.iterator_params = None
        iterator_list = self.iterator.iterators if isinstance(self.iterator, ConcatIterator) else [self.iterator]
        self.simple_iterator_list = [SimpleIterator(it) for it in iterator_list]
        self.n_batches = [len(it) for it in self.simple_iterator_list]
        self.batch_size = [it.get_batch_size() for it in iterator_list]

        self.wait_for_me_supplier = threading.Event()
        self.wait_for_me_supplier.clear()  # will be set to the main iterator
        self.wait_for_me_consumer = threading.Event()
        self.wait_for_me_consumer.clear()
        self.wait_for_me_consumer_hsm = threading.Event()
        self.wait_for_me_consumer_hsm.clear()
        self.enqueuer = None
        self.generator = None

    def set_enqueuer(self, enqueuer):
        self.enqueuer = enqueuer
        self.enqueuer.wait_for_me_supplier = self.wait_for_me_supplier
        self.enqueuer.wait_for_me_consumer = self.wait_for_me_consumer
        self.generator = self.enqueuer.get_wfm(self.wait_for_me_consumer_hsm)

    def close(self):
        if self.data_aug_param is not None:
            self.iterator.enable_random_transforms(self.data_aug_param)
        if self.iterator_params is not None:
            self.iterator.enequeuer_end(self.iterator_params)
        self.iterator.close()

    def need_compute(self, epoch):
        return epoch + 1 + self.start_epoch >= self.start_from_epoch and (self.period == 1 or (epoch + 1 + self.start_epoch - self.start_from_epoch) % self.period == 0)

    def initialize(self):
        if self.need_compute(-1):
            self.on_epoch_end(-1)
        else:
            if self.wait_for_me_supplier is not None:
                self.wait_for_me_supplier.set()  # unlock main generator supplier
                self.wait_for_me_consumer.set()  # unlock main generator consumer

    def on_epoch_begin(self, epoch, logs=None):
        if self.n_metrics > 1 or self.need_compute(epoch):
            self.wait_for_me_supplier.clear()  # will lock main generator at end of epoch

    def on_epoch_end(self, epoch, logs=None):
        if self.need_compute(epoch):
            metrics = self.compute_metrics()
            gc.collect()
            first = self.proba_per_metric is None
            self.proba_per_metric = get_index_probability(metrics, enrich_factor=self.enrich_factor, quantile_max=self.quantile_max, quantile_min=self.quantile_min, verbose=self.verbose)
            self.n_metrics = self.proba_per_metric.shape[0] if len(self.proba_per_metric.shape) == 2 else 1
            if first and self.n_metrics > self.period:
                warnings.warn(f"Hard sample mining period = {self.period} should be greater than metric number = {self.n_metrics}")
            #print("Hard sample mining metrics computed", flush=True)
        if self.proba_per_metric is not None and not self.wait_for_me_supplier.is_set():
            if len(self.proba_per_metric.shape) == 2:
                self.metric_idx = (self.metric_idx + 1) % self.n_metrics
                proba = self.proba_per_metric[self.metric_idx]
            else:
                proba = self.proba_per_metric
            self.target_iterator.set_index_probability(proba)
            self.wait_for_me_supplier.set()  # release lock

    def on_train_end(self, logs=None):
        self.close()

    def compute_metrics(self):
        metric_list = []
        if self.verbose >=1:
            print(f"Hard Sample Mining: computing metrics...", flush=True)
        if self.enqueuer is not None:
            main_sequence = self.enqueuer.iterator
            self.wait_for_me_consumer.clear()  # lock the main generator consumer
            #main_sequence.close()
        for i in range(len(self.simple_iterator_list)):
            # unlock temporarily the corresponding enqueuer so that it starts
            #print(f"compute metrics for iterator #{i}: start of loop", flush=True)
            if self.enqueuer is not None:
                #self.simple_iterator_list[i].open()
                self.enqueuer.iterator = self.simple_iterator_list[i]
                self.enqueuer.wait_for_me_supplier_relock = True # re-lock so that supplier stops at end of epoch (only one epoch for HSM)
                self.wait_for_me_supplier.set()
                self.wait_for_me_consumer_hsm.set()  # unlock hsm consumer
                gen = self.generator
            else:
                gen = self.simple_iterator_list[i]
            #print(f"compute metrics for iterator #{i} start computing", flush=True)
            compute_metrics_fun = get_compute_metrics_fun(self.predict_fun, self.metrics_fun)
            metrics = compute_metrics_loop(compute_metrics_fun, gen, self.batch_size[i], self.n_batches[i], self.verbose)
            #print(f"compute metrics for iterator #{i} metrics computed", flush=True)
            metric_list.append(metrics)
            #self.simple_iterator_list[i].close()
        if self.enqueuer is not None:
            #main_sequence.open()
            self.enqueuer.iterator = main_sequence
            self.wait_for_me_consumer_hsm.clear()  # lock the hsm consumer
            self.wait_for_me_consumer.set()  # unlock the main consumer
        return np.concatenate(metric_list, axis=0)


def get_index_probability_1d(metric, enrich_factor:float=10., quantile_min:float=0.01, quantile_max:float=None, max_power:int=10, power_accuracy:float=0.1, verbose:int=1):
    assert 0.5 > quantile_min >= 0, f"invalid min quantile: {quantile_min}"
    if 1. / enrich_factor < quantile_min: # incompatible enrich factor and quantile
        quantile_min = 1. / enrich_factor
        #print(f"modified quantile_min to : {quantile_min}")
    if quantile_max is None:
        quantile_max = 1 - quantile_min
    metric_quantiles = np.quantile(metric, [quantile_min, quantile_max])

    Nh = metric[metric <= metric_quantiles[0]].shape[0] # hard examples (low metric)
    Ne = metric[metric >= metric_quantiles[1]].shape[0] # easy examples (high metric)
    metric_sub = metric[(metric < metric_quantiles[1]) & (metric > metric_quantiles[0])]
    Nm = metric_sub.shape[0]
    S = np.sum( ((metric_sub - metric_quantiles[1]) / (metric_quantiles[0] - metric_quantiles[1])) )
    p_max = enrich_factor / metric.shape[0]
    p_min = (1 - p_max * (Nh + S)) / (Nm + Ne - S) if (Nm + Ne - S) != 0 else -1
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
    #if verbose > 1:
    #    print(f"metric proba range: [{np.min(proba) * metric.shape[0]}, {np.max(proba) * metric.shape[0]}] (target range: [{p_min}; {p_max}]) power: {power} sum: {p_sum} quantiles: [{quantile_min}; {quantile_max}]", flush=True)
    return proba


def get_index_probability(metrics, enrich_factor:float=10., quantile_max:float=0.99, quantile_min:float=None, verbose:int=1):
    if len(metrics.shape) == 1:
        return get_index_probability_1d(metrics, enrich_factor=enrich_factor, quantile_max=quantile_max, quantile_min=quantile_min, verbose=verbose)
    probas_per_metric = [get_index_probability_1d(metrics[:, i], enrich_factor=enrich_factor, quantile_max=quantile_max, quantile_min=quantile_min, verbose=verbose) for i in range(metrics.shape[1])]
    probas_per_metric = np.stack(probas_per_metric, axis=0)
    #return probas_per_metric
    proba = np.mean(probas_per_metric, axis=0)
    return proba / np.sum(proba)


def compute_metrics(iterator, predict_function, metrics_function, disable_augmentation:bool=True, disable_channel_postprocessing:bool=False, workers:int=None, verbose:int=1):
    if isinstance(iterator, ConcatIterator):
        metric_list = [compute_metrics(it, predict_function, metrics_function, disable_augmentation, disable_channel_postprocessing, workers, verbose) for it in iterator.iterators]
        return np.concatenate(metric_list, axis=0)
    iterator.open()
    data_aug_param = iterator.disable_random_transforms(disable_augmentation, disable_channel_postprocessing)
    simple_iterator = SimpleIterator(iterator)
    batch_size = iterator.get_batch_size()
    n_batches = len(simple_iterator)

    compute_metrics_fun = get_compute_metrics_fun(predict_function, metrics_function)
    if workers is None:
        workers = os.cpu_count()
    enq = OrderedEnqueuerCF(simple_iterator, single_epoch=True, shuffle=False)
    enq.start(workers=workers, max_queue_size=max(3, min(n_batches, workers)))
    gen = enq.get()
    if verbose >= 1:
        print(f"Hard Sample Mining: computing metrics...", flush=True)
    metrics = compute_metrics_loop(compute_metrics_fun, gen, batch_size, n_batches, verbose)
    enq.stop()
    if data_aug_param is not None:
        iterator.enable_random_transforms(data_aug_param)
    iterator.close()
    #for i in range(metrics.shape[1]):
    #    print(f"metric: range: [{np.min(metrics[:,i])}, {np.max(metrics[:,i])}] mean: {np.mean(metrics[:,i])}")
    return metrics


def compute_metrics_loop(compute_metrics_fun, gen, batch_size, n_batches, verbose):
    if verbose >= 1:
        progbar = tf.keras.utils.Progbar(n_batches)
    n_tiles = None
    metrics = []
    for i in range(n_batches):
        x, y_true = next(gen)
        batch_metrics = compute_metrics_fun(x, y_true)
        bs = x.shape[0]
        if bs > batch_size or n_tiles is not None:
            if n_tiles is None:  # record n_tile which is constant but last batch may have fewer elements
                n_tiles = bs // batch_size
            batch_metrics = tf.reshape(batch_metrics, shape=(n_tiles, -1, batch_metrics.shape[1]))
            batch_metrics = tf.reduce_max(batch_metrics, axis=0, keepdims=False)  # record hardest tile per sample
        metrics.append(batch_metrics)
        if verbose >= 1:
            progbar.update(i + 1)
    return tf.concat(metrics, axis=0).numpy()


def get_compute_metrics_fun(predict_function, metrics_function):
    @tf.function(reduce_retracing=True)
    def compute_metrics(x, y_true):
        y_pred = predict_function(x)
        return metrics_function(y_true, y_pred)

    return compute_metrics


class SimpleIterator(IndexArrayIterator):
    def __init__(self, iterator, input_scaling_function=None):
        index_array = iterator._get_index_array(choice=False)
        super().__init__(len(index_array), iterator.get_batch_size(), False, 0, incomplete_last_batch_mode=INCOMPLETE_LAST_BATCH_MODE[0], step_number=0)
        self.set_allowed_indexes(index_array)
        self.iterator = iterator
        self.input_scaling_function = batchwise_inplace(input_scaling_function) if input_scaling_function is not None else None

    def _get_batches_of_transformed_samples(self, index_array):
        batch = self.iterator._get_batches_of_transformed_samples(index_array)
        if isinstance(batch, (list, tuple)):
            x, y = batch
            batch = x, y
        else:
            if self.input_scaling_function is not None:
                x = self.input_scaling_function(batch)
            batch = x
        return batch

    def open(self):
        self.iterator.open()

    def close(self, force:bool=False):
        self.iterator.close(force)

    def enqueuer_init(self):
        try:
            return self.iterator.enqueuer_init()
        except AttributeError:
            return None

    def enqueuer_end(self, params):
        try:
            self.iterator.enequeuer_end(params)
        except AttributeError:
            pass

def batchwise_inplace(function):
    def fun(batch):
        for i in range(batch.shape[0]):
            batch[i] = function(batch[i])
        return batch
    return fun
