import gc
import time
import warnings
import os
import numpy as np
import tensorflow as tf
from .utils import is_keras_3
if not is_keras_3():
    from tensorflow.keras.callbacks import Callback
    from tensorflow.keras.utils import Progbar
else:
    from keras.callbacks import Callback
    from keras.utils import Progbar
from .index_array_iterator import IndexArrayIterator, INCOMPLETE_LAST_BATCH_MODE
from .concat_iterator import ConcatIterator
from dataset_iterator.ordered_enqueuer_cf import OrderedEnqueuerCF
import threading

class HardSampleMiningCallback(Callback):
    def __init__(self, iterator, target_iterator, predict_fun, metrics_fun, period:int=10, start_epoch:int=0, start_from_epoch:int=0, enrich_factor:float=10., quantile_max:float=0.99, quantile_min:float=None, verbose:int=1):
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
        self.verbose = verbose
        self.metric_idx = -1
        self.proba_per_metric = None
        self.n_tiles = None
        self.n_metrics = 0
        try:
            self.iterator_params = self.iterator.enqueuer_init()
        except AttributeError:
            self.iterator_params = None
        iterator_list = self.iterator.iterators if isinstance(self.iterator, ConcatIterator) else [self.iterator]
        self.simple_iterator_list = [SimpleIterator(it) for it in iterator_list]
        self.n_batches = [len(it) for it in self.simple_iterator_list]
        self.batch_size = [it.get_batch_size() for it in iterator_list]

        self.wait_for_me_consumer_hsm = threading.Event()
        self.wait_for_me_consumer_hsm.clear()
        self.enqueuer = None
        self.generator = None
        self.request_lock_index = -1

    def set_enqueuer(self, enqueuer:OrderedEnqueuerCF):
        self.enqueuer = enqueuer
        self.generator = self.enqueuer.get_wfm(self.wait_for_me_consumer_hsm)
        self.enqueuer.wait_for_me_supplier.clear()
        self.enqueuer.wait_for_me_consumer.clear()
        self.request_lock_index = self.enqueuer.append_request_lock(False)

    def close(self):
        if self.iterator_params is not None:
            self.iterator.enqueuer_end(self.iterator_params)
        self.iterator.close()

    def need_compute(self, epoch):
        return epoch + 1 >= self.start_from_epoch and (self.period == 1 or (epoch + 1 - self.start_from_epoch) % self.period == 0)

    def initialize(self):
        if self.need_compute(self.start_epoch-1):
            self.on_epoch_end(self.start_epoch-1)
        elif self.start_epoch >= self.start_from_epoch: # force compute & set metrics
            self.set_metrics()
            self.on_epoch_end(self.start_epoch-1)
        if self.n_metrics == 0: # metrics not computed during initialization, so main enqueuer may not be already unlocked: unlock main enqueuer that was locked at set_enqueuer
            self.enqueuer.request_lock_list[self.request_lock_index] = self.need_compute(self.start_epoch)
            self.enqueuer.wait_for_me_consumer.set()
            if not self.enqueuer.wait_for_me_supplier.is_set():
                self.enqueuer.wait_for_me_supplier.set()
            if self.need_compute(self.start_epoch + 1):  # request lock for next epoch
                #print("HSM: requesting lock for next epoch, waiting...", flush=True)
                self.enqueuer.supplying_signal.wait()
                #print("HSM: requesting lock for next epoch, waiting done", flush=True)
                self.enqueuer.request_lock_list[self.request_lock_index] = True

    def on_epoch_end(self, epoch, logs=None):
        if self.need_compute(epoch):
            self.set_metrics()
            #print("Hard sample mining metrics computed", flush=True)
        if not self.enqueuer.request_lock_list[self.request_lock_index]:
            self.enqueuer.request_lock_list[self.request_lock_index] = self.proba_per_metric is not None # flag need lock
        if self.proba_per_metric is not None:
            #if not self.enqueuer.supplying_end_signal.is_set():
            #    print(f"waiting supplier signal end...", flush=True)
            self.enqueuer.supplying_end_signal.wait()
            self.metric_idx = (self.metric_idx + 1) % self.n_metrics
            proba = self.proba_per_metric[self.metric_idx]
            #print(f"setting proba for metrics: {self.metric_idx+1}/{self.n_metrics}", flush=True)
            self.target_iterator.set_index_probability(proba, n_tiles = self.n_tiles)
            self.enqueuer.wait_for_me_supplier.set()
            #print(f"HSM: main enqueuer unlocked", flush=True)
        elif self.need_compute(epoch+1): # request lock for next epoch
            if not self.enqueuer.wait_for_me_supplier.is_set():
                self.enqueuer.wait_for_me_supplier.set()  # release lock
            #print("HSM: requesting lock for next epoch, waiting...", flush=True)
            self.enqueuer.supplying_signal.wait()
            #print("HSM: requesting lock for next epoch, waiting done", flush=True)
            self.enqueuer.request_lock_list[self.request_lock_index] = True

    def on_train_end(self, logs=None):
        self.close()

    def set_metrics(self):
        metrics, n_tiles = self.compute_metrics()  # [ B, M or B x T, M ] , T can differ between iterators
        gc.collect()
        first = self.proba_per_metric is None
        if first:
            for i in range(metrics.shape[1]):
                print(f"metric: range: [{np.min(metrics[:,i])}, {np.max(metrics[:,i])}] mean: {np.mean(metrics[:,i])}")
        self.proba_per_metric = get_index_probability(metrics, enrich_factor=self.enrich_factor,  quantile_max=self.quantile_max, quantile_min=self.quantile_min, verbose=self.verbose)
        self.n_metrics = self.proba_per_metric.shape[0]
        if first:
            for i in range(self.n_metrics):
                print( f"proba for metric: {i + 1}/{self.n_metrics}: range: [{np.min(self.proba_per_metric[i])}; {np.max(self.proba_per_metric[i])}]: NA count: {np.sum(np.isnan(self.proba_per_metric[i]))}")
        if first and self.n_metrics > self.period:
            warnings.warn(  f"Hard sample mining period = {self.period} should be greater than metric number = {self.n_metrics}")
        self.n_tiles = n_tiles

    def compute_metrics(self):
        metric_list = []
        tile_list = []
        if self.verbose >=1:
            print(f"Hard Sample Mining: computing metrics...", flush=True)
        if self.enqueuer is not None:
            self.enqueuer.supplying_end_signal.wait()
            main_sequence = self.enqueuer.iterator
            self.enqueuer.wait_for_me_consumer.clear()  # lock the main generator consumer
            self.enqueuer.request_lock_list[self.request_lock_index] = True # will lock at each epoch (one epoch per iterator)
            self.wait_for_me_consumer_hsm.set()  # unlock hsm consumer
        for i in range(len(self.simple_iterator_list)):
            # unlock temporarily the corresponding enqueuer so that it starts
            #print(f"compute metrics for iterator #{i}: start of loop", flush=True)
            if self.enqueuer is not None: # reuse the same enqueur -> set the iterator
                if not self.enqueuer.supplying_end_signal.is_set():
                    #print(f"HSM: waiting supplying end signal (compute metrics @{i})", flush=True)
                    self.enqueuer.supplying_end_signal.wait()
                self.enqueuer.iterator = self.simple_iterator_list[i]
                self.enqueuer.wait_for_me_supplier.set()
                #print(f"HSM: supplier unlock (compute metrics @{i})", flush=True)
                #print(f"hsm consumer unlock", flush=True)
                gen = self.generator
            else:
                gen = self.simple_iterator_list[i]
            #print(f"HSM: compute metrics for iterator #{i} start computing", flush=True)
            compute_metrics_fun = get_compute_metrics_fun(self.predict_fun, self.metrics_fun)
            metrics, n_tiles = compute_metrics_loop(compute_metrics_fun, gen, self.batch_size[i], self.n_batches[i], self.verbose) # B, M or B, T, M
            #print(f"HSM: iterator {i}: {metrics.shape[1]} computed metrics on {metrics.shape[0]//n_tiles} samples x {n_tiles} tiles", flush=True)
            metric_list.append(metrics)
            tile_list.append(n_tiles)
        if self.enqueuer is not None:
            self.wait_for_me_consumer_hsm.clear()  # lock the hsm consumer
            if not self.enqueuer.supplying_end_signal.is_set():
                  #print(f"HSM: end of metric computation: waiting for end of supplyer...", flush=True)
                self.enqueuer.supplying_end_signal.wait()
            #print(f"HSM: end of metric computation: reset main enqueuer", flush=True)
            self.enqueuer.iterator = main_sequence
            self.enqueuer.wait_for_me_consumer.set()  # unlock the main consumer
        return np.concatenate(metric_list, axis=0), tile_list


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
    proba = proba / float(np.sum(proba))
    #if verbose > 1:
    #    print(f"metric proba range: [{np.min(proba) * metric.shape[0]}, {np.max(proba) * metric.shape[0]}] (target range: [{p_min}; {p_max}]) power: {power} sum: {p_sum} quantiles: [{quantile_min}; {quantile_max}]", flush=True)
    return proba


def get_index_probability(metrics, enrich_factor:float=10., quantile_max:float=0.99, quantile_min:float=None, verbose:int=1): # (Samples, Metrics) or (Samples,)
    if len(metrics.shape) == 1:
        return get_index_probability_1d(metrics, enrich_factor=enrich_factor, quantile_max=quantile_max, quantile_min=quantile_min, verbose=verbose)[np.newaxis]
    probas_per_metric = [get_index_probability_1d(metrics[:, i], enrich_factor=enrich_factor, quantile_max=quantile_max, quantile_min=quantile_min, verbose=verbose) for i in range(metrics.shape[1])]
    probas_per_metric = np.stack(probas_per_metric, axis=0) # (Metrics, Samples)
    return probas_per_metric


def compute_metrics(iterator, predict_function, metrics_function, disable_augmentation:bool=True, disable_channel_postprocessing:bool=False, workers:int=None, verbose:int=1):
    if isinstance(iterator, ConcatIterator):
        metric_list = []
        n_tile_list = []
        bs_list = []
        for it in iterator.iterators:
            metrics, (bs, n_tiles) = compute_metrics(it, predict_function, metrics_function, disable_augmentation, disable_channel_postprocessing, workers, verbose)
            metric_list.append(metrics)
            n_tile_list.append(n_tiles)
            bs_list.append(bs)
        return np.concatenate(metric_list, axis=0), n_tile_list
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
    metrics, n_tiles = compute_metrics_loop(compute_metrics_fun, gen, batch_size, n_batches, verbose) # B, M or B, T, M
    enq.stop()
    if data_aug_param is not None:
        iterator.enable_random_transforms(data_aug_param)
    iterator.close()
    #for i in range(metrics.shape[1]):
    #    print(f"metric: range: [{np.min(metrics[:,i])}, {np.max(metrics[:,i])}] mean: {np.mean(metrics[:,i])}")
    #print(f"{metrics.shape[1]} metrics computed on {metrics.shape[0]//n_tiles} samples x {n_tiles} tiles", flush=True)
    return metrics, (metrics.shape[0]//n_tiles, n_tiles)


def compute_metrics_loop(compute_metrics_fun, gen, batch_size, n_batches, verbose):
    if verbose >= 1:
        progbar = Progbar(n_batches)
    n_tiles = None
    metrics = []
    for i in range(n_batches):
        x, y_true = next(gen)
        batch_metrics = compute_metrics_fun(x, y_true)
        bs = x[0].shape[0] if isinstance(x, (tuple, list)) else x.shape[0]
        if bs > batch_size or n_tiles is not None:
            if n_tiles is None:  # record n_tile which is constant but last batch may have fewer batch_items
                n_tiles = bs // batch_size
            batch_metrics = tf.reshape(batch_metrics, shape=(n_tiles, -1, batch_metrics.shape[1]))
        metrics.append(batch_metrics)
        if verbose >= 1:
            progbar.update(i + 1)
    if n_tiles is None:
        return tf.concat(metrics, axis=0).numpy(), 1 # B, M
    else:
        res = tf.concat(metrics, axis=1).numpy() # T, B, M
        res = np.transpose(res, [1, 0, 2]) # B, T, M
        return np.reshape(res, (-1, res.shape[-1])), n_tiles # [B * T, M], T

def get_compute_metrics_fun(predict_function, metrics_function):
    @tf.function(reduce_retracing=True)
    def compute_metrics(x, y_true):
        y_pred = predict_function(x)
        return metrics_function(y_true, y_pred)

    return compute_metrics


class SimpleIterator(IndexArrayIterator):
    def __init__(self, iterator, input_scaling_function=None):
        index_array = iterator._get_index_array(choice=False)
        self.iterator = iterator
        super().__init__(len(index_array), iterator.get_batch_size(), False, 0, incomplete_last_batch_mode=INCOMPLETE_LAST_BATCH_MODE[0], step_number=0)
        self.set_allowed_indexes(index_array)
        self.input_scaling_function = batchwise_inplace(input_scaling_function) if input_scaling_function is not None else None

    def _get_batches_of_transformed_samples(self, index_array):
        batch = self.iterator._get_batches_of_transformed_samples(index_array)
        if len(batch)==1 and self.input_scaling_function is not None:
            return self.input_scaling_function(batch[0]),
        else:
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
            self.iterator.enqueuer_end(params)
        except AttributeError:
            pass

def batchwise_inplace(function):
    def fun(batch):
        for i in range(batch.shape[0]):
            batch[i] = function(batch[i])
        return batch
    return fun
