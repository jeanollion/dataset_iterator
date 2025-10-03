from math import cos, pi
from .utils import is_keras_3
if not is_keras_3():
    from tensorflow.keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau
    from tensorflow.keras import backend
else:
    from keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau
    from keras import backend
import csv
import os
import time
import numpy as np

class SafeModelCheckpoint(ModelCheckpoint):
    def __init__(self,
                 filepath,
                 n_retry:int = 10,
                 sleep_time:int = 5,
                 **kwargs):
        super().__init__(filepath, **kwargs)
        self.n_retry = n_retry
        self.sleep_time = sleep_time
        self._alternative_path = False

    def _save_model(self, epoch, batch, logs):
        if logs is not None:
            loss = logs.get("loss")
            if loss is not None:
                if np.isnan(loss) or np.isinf(loss):
                    print(f"Invalid loss, skipping checkpoint.")
                    return
        if isinstance(self.save_freq, int) or self.epochs_since_last_save >= self.period:
            for i in range(self.n_retry):
                if i>1:
                    print(f"save_model: start of for loop: {i + 1}/{self.n_retry}")
                try:
                    self._alternative_path = False
                    path = self._get_file_path(epoch, batch, logs)
                    tmp_path = self._tmp_path(path)
                    self._alternative_path = True
                    super()._save_model(epoch, batch, logs) # save to tmp path
                    self._alternative_path = False
                    #print(f"save_model: try {i + 1}/{self.n_retry}, path: {path} temp path: {tmp_path}", flush=True)
                    if os.path.exists(tmp_path):  # move tmp file to new file
                        if os.path.exists(path):
                            os.remove(path)
                        os.rename(tmp_path, path)
                        #print(f"moved: {path} to {tmp_path}", flush=True)
                    if i == self.n_retry - 1:
                        self._alternative_path = True
                    super()._save_model(epoch, batch, logs)
                    #print(f"saved epoch: {epoch}", flush=True)
                except BlockingIOError as error:
                    print(f"Error saving weights: {error}. Retry: {i+1}/{self.n_retry}", flush=True)
                    time.sleep(self.sleep_time)  # wait for X seconds before trying to save again
                    print(f"slept: {self.sleep_time}s")
                    os.remove(self._get_file_path(epoch, batch, logs))
                    print(f"file {self._get_file_path(epoch, batch, logs)} removed")
                    self._alternative_path = False
                else:
                    self._alternative_path = False
                    tmp = os.path.exists(tmp_path)
                    main = os.path.exists(path)
                    if main and tmp:
                        os.remove(tmp_path)
                        return
                    elif main:
                        return
                    elif tmp and not main and i == self.n_retry-1:
                        os.rename(tmp_path, path)
                        print(f"model could not be saved at {self._get_file_path(epoch, batch, logs)} after {self.n_retry} retry, restoring temp file", flush=True)
                        return
                    time.sleep(self.sleep_time)
                print(f"save_model: end of for loop: {i+1}/{self.n_retry}", flush=True)

        # 21:45:07.667:   File "h5py/h5f.pyx", line 126, in h5py.h5f.create
        # 21:45:07.667: BlockingIOError: [Errno 11] Unable to create file (unable to lock file, errno = 11, error message = 'Resource temporarily unavailable')

    def _get_file_path(self, epoch, batch, logs):
        path = super()._get_file_path(epoch, batch, logs)
        if self._alternative_path:
            return self._tmp_path(path)
        return path

    def _tmp_path(self, path):
        split = os.path.splitext(path)
        return split[0] + "_tmp" + split[1]


class StopOnLR(Callback):
    def __init__( self, min_lr, **kwargs, ):
        super().__init__()
        self.min_lr = min_lr

    def on_epoch_end(self, epoch, logs=None):
        lr = backend.get_value(self.model.optimizer.lr)
        if lr <= self.min_lr:
            print(f"Learning rate {lr} <= {self.min_lr} : training will be stopped", flush=True)
            self.model.stop_training = True

class LogLRCallback(Callback):
    def __init__(self):
        super().__init__()

    def on_train_batch_end(self, batch, logs=None):
        if logs is not None:
            logs['lr'] = backend.get_value(self.model.optimizer.lr)


class EpsilonCosineDecayCallback(Callback):
    """Reduce optimizer epsilon parameter.
    Args:
      factor: factor by which epsilon will be reduced.
        `new_epsilon = epsilon * factor`.
      period: number of epochs after which epsilon will be reduced.
      verbose: int. 0: quiet, 1: update messages.
      min_epsilon: lower bound on the learning rate.
    """

    def __init__(self, decay_steps:int, start_epsilon=1, min_epsilon=1e-2, start_step:int=0, verbose=1):
        super(EpsilonCosineDecayCallback, self).__init__()
        self.step_counter = start_step
        self.start_epsilon = start_epsilon
        self.decay_steps=decay_steps
        self.alpha=min_epsilon / start_epsilon
        self.verbose = verbose

    def on_train_batch_begin(self, batch, logs=None):
        epsilon = self._decayed_epsilon(self.step_counter)
        self.model.optimizer.epsilon = epsilon

    def on_train_batch_end(self, batch, logs=None):
        self.step_counter +=1
        if logs is not None:
            logs['epsilon'] = self.model.optimizer.epsilon

    def _decayed_epsilon(self, step):
        step = min(step, self.decay_steps)
        cosine_decay = 0.5 * (1 + cos(pi * step / self.decay_steps))
        decayed = (1 - self.alpha) * cosine_decay + self.alpha
        return self.start_epsilon * decayed


class ReduceLROnPlateau2(ReduceLROnPlateau):
    """Small variation from Original: best value is reset each time LR is reduced.
    """

    def on_epoch_end(self, epoch, logs=None):
        old_lr = backend.get_value(self.model.optimizer.lr)
        super().on_epoch_end(epoch, logs)
        lr = backend.get_value(self.model.optimizer.lr)
        if lr < old_lr:
            self._reset()
            self.cooldown_counter = self.cooldown


class LogsCallback(Callback):
    def __init__(self, filepath, start_epoch=0):
        super().__init__()
        self.filepath = filepath
        self.start_epoch=start_epoch

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            epoch += self.start_epoch
            losses = list(logs.values())
            losses.insert(0, epoch)
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            with open(filepath, "a") as file_object:
                writer = csv.writer(file_object, delimiter=';',  quotechar='"', quoting=csv.QUOTE_MINIMAL)
                if os.path.getsize(filepath) == 0: # write header only if file is empty
                    headers = list(logs.keys())
                    headers.insert(0, "epoch")
                    writer.writerow(headers)
                writer.writerow(losses)
