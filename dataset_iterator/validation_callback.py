from typing import List
from .utils import is_keras_3
if not is_keras_3():
    from tensorflow.keras.callbacks import Callback
else:
    from keras.callbacks import Callback
from dataset_iterator.ordered_enqueuer_cf import OrderedEnqueuerCF
import threading

class ValidationCallback(Callback):
    def __init__(self, iterator, step_number:int, validation_freq:int=1, start_epoch:int = 0):
        super().__init__()
        self.iterator = iterator
        self.step_number=step_number
        self.validation_freq = validation_freq
        self.start_epoch = start_epoch

        self.train_enqueuer = None
        self.val_enqueuer = None
        self.current_epoch = -1
        self.request_lock_index = -1

    def set_enqueuer(self, val_enqueuer:OrderedEnqueuerCF, train_enqueuer:OrderedEnqueuerCF):
        self.val_enqueuer = val_enqueuer
        self.val_enqueuer.append_request_lock(True) # always lock at end of epoch -> validation lasts 1 epoch
        self.train_enqueuer = train_enqueuer
        self.val_enqueuer.wait_for_me_supplier.clear()
        self.request_lock_index = train_enqueuer.append_request_lock(False)

    def initialize(self):
        self.train_enqueuer.request_lock_list[self.request_lock_index] = self._should_eval(self.start_epoch)
        if not self.train_enqueuer.wait_for_me_supplier.is_set():
            self.train_enqueuer.wait_for_me_supplier.set()

    def start_test(self):
        #print(f"start test train enq is set={self.train_enqueuer.wait_for_me_supplier.is_set()}")
        self.train_enqueuer.supplying_end_signal.wait()
        self.val_enqueuer.wait_for_me_supplier.set()

    def stop_test(self):
        #print(f"test end", flush=True)
        eval_at_next_epoch = self._should_eval(self.current_epoch + 1)
        self.train_enqueuer.request_lock_list[self.request_lock_index] = False
        if not self.train_enqueuer.request_lock(): # no other agent had required lock -> unlock
            self.train_enqueuer.request_lock_list[self.request_lock_index] = eval_at_next_epoch
            #print(f"unlock train enqueuer", flush=True)
            self.train_enqueuer.wait_for_me_supplier.set()  # unlock train enqueuer
        self.train_enqueuer.request_lock_list[self.request_lock_index] = eval_at_next_epoch
        #print(f"stop test done.", flush=True)

    def on_test_end(self, logs=None):
        #print(f"test end.", flush=True)
        self.stop_test()

    def on_train_batch_end(self, batch, logs=None):
        if batch == self.step_number - 1 and self._should_eval(self.current_epoch): # the test phase cannot start if the validation enqueuer has not started before on_test_begin is called, because consumer needs to yield an item before start
            self.start_test()
            #print(f"release val lock at last train batch", flush = True)

    def _should_eval(self, epoch):
        epoch = epoch + 1
        if isinstance(self.validation_freq, int):
            return epoch % self.validation_freq == 0
        elif isinstance(self.validation_freq, list):
            return epoch in self.validation_freq
        else:
            raise ValueError(
                "Expected `validation_freq` to be a list or int. "
                f"Received: validation_freq={self.validation_freq} of the "
                f"type {type(self.validation_freq)}."
            )

    def on_epoch_begin(self, epoch, logs=None):
        #print(f"epoch {epoch} begins", flush=True)
        self.current_epoch = epoch

    def on_epoch_end(self, epoch, logs=None):
        #print(f"epoch {epoch} end", flush=True)
        if not self._should_eval(epoch) and self._should_eval(epoch + 1):
            self.train_enqueuer.supplying_signal.wait()
            self.train_enqueuer.request_lock_list[self.request_lock_index] = self._should_eval(epoch + 1) # in case test was not run on this epoch

