from typing import List

import tensorflow as tf
from tensorflow.keras.metrics import Metric

class EMANormalization(Metric):
    def __init__(self, num_losses: int, step_number: int, alpha: float = 0.95, name="ema_normalized_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_losses = num_losses
        self.alpha = alpha  # Epoch-level alpha
        self.step_number = float(step_number)
        self.step_alpha = 1 - (1 - alpha) / self.step_number


        # Track EMA for each loss. stateful variables among epochs
        self.ema_losses = self.add_weight(name="ema_losses", shape=(num_losses,), initializer="zeros")

        # Track accumulated normalized loss and sample weight
        self.normalized_loss_sum = self.add_weight(name="normalized_loss_sum", initializer="zeros")
        self.sample_weight_sum = self.add_weight(name="sample_weight_sum", initializer="zeros")

    def update_state(self, losses, sample_weight=None, val: bool = False):
        # Cast inputs
        losses = tf.cast(losses, self._dtype)
        sample_weight = 1.0 if sample_weight is None else tf.cast(sample_weight, self._dtype)

        # Update EMA for each loss
        if not val: # only accumulated at train time: must be the same for training and validation
            self.ema_losses.assign( tf.cond(self.sample_weight_sum == 0, lambda:losses, lambda: self.step_alpha * self.ema_losses + (1 - self.step_alpha) * losses) )

        # Compute normalized loss for this batch
        normalized_loss = tf.reduce_mean(losses / (self.ema_losses + tf.keras.backend.epsilon()))

        # Accumulate normalized loss and sample weight
        self.normalized_loss_sum.assign_add(normalized_loss * sample_weight)
        self.sample_weight_sum.assign_add(sample_weight)

    def merge_state(self, metrics):
        weighted_ema_sum = self.ema_losses * self.sample_weight_sum
        total_weight = self.sample_weight_sum

        # Merge EMAs as a weighted average (approximation)
        for m in metrics:
            total_weight += m.sample_weight_sum
            weighted_ema_sum += m.ema_losses * m.sample_weight_sum

        if weighted_ema_sum > 0:
            self.ema_losses.assign( tf.math.divide_no_nan(weighted_ema_sum, total_weight) )

        # Accumulate normalized loss and sample weight
        for m in metrics:
            self.normalized_loss_sum.assign_add(m.normalized_loss_sum)
            self.sample_weight_sum.assign_add(m.sample_weight_sum)

    def result(self):
        return tf.math.divide_no_nan(self.normalized_loss_sum, self.sample_weight_sum)

    def reset_state(self):
        self.normalized_loss_sum.assign(0.0)
        self.sample_weight_sum.assign(0.0)
        # do not reset EMA or step counter as it should be kept between epochs
