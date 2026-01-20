import tensorflow as tf

class InferenceLayer:
    def __init__(self, *args, **kwargs):
        self.inference_mode = False
        super().__init__(*args, **kwargs)

# for compat with tf2.7
class Identity(tf.keras.layers.Layer):
    def __init__(self, autocast=False, **kwargs):
        super().__init__(autocast=autocast **kwargs)

    def call(self, input, training=None):
        if not training:
            return input
        return tf.identity( input, name=self.name )
