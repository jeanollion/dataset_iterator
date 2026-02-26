import tensorflow as tf

class InferenceLayer:
    def __init__(self, *args, **kwargs):
        self.inference_mode = False
        super().__init__(*args, **kwargs)

