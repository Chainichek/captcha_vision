import keras
from keras.src.legacy import layers
from keras.src.legacy.backend import ctc_batch_cost
import tensorflow as tf


@keras.saving.register_keras_serializable()
class CTCLayer(layers.Layer):
    def __init__(self, trainable=False, dtype=None):
        super().__init__(name="ctc_loss", trainable=trainable, dtype=dtype)
        self.loss_fn = ctc_batch_cost

    def call(self, inputs):
        y_pred, y_true = inputs
        batch_length = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_length, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_length, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        return y_pred
