import os

from keras.src import layers
from keras.src.saving import load_model

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np

import tensorflow as tf


class CTCLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loss_fn = tf.keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        batch_length = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_length, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_length, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        return y_pred


# Загрузка модели
MODEL_PATH = "captcha_mpets3.keras"
model = load_model(MODEL_PATH, custom_objects={'CTCLayer': CTCLayer})

char_to_num = layers.StringLookup(
    vocabulary=list(['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']),
    num_oov_indices=0,
    mask_token=None
)

num_to_char = layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(),
    mask_token=None,
    invert=True
)


# Функции для подготовки данных
def encode_single_sample(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.io.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.transpose(image, perm=[1, 0, 2])
    label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
    return image, label


def process_dataset(X, y):
    images = []
    labels = []
    for img_path, label in zip(X, y):
        img, lbl = encode_single_sample(img_path, label)
        images.append(img.numpy())
        labels.append(lbl.numpy())
    return np.array(images), np.array(labels)


# Пример новых данных для распознавания
image1 = "../samples/137052.png"
new_image_paths = [image1]  # Укажите пути к новым изображениям
new_labels = ["137052"]  # Укажите ожидаемые метки для новых изображений

X_new_processed, y_new_processed = process_dataset(new_image_paths, new_labels)
X_new_input = {"Input": X_new_processed, "Label": y_new_processed}

# Предсказание
preds = model.predict(X_new_input)
input_length = np.ones(preds.shape[0]) * preds.shape[1]
results = tf.keras.backend.ctc_decode(preds, input_length=input_length, greedy=True)[0][0][:, :6]

# Декодирование результатов
pred_texts = []
for result in results:
    res = tf.strings.reduce_join(num_to_char(result + 1)).numpy().decode("utf-8")
    pred_texts.append(res)

# Вывод результатов
for img_path, expected, predicted in zip(new_image_paths, new_labels, pred_texts):
    print(f"Image: {img_path}")
    print(f"Expected: {expected}")
    print(f"Predicted: {predicted}")
    print()
