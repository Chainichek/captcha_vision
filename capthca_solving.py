import numpy as np
import tensorflow as tf
from PIL import Image
from keras import utils
from keras.src.ops import ctc_decode
from tensorflow.keras import layers, models

# Определение пользовательского слоя с поддержкой дополнительных аргументов
class CTCLayer(layers.Layer):
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

# Загрузка модели с указанием пользовательского слоя
model_path = "captcha_mpets.keras"
model = tf.keras.models.load_model(model_path, custom_objects={'CTCLayer': CTCLayer})

# Проверка структуры модели
model.summary()

# Загрузка изображения
image_path = "mpets/samples/865253.jpg"  # Замените на путь к вашему изображению
image = Image.open(image_path)

# Предобработка изображения
image = image.convert('L')  # Преобразование в оттенки серого
image = image.resize((50, 200))  #  # Преобразуем изображение в формат RGB
image_array = np.array(image)  # Преобразуем изображение в массив numpy
image_array = image_array / 255.0  # Нормализуем значения пикселей до диапазона [0, 1]

# Добавляем измерение для батча (если модель ожидает пакет изображений)
image_array = np.expand_dims(image_array, axis=0)

label_array = np.zeros((1, 6))

char_to_num = layers.StringLookup(
    vocabulary = list({'1', '2', '3', '4', '5', '6', '7', '8', '9', '0'}),
    num_oov_indices = 0,
    mask_token = None
)

num_to_char = layers.StringLookup(
    vocabulary = char_to_num.get_vocabulary(),
    mask_token = None,
    invert = True
)

# Передача изображения в модель для предсказания
predictions = model.predict([image_array, label_array])
# Получение результата
sequence_length = [image_array.shape[2]]  # Предполагается, что изображение представлено как (batch_size, height, width, channels)

# Декодирование предсказаний
decoded_result, _ = tf.keras.backend.ctc_decode(predictions, input_length=sequence_length)

pred_texts = []
for result in decoded_result:
    res = tf.strings.reduce_join(num_to_char(result[:, :6] + 1)).numpy().decode("utf-8")
    pred_texts.append(res)

print(pred_texts)