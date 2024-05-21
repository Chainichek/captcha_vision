import numpy as np
import tensorflow as tf
from PIL import Image
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
# image_path = "mpets/cap/7.jpg"
image_path = "mp/samples/111121.png"
image = Image.open(image_path)

# Предобработка изображения
# image = image.convert('L')  # Преобразование в оттенки серого
# image = image.resize((200, 50))  # Изменение размера изображения

image_array = np.array(image)

# Нормализуем значения пикселей до диапазона [0, 1]
image_array = image_array / 255.0

# Добавляем ось каналов для совместимости с TensorFlow (для черно-белого изображения)
image_array = np.expand_dims(image_array, axis=-1)

# Преобразуем массив numpy в тензор TensorFlow
image_tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)

# Транспонируем тензор изображения
image_tensor = tf.transpose(image_tensor, perm=[1, 0, 2, 3])

# Добавляем батч размер для соответствия форме входных данных модели
image_tensor = tf.expand_dims(image_tensor, axis=0)

# Метки (заглушка, т.к. мы предсказываем)
label_array = np.zeros((1, 6))  # Предположим, что длина метки равна 6


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
predictions = model.predict([image_tensor, label_array])
# Получение результата


sequence_length = [37]
# Декодирование предсказаний
decoded_result, _ = tf.keras.backend.ctc_decode(predictions, input_length=sequence_length)

pred_texts = []
for result in decoded_result:
    res = tf.strings.reduce_join(num_to_char(result[:, :6] + 1)).numpy().decode("utf-8")
    pred_texts.append(res)

print(pred_texts)