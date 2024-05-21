import asyncio
import os

import numpy as np
import requests
import tensorflow as tf
from PIL import Image
from python_rucaptcha.core.enums import ServiceEnm
from python_rucaptcha.image_captcha import ImageCaptcha

from learning import mpets
from utils.CTCLayer import CTCLayer
from tensorflow.keras import layers

# Путь до модели
MODEL_PATH = "cap1.keras"
model = tf.keras.models.load_model(MODEL_PATH, custom_objects={'CTCLayer': CTCLayer})
model.summary()


async def recognize_captcha(image_path: str) -> str:
    image = Image.open(image_path)
    image_array = np.array(image)
    image_array = image_array / 255.0  # Нормализуем значения пикселей до диапазона [0, 1]

    # Добавляем ось каналов для совместимости с TensorFlow (для черно-белого изображения)
    image_array = np.expand_dims(image_array, axis=-1)

    # Преобразуем массив numpy в тензор TensorFlow
    image_tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)

    # Транспонируем тензор изображения
    image_tensor = tf.transpose(image_tensor, perm=[1, 0, 2, 3])

    # Добавляем батч размер для соответствия форме входных данных модели
    image_tensor = tf.expand_dims(image_tensor, axis=0)

    # Метки (заглушка, т.к. мы предсказываем)
    label_array = np.zeros((1, 6))

    char_to_num = layers.StringLookup(
        vocabulary=list({'1', '2', '3', '4', '5', '6', '7', '8', '9', '0'}),
        num_oov_indices=0,
        mask_token=None
    )

    num_to_char = layers.StringLookup(
        vocabulary=char_to_num.get_vocabulary(),
        mask_token=None,
        invert=True
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

    # Возвращает строчку с кодом: "123456"
    return ''.join(pred_texts)


API_KEY = "be495762071b761797447785ccf4b3d1"


async def incorrect_captcha(task_id: str):
    params = {"key": API_KEY,
              "action": "reportbad",
              "id": task_id,
              "json": 1,
              "header_acao": "1"}
    resp = requests.post(
        f"https://rucaptcha.com/res.php", params=params)
    return resp


async def rucaptcha(file):
    result = await ImageCaptcha(rucaptcha_key=API_KEY,
                          service_type=ServiceEnm.RUCAPTCHA.value,
                          numeric=1,
                          minLength=6,
                          maxLength=6
                          ).aio_captcha_handler(captcha_file=f"{file}")
    print(result)
    cap_text: str = result.get("solution").get('text')

    if len(cap_text) != 6 \
            or not cap_text.isdigit():
        task_id = result.get('taskId')
        await incorrect_captcha(task_id=task_id)
        return None, None
    return cap_text, result.get('taskId')


total = 0
correct = 0


async def main():
    global total, correct
    for i in range(10):
        total += 1
        result = await mpets.get_captcha()
        image_path = result["captcha"]
        code = await recognize_captcha(image_path) # С помощью нейронки
        # code, task_id = await rucaptcha(image_path)
        if code is None:
            os.remove(image_path)
            continue
        captcha_result = await mpets.is_correct_captcha(result["cookie"], code)
        print(f"Распознанный код: {code} | Результат: {captcha_result}")

        if captcha_result:
            correct += 1
            with open(f"./new/{code}.png", 'wb') as f:
                temp = open(f"{image_path}", 'rb')
                f.write(temp.read())
        elif captcha_result is False:
            await incorrect_captcha(task_id=task_id)

        os.remove(image_path)
        print(f"Точность: {correct}/{total} = {correct / total}")

        await asyncio.sleep(1)


async def start():
    tasks = []
    for i in range(5):
        tasks.append(asyncio.create_task(main()))

    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(start())
