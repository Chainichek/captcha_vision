import os
import random
from threading import Thread

from python_rucaptcha.image_captcha import ImageCaptcha

API_KEY = "be495762071b761797447785ccf4b3d1"


def start(file):
    print("Processing", file)
    result = ImageCaptcha(rucaptcha_key=API_KEY,
                          ).captcha_handler(captcha_file=f"./cap/{file}")
    ERROR = f"ERROR_{random.randint(10000, 90000)}"

    # save result to file
    with open(f"./samples/{result.get('solution', {}).get('text', ERROR)}.png", 'wb') as f:
        temp = open(f"./cap/{file}", 'rb')
        f.write(temp.read())
    print(result)
    os.remove(f"./cap/{file}")


def main():
    for dir in os.walk('./cap'):
        for file in dir[2]:
            # add threading here
            Thread(target=start, args=(file,)).start()

