import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Thread

from python_rucaptcha.control import Control
from python_rucaptcha.core.enums import ServiceEnm
from python_rucaptcha.image_captcha import ImageCaptcha

API_KEY = "be495762071b761797447785ccf4b3d1"


def start(file):
    print("Processing", file)
    result = ImageCaptcha(rucaptcha_key=API_KEY,
                          service_type=ServiceEnm.RUCAPTCHA.value,
                          ).captcha_handler(captcha_file=f"./cap/{file}")
    cap_text: str = result.get("solution").get('text')

    if len(cap_text) != 6\
            or not cap_text.isdigit():
        resp = Control(rucaptcha_key=API_KEY).reportIncorrect(id=result.get('taskId'))
        print("Incorrect captcha", result.get('taskId'))
        os.remove(f"./cap/{file}")
        return

    # save result to file
    with open(f"./samples/{result.get('solution', {}).get('text')}.png", 'wb') as f:
        temp = open(f"./cap/{file}", 'rb')
        f.write(temp.read())
    os.remove(f"./cap/{file}")


def main():
    with ThreadPoolExecutor(max_workers=5) as executor:
        for dirpath, dirnames, filenames in os.walk('./cap'):
            for file in filenames:
                # Submit the task to the thread pool
                executor.submit(start, file)


if __name__ == '__main__':
    main()