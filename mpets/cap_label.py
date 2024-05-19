import os
import random
from concurrent.futures import ThreadPoolExecutor
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
    with ThreadPoolExecutor(max_workers=5) as executor:
        for dirpath, dirnames, filenames in os.walk('./cap'):
            for file in filenames:
                # Submit the task to the thread pool
                executor.submit(start, file)


if __name__ == '__main__':
    main()