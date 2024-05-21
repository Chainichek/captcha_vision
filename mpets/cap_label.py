import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Thread

import requests
from python_rucaptcha.control import Control
from python_rucaptcha.core.enums import ServiceEnm
from python_rucaptcha.image_captcha import ImageCaptcha

API_KEY = "be495762071b761797447785ccf4b3d1"


def start(i):
    file = f"{i}.jpg"
    print("Processing", file)
    url = 'http://cap.mpets.mobi/captcha?r=422'
    Path(f"./cap/{i}.jpg").write_bytes(requests.get(url).content)
    result = ImageCaptcha(rucaptcha_key=API_KEY,
                          service_type=ServiceEnm.RUCAPTCHA.value,
                          numeric=1,
                          minLength=6,
                          maxLength=6
                          ).captcha_handler(captcha_file=f"./cap/{file}",

                                            )
    print(result)
    cap_text: str = result.get("solution").get('text')

    if len(cap_text) != 6 \
            or not cap_text.isdigit():
        task_id = result.get('taskId')
        print("Incorrect captcha", task_id)
        params = {"key": API_KEY,
                  "action": "reportbad",
                  "id": task_id,
                  "json": 1,
                  "header_acao": "1"}
        resp = requests.post(
            f"https://rucaptcha.com/res.php", params=params)
        print(resp.json())
        os.remove(f"./cap/{file}")
        return

    # save result to file
    with open(f"./samples/{result.get('solution', {}).get('text')}.png", 'wb') as f:
        temp = open(f"./cap/{file}", 'rb')
        f.write(temp.read())
    os.remove(f"./cap/{file}")


def main():
    MAX_CAPS = 100
    with ThreadPoolExecutor(max_workers=50) as executor:
        for i in range(0, MAX_CAPS):
            # Submit the task to the thread pool
            executor.submit(start, i)


if __name__ == '__main__':
    main()
