from pathlib import Path

import requests


def get_cap(i):
    url = 'http://cap.mpets.mobi/captcha?r=422'
    Path(f"{i}.jpg").write_bytes(requests.get(url).content)


for i in range(200):
    print(i)
    get_cap(i)
