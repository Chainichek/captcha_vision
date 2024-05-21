import time

import aiohttp

HOST_URL = "https://mpets.mobi"


async def get_captcha():
    """
        Возвращает имя капчи
    """
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(10),
                                     connector=None) as ses:
        async with ses.get(f'{HOST_URL}/captcha?r=300') as resp:
            for item in resp.cookies.items():
                cookie = str(item[1]).split("=")[1].split(";")[0]

            filename = f"{str(time.time())}.jpg"
            with open(filename, 'wb') as fd:
                fd.write(await resp.read())
            await ses.close()
            return {"status": True,
                    "captcha": filename,
                    "cookie": {"PHPSESSID": cookie}}


async def is_correct_captcha(cookies: dict,
                             code: str) -> bool | None:
    """
        Проверка капчи
    """
    NAME = "Keras"
    PASSWORD = "t3ns0r_my_c4ptch4"
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(10),
                                     cookies=cookies) as ses:
        data = {"name": NAME,
                "password": PASSWORD,
                "captcha": code}
        async with ses.post(f'{HOST_URL}/login', data=data) as resp:
            text = await resp.text()
            if "Магазин" in text:
                return True
            elif "Неверная captcha." in text:
                return False
    return None

