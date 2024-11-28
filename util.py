import io
import numpy as np
import requests
import time
import tkinter as tk

from enum import IntEnum
from PIL import Image, ImageTk

class CaptchaType(IntEnum):
    PAYMENT = 0
    ZFW = 1

def input_captcha_type(prompt='Select captcha type: '):
    for type in CaptchaType:
        print(f'{type.value}: {type.name.lower()}')
    return CaptchaType(int(input(prompt)))

def get_captcha_image(type: CaptchaType):
    if type == CaptchaType.PAYMENT:
        rsp = requests.get('https://payment.xidian.edu.cn/NetWorkUI/authImage')
        return Image.open(io.BytesIO(rsp.content))

    if type == CaptchaType.ZFW:
        # 先刷新再获取
        requests.get('https://zfw.xidian.edu.cn/site/captcha', params={
            'refresh': 1,
            '_': int(round(time.time() * 1000)) # 毫秒级时间戳
        })
        rsp = requests.get('https://zfw.xidian.edu.cn/site/captcha') # zfw 是自服务的意思
        return Image.open(io.BytesIO(rsp.content))

    raise ValueError('Unknown captcha type')

def get_captcha_num_classes(type: CaptchaType):
    if type == CaptchaType.PAYMENT:
        return 9 # 只有数字 1-9
    return 10

def get_captcha_label_class(type: CaptchaType, label: int):
    if type == CaptchaType.PAYMENT:
        return label - 1
    return label

def get_captcha_class_label(type: CaptchaType, klass: int):
    if type == CaptchaType.PAYMENT:
        return klass + 1
    return klass

def get_captcha_model_path(type: CaptchaType, ext: str):
    return f'./model/captcha-solver-{type.name.lower()}.{ext}'

def get_captcha_dataset_dir(type: CaptchaType):
    return f'./data/{type.name.lower()}'

def captcha_image_to_array(type: CaptchaType, img: Image.Image):
    import keras # 加载比较耗时，所以放在函数内

    if type == CaptchaType.PAYMENT:
        return keras.utils.img_to_array(img.convert('L')) / 255.0
    return keras.utils.img_to_array(img) / 255.0 # 已经是灰度图

def preprocess_captcha_image(type: CaptchaType, img: Image.Image) -> Image.Image | None:
    if type != CaptchaType.ZFW:
        return img

    # 灰度
    img = img.convert('L')

    # 反色
    for i in range(img.width):
        for j in range(img.height):
            img.putpixel((i, j), 255 - img.getpixel((i, j)))

    bb = list(img.point(lambda p: 255 if p > 250 else 0).getbbox(alpha_only=False))

    # 向外扩展一个像素
    bb[0] = max(0, bb[0] - 1) # left
    bb[1] = max(0, bb[1] - 1) # upper
    bb[2] = min(img.width, bb[2] + 1) # right，不包含该点
    bb[3] = min(img.height, bb[3] + 1) # lower，不包含该点

    # 宽度太小，数字可能都挤在一起
    if bb[2] - bb[0] < 44:
        return None

    def lerp(a, b, t):
        return a + (b - a) * t

    def load(x, y):
        x = min(max(x, bb[0]), bb[2] - 1)
        y = min(max(y, bb[1]), bb[3] - 1)
        return img.getpixel((x, y))

    # 采样同时去除大部分噪声
    def sample(u, v):
        u = lerp(bb[0], bb[2] - 1, u)
        v = lerp(bb[1], bb[3] - 1, v)
        x, y = int(np.floor(u)), int(np.floor(v))
        return min(load(x, y), load(x + 1, y), load(x, y + 1), load(x + 1, y + 1))

    # 大小和 payment 的验证码一致
    result = Image.new('L', (200, 80))
    for i in range(result.width):
        for j in range(result.height):
            u, v = float(i) / result.width, float(j) / result.height
            result.putpixel((i, j), sample(u, v))
    return result

def enumerate_captcha_num_images(img: Image.Image):
    for i in range(4):
        img_width, img_height = img.size
        x_min = 0.25 * img_width * i
        x_max = 0.25 * img_width * (i + 1)
        yield i, img.crop((x_min, 0, x_max, img_height))

def get_epochs_and_batch_size(type: CaptchaType):
    if type == CaptchaType.PAYMENT:
        return 100, 32
    if type == CaptchaType.ZFW:
        return 200, 64
    raise ValueError('Unknown captcha type')

class CaptchaWidget(object):
    def __init__(self, type: CaptchaType, width=350, height=200):
        self.closed = False
        self.type = type
        self.root = tk.Tk()
        self.root.title("回车继续，关闭退出")

        # 显示在屏幕中央
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        position_left = (screen_width - width) // 2
        position_top = (screen_height - height) // 2
        self.root.geometry(f"{width}x{height}+{position_left}+{position_top}")

        def on_close():
            self.closed = True
            self.root.destroy()
        self.root.protocol("WM_DELETE_WINDOW", on_close)

    def run(self):
        if self.closed:
            return False

        with get_captcha_image(self.type) as original_img:
            img = preprocess_captcha_image(self.type, original_img)

            if img is None:
                self.root.destroy()
                return True

            captcha_img = ImageTk.PhotoImage(original_img)
            label_image = tk.Label(self.root, image=captcha_img)
            label_image.pack(pady=10)

            entry = tk.Entry(self.root)
            entry.pack(pady=10)
            entry.insert(0, self.get_initial_entry_text(img))

            def on_submit(event):
                self.on_submit(img, entry.get().strip())
                self.root.destroy()
            entry.bind("<Return>", on_submit)

            entry.focus_force()
            self.root.mainloop()
        return not self.closed

    def get_initial_entry_text(self, img: Image.Image) -> str:
        return ''

    def on_submit(self, img: Image.Image, text: str) -> None:
        pass
