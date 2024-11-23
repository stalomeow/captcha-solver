import io
import requests
import tkinter as tk

from PIL import Image, ImageTk

class CaptchaWidget(object):
    def __init__(self, width=350, height=200):
        self.closed = False
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

        rsp = requests.get('https://payment.xidian.edu.cn/NetWorkUI/authImage')
        with Image.open(io.BytesIO(rsp.content)) as img:
            captcha_img = ImageTk.PhotoImage(img)
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

def enumerate_captcha_num_images(img: Image.Image):
    for i in range(4):
        img_width, img_height = img.size
        x_min = 0.25 * img_width * i
        x_max = 0.25 * img_width * (i + 1)
        yield i, img.crop((x_min, 0, x_max, img_height))

def image_to_grayscale_float_array(img: Image.Image):
    import keras # 加载比较耗时，所以放在函数内
    return keras.utils.img_to_array(img.convert('L')) / 255.0
