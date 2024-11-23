import numpy as np
import os
import time
import util

from PIL import Image
from sklearn.model_selection import train_test_split

def load(test_size, random_state):
    images = []
    labels = []

    for filename in os.listdir('./data'):
        if not filename.endswith('.jpg'):
            continue

        with Image.open(os.path.join('./data', filename)) as img:
            images.append(util.image_to_grayscale_float_array(img))
        labels.append(int(filename.split('.')[0]) - 1) # 数字 1-9 转为 0-8，为 one-hot 做准备

    import keras # 加载比较耗时，所以放在函数内

    x = np.array(images)
    y = keras.utils.to_categorical(np.array(labels), num_classes=9) # 转为 one-hot
    return train_test_split(x, y, test_size=test_size, random_state=random_state)

class MakeDataSetWidget(util.CaptchaWidget):
    def on_submit(self, img, text):
        for i, part in util.enumerate_captcha_num_images(img):
            part.save(f'./data/{text[i]}.{int(time.time())}.{i}.jpg')

if __name__ == '__main__':
    while MakeDataSetWidget().run():
        ...
