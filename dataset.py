import numpy as np
import os
import time
import util

from PIL import Image
from sklearn.model_selection import train_test_split

def load(type: util.CaptchaType, test_size, random_state):
    images = []
    classes = []
    dataset_dir = util.get_captcha_dataset_dir(type)

    for filename in os.listdir(dataset_dir):
        if not filename.endswith('.jpg'):
            continue

        with Image.open(f'{dataset_dir}/{filename}') as img:
            images.append(util.captcha_image_to_array(type, img))
        classes.append(util.get_captcha_label_class(type, int(filename.split('.')[0])))

    import keras # 加载比较耗时，所以放在函数内

    x = np.array(images)
    y = keras.utils.to_categorical(np.array(classes), num_classes=util.get_captcha_num_classes(type)) # 转为 one-hot
    return train_test_split(x, y, test_size=test_size, random_state=random_state)

class MakeDataSetWidget(util.CaptchaWidget):
    def on_submit(self, img, text):
        dataset_dir = util.get_captcha_dataset_dir(self.type)
        for i, part in util.enumerate_captcha_num_images(img):
            part.save(f'{dataset_dir}/{text[i]}.{int(time.time())}.{i}.jpg')

if __name__ == '__main__':
    captcha_type = util.input_captcha_type()
    while MakeDataSetWidget(captcha_type).run():
        ...
