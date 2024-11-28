import numpy as np
import util

from tensorflow.lite.python.interpreter import Interpreter

class PredictWidget(util.CaptchaWidget):
    def __init__(self, interpreter, type: util.CaptchaType, width=350, height=200):
        super().__init__(type, width, height)
        self.interpreter = interpreter

    def _infer_num(self, img):
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        # print(input_details[0]['shape'], output_details[0]['shape'])

        input_data = np.expand_dims(util.captcha_image_to_array(self.type, img), axis=0) # 增加 batch 的维度
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        output_data = interpreter.get_tensor(output_details[0]['index'])
        predicted_class = np.argmax(output_data, axis=-1)
        return util.get_captcha_class_label(self.type, predicted_class[0])

    def get_initial_entry_text(self, img):
        nums = [self._infer_num(part) for _, part in util.enumerate_captcha_num_images(img)]
        return ''.join(map(str, nums))

if __name__ == '__main__':
    captcha_type = util.input_captcha_type()
    interpreter = Interpreter(model_path=util.get_captcha_model_path(captcha_type, 'tflite'))
    interpreter.allocate_tensors()

    while PredictWidget(interpreter, captcha_type).run():
        ...
