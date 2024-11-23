import numpy as np
import util

from tensorflow.lite.python.interpreter import Interpreter

class PredictWidget(util.CaptchaWidget):
    def __init__(self, interpreter, width=350, height=200):
        super().__init__(width, height)
        self.interpreter = interpreter

    def _infer_num(self, img):
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        # print(input_details[0]['shape'], output_details[0]['shape'])

        input_data = np.expand_dims(util.image_to_grayscale_float_array(img), axis=0) # 增加 batch 的维度
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        output_data = interpreter.get_tensor(output_details[0]['index'])
        predicted_class = np.argmax(output_data, axis=-1)
        return predicted_class[0] + 1

    def get_initial_entry_text(self, img):
        nums = [self._infer_num(part) for _, part in util.enumerate_captcha_num_images(img)]
        return ''.join(map(str, nums))

if __name__ == '__main__':
    interpreter = Interpreter(model_path="./model/captcha-solver.tflite")
    interpreter.allocate_tensors()
    while PredictWidget(interpreter).run():
        ...
