import dataset
import keras
import matplotlib.pyplot as plt
import tensorflow as tf
import util

def get_model(type: util.CaptchaType, input_shape):
    if type == util.CaptchaType.PAYMENT:
        return keras.models.Sequential([
            keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),

            keras.layers.Conv2D(32, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),

            keras.layers.Conv2D(32, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D(pool_size=(4, 4)),

            keras.layers.Flatten(),
            keras.layers.Dense(64, activation='relu'), # 适当减少神经元
            keras.layers.Dropout(0.3),                 # Dropout 防止过拟合
            keras.layers.Dense(util.get_captcha_num_classes(type), activation='softmax')
        ])

    if type == util.CaptchaType.ZFW:
        return keras.models.Sequential([
            keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),

            keras.layers.Conv2D(32, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),

            keras.layers.Conv2D(32, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D(pool_size=(4, 4)),

            keras.layers.Flatten(),
            keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.06)), # L2 正则化防止过拟合
            keras.layers.Dropout(0.55), # Dropout 防止过拟合
            keras.layers.Dense(util.get_captcha_num_classes(type), activation='softmax')
        ])

    raise ValueError('Unknown captcha type')

def save_model(type: util.CaptchaType, model):
    model.save(util.get_captcha_model_path(type, 'h5'))
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    with open(util.get_captcha_model_path(type, 'tflite'), 'wb') as f:
        f.write(converter.convert())

def plot(history):
    _, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].plot(history.history['accuracy'], label='train accuracy')
    axes[0].plot(history.history['val_accuracy'], label='val accuracy')
    axes[0].set_title('Accuracy over Epochs')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()

    axes[1].plot(history.history['loss'], label='train loss')
    axes[1].plot(history.history['val_loss'], label='val loss')
    axes[1].set_title('Loss over Epochs')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()

    plt.tight_layout()
    plt.show()

def train(type: util.CaptchaType):
    x_train, x_val, y_train, y_val = dataset.load(type, test_size=0.2, random_state=42)
    print(len(x_train), len(x_val))

    model = get_model(type, x_train[0].shape)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    epochs, batch_size = util.get_epochs_and_batch_size(type)
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_val, y_val))
    save_model(type, model)
    plot(history)

if __name__ == '__main__':
    train(util.input_captcha_type())
