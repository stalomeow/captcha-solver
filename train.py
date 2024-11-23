import dataset
import keras
import matplotlib.pyplot as plt
import tensorflow as tf

def get_model(input_shape):
    return keras.models.Sequential([
        keras.layers.Conv2D(8, (3, 3), activation='relu', input_shape=input_shape),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),

        keras.layers.Conv2D(16, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(4, 4)),

        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),    # 适当减少神经元
        keras.layers.Dropout(0.3),                    # Dropout 防止过拟合
        keras.layers.Dense(9, activation='softmax')   # 9 个类别，数字 1-9
    ])

def save_model(model):
    model.save('./model/captcha-solver.h5')
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    with open('./model/captcha-solver.tflite', 'wb') as f:
        f.write(converter.convert())

def plot(history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

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

def train():
    x_train, x_val, y_train, y_val = dataset.load(test_size=0.2, random_state=42)
    print(len(x_train), len(x_val))

    model = get_model(x_train[0].shape)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    history = model.fit(x_train, y_train, epochs=100, batch_size=32, validation_data=(x_val, y_val))
    save_model(model)
    plot(history)

if __name__ == '__main__':
    train()
