import numpy as np
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
for physical_device in physical_devices:
    tf.config.experimental.set_memory_growth(physical_device, True)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D

from work_object_detection_model import const


class Model(Sequential):
    def __init__(self):
        super().__init__()
        self._build()

    def _build(self):
        self.add(Conv2D(16, 3, padding='same', activation='relu', input_shape=(const.IMAGE_HEIGHT, const.IMAGE_WIDTH, 3)))
        self.add(MaxPooling2D())
        self.add(Conv2D(32, 3, padding='same', activation='relu'))
        self.add(MaxPooling2D())
        self.add(Conv2D(64, 3, padding='same', activation='relu'))
        self.add(MaxPooling2D())
        self.add(Dropout(0.2))
        self.add(Flatten())
        self.add(Dense(512, activation='relu'))
        self.add(Dense(1, activation='sigmoid'))
        return

    def compile(self):
        super().compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            # optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
            loss='binary_crossentropy',
            metrics=['accuracy'])
        return

    def train(self, train_gen, val_gen, weights_dir, weights_prefix):
        callbacks = [
            # tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
            tf.keras.callbacks.TensorBoard(log_dir='./logs'),
            # tf.keras.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5')
            tf.keras.callbacks.ModelCheckpoint(filepath=weights_dir + '/' + weights_prefix + '-epoch={epoch:02d}-val_loss={val_loss:.2f}')
        ]
        history = self.fit_generator(
            train_gen,
            steps_per_epoch=train_gen.n // const.BATCH_SIZE,
            epochs=const.EPOCHS,
            validation_data=val_gen,
            validation_steps=val_gen.n // const.BATCH_SIZE,
            callbacks=callbacks
        )
        return history

    # def save_weights(self, path):
    #     self.model.save_weights(path)
    #     return

    # def load_weights(self, path):
    #     self.model.load_weights(path)
    #     return

    def evaluate(self, images, labels):
        assert len(images) == len(labels)
        loss, accuracy = super().evaluate(images, labels, batch_size=len(images))
        return loss, accuracy

    def predict(self, images):
        results = super().predict(images)
        return results

    def predict_plus(self, images):
        results = self.predict(images)
        accuracy = results[:, 0]
        labels = np.where(accuracy < 0.5, True, False)
        return accuracy, labels
