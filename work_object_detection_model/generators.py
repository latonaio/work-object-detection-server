from tensorflow.keras.preprocessing.image import ImageDataGenerator

from work_object_detection_model import const


class ImagesGenerators():
    def get_train_generators(self):
        # Arguments of ImageDataGenerator
        #   rescale=1./255
        #   horizontal_flip=True: 反転する
        #   rotation_range=45: 回転する
        #   zoom_range=0.5: ズームする
        #   height_shift_range=.15: x軸方向にシフトする
        #   width_shift_range=.15: y軸方向にシフトする
        train_image_generator = ImageDataGenerator(rescale=1./255)
        train_data_gen = train_image_generator.flow_from_directory(
            batch_size=const.BATCH_SIZE,
            directory=const.TRAIN_DIR,
            shuffle=True,
            target_size=(const.IMAGE_HEIGHT, const.IMAGE_WIDTH),
            color_mode='rgb',
            class_mode='binary'
        )
        return train_data_gen

    def get_val_generators(self):
        validation_image_generator = ImageDataGenerator(rescale=1./255)
        val_data_gen = validation_image_generator.flow_from_directory(
            batch_size=const.BATCH_SIZE,
            directory=const.VAL_DIR,
            target_size=(const.IMAGE_HEIGHT, const.IMAGE_WIDTH),
            color_mode='rgb',
            class_mode='binary'
        )
        return val_data_gen

    def get_test_generators(self):
        test_image_generator = ImageDataGenerator(rescale=1./255)
        test_data_gen = test_image_generator.flow_from_directory(
            batch_size=const.BATCH_SIZE,
            directory=const.VAL_DIR,
            target_size=(const.IMAGE_HEIGHT, const.IMAGE_WIDTH),
            color_mode='rgb',
            class_mode='binary'
        )
        return test_data_gen


if __name__ == "__main__":
    generators = ImagesGenerators()
    train_gen = generators.get_train_generators()
    images, labels = next(train_gen)
    print(images.shape)
    print(labels.shape)

#    import os
#    import cv2
#    tmp_dir = './tmp'
#    os.makedirs(tmp_dir, exist_ok=True)
#    count = 0
#    for image in images[:20]:
#        count += 1
#        image_name = os.path.join(tmp_dir, f'sample_{count:02d}.jpg')
#        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#        cv2.imwrite(image_name, image)
#        print(f'Save {image_name}')
