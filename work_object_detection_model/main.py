import datetime
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from work_object_detection_model.generators import ImagesGenerators
from work_object_detection_model.model import Model


def model():
    model = Model()
    model.summary()
    return


def train():
    weights_dir = f'./weights'
    os.makedirs(weights_dir, exist_ok=True)

    load_weights_name = 'xxx'
    load_weights_path = f'{weights_dir}/{load_weights_name}'

    save_weights_name = f'{datetime.datetime.now().isoformat()}'
    save_weights_path = f'{weights_dir}/{save_weights_name}'

    generators = ImagesGenerators()
    train_gen = generators.get_train_generators()
    val_gen = generators.get_val_generators()

    model = Model()
    model.compile()
    if os.path.exists(load_weights_path):
        model.load_weights(load_weights_path)
    model.summary()
    history = model.train(train_gen, val_gen, weights_dir, save_weights_name)
    print(history)
    model.save_weights(save_weights_path)
    return


def eval():
    # weights_path = f'./weights/2020-05-18T12:28:25.901703'
    weights_path = f'./weights/2020-05-18T18:12:22.168327-epoch=02-val_loss=0.01/variables/variables'
    model = Model()
    model.compile()
    model.load_weights(weights_path)

    generators = ImagesGenerators()
    test_gen = generators.get_test_generators()
    images, labels = next(test_gen)
    print('images.shape => ', images.shape)
    print('labels.shape => ', labels.shape)

    loss, accuracy = model.evaluate(images, labels)
    print(f'loss => {loss}')
    print(f'accuracy => {accuracy}')
    return


def predict():
    # weights_path = f'./weights/2020-05-18T12:28:25.901703'
    # weights_path = f'./weights/2020-05-24T15:39:11.604190'
    weights_path = f'./weights/2020-05-24T19:30:23.098970-epoch=05-val_loss=0.02/variables/variables'
    # weights_path = f'./weights/2020-05-18T18:12:22.168327-epoch=02-val_loss=0.01/variables/variables'
    model = Model()
    model.compile()
    model.load_weights(weights_path)

    generators = ImagesGenerators()
    test_gen = generators.get_test_generators()
    images, labels = next(test_gen)

    # image_path = '/Users/koiketaisuke/sewing_machine/Runtime/work-object-detection-server/model/images/test/not_work/test_00001.jpg'
    # import cv2
    # image = cv2.imread(image_path)
    # image = cv2.resize(image, (const.IMAGE_HEIGHT, const.IMAGE_WIDTH))
    # images = image.reshape((-1, *image.shape))
    # labels = np.array([1])

    results = model.predict(images)
    print('results.shape => ', results.shape)
    print(results[:, 0])
    print(labels)
    print(np.where(np.where(results[:, 0] < 0.5, 0., 1.) == labels, True, False))
    return


if __name__ == "__main__":
    # model()
    train()
    # eval()
    # predict()
