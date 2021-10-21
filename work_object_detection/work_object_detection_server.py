import base64
import multiprocessing
import sys
from concurrent import futures

import cv2
import grpc
import numpy as np

import work_object_detection_pb2
import work_object_detection_pb2_grpc
from work_object_detection_model import const
from work_object_detection_model.model import Model

# SERVER_PORT = 50051
SERVER_PORT = 50052
# WEIGHTS_PATH = f'./weights/2020-05-18T12:28:25.901703'
WEIGHTS_PATH = f'/home/latona/hades/BackendService/work-object-detection-server/weights/2020-05-24T19:30:23.098970-epoch=05-val_loss=0.02/variables/variables'

IMAGE_HEIGHT = 1200
IMAGE_WIDTH = 1920
N_SPLIT_HEIGHT = 2
N_SPLIT_WIDTH = 3
SPLIT_HEIGHT = 600
SPLIT_WIDTH = 640

model = Model()
model.compile()
model.load_weights(WEIGHTS_PATH)


def to_ndarray(image):
    image = base64.b64decode(image)
    image = np.frombuffer(image, dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image


def predict(image):
    image = cv2.resize(image, (const.IMAGE_WIDTH, const.IMAGE_HEIGHT))
    images = image[np.newaxis, :]
    images = images / 255
    results = model.predict(images)
    accuracy = float(results[0, 0])
    is_work = True if accuracy > 0.5 else False
    return accuracy, is_work


def split(image):
    images = []
    for n_w in range(N_SPLIT_WIDTH):
        for n_h in range(N_SPLIT_HEIGHT):
            h1 = SPLIT_HEIGHT * n_h
            h2 = SPLIT_HEIGHT * (n_h + 1)
            w1 = SPLIT_WIDTH * n_w
            w2 = SPLIT_WIDTH * (n_w + 1)
            _image = image[h1:h2, w1:w2]
            images.append(_image)
    return images


def predict_multiple(images):
    new_images = []
    for image in images:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image / 255
        image = cv2.resize(image, (const.IMAGE_WIDTH, const.IMAGE_HEIGHT))
        new_images.append(image)
    new_images = np.array(new_images)

    results = model.predict(new_images)
    all_accuracy = results[:, 0]
    all_is_work = np.where(all_accuracy > 0.5, True, False)

    accuracy = float(np.mean(all_accuracy))
    is_work = bool(np.any(all_is_work))

    return accuracy, is_work, all_accuracy.tolist(), all_is_work.tolist()


def put_text(image, accuracy, is_work):
    height, width, _ = image.shape

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    font_color = (255, 255, 255)
    thickness = 2

    text1 = f'{accuracy}'
    text_width1, text_height1 = cv2.getTextSize(
        text1, font, font_scale, thickness)[0]
    cv2.putText(
        image, text1,
        (width - text_width1, text_height1),
        font, font_scale, font_color, thickness)

    text2 = f'{is_work}'
    text_width2, text_height2 = cv2.getTextSize(
        text2, font, font_scale, thickness)[0]
    cv2.putText(
        image, text2,
        (width - text_width2, text_height1 + text_height2),
        font, font_scale, font_color, thickness)
    return


def debug_show(queue):
    cv2.namedWindow('work-object-detection-server')
    while True:
        debug_name, images, all_accuracy, all_is_work = queue.get()

        for image, accuracy, is_work in zip(images, all_accuracy, all_is_work):
            put_text(image, accuracy, is_work)

        if debug_name == 'single':
            image = images[0]
        elif debug_name == 'multiple':
            image1 = np.concatenate((images[0], images[1]), axis=0)
            image2 = np.concatenate((images[2], images[3]), axis=0)
            image3 = np.concatenate((images[4], images[5]), axis=0)
            image = np.concatenate((image1, image2, image3), axis=1)
        else:
            print(f'Ignore debug request which debug_name is {debug_name}.')

        cv2.imshow('work-object-detection-server', image)
        cv2.waitKey(1)
    return


class WorkObjectDetectionServicer(work_object_detection_pb2_grpc.WorkObjectDetectionServicer):
    def __init__(self):
        super().__init__()
        if sys.flags.debug:
            print('*' * 50)
            print('Debug start')
            print('*' * 50)
            ctx = multiprocessing.get_context('spawn')
            self.queue = ctx.Queue()
            self.process = ctx.Process(target=debug_show, args=(self.queue,))
            self.process.start()

    def __del__(self):
        if sys.flags.debug:
            self.process.terminate()

    def Predict(self, request, context):
        print(f'request.date = {request.date}')
        image = to_ndarray(request.image)
        accuracy, is_work = predict(image)
        if sys.flags.debug:
            self.queue.put(('single', [image], [accuracy], [is_work]))
        return work_object_detection_pb2.Detection(accuracy=accuracy, is_work=is_work)

    def SplitAndPredict(self, request, context):
        print(f'request.date = {request.date}.')
        image = to_ndarray(request.image)
        height, width, channel = image.shape
        if not (height == IMAGE_HEIGHT and width == IMAGE_WIDTH):
            error = f'Image shape must be ({IMAGE_HEIGHT, IMAGE_WIDTH, 3}, but requested image shape is ({height}, {width}, {channel}))'
            return work_object_detection_pb2.Detections(
                status=False, error=error,
                accuracy=None, is_work=None,
                all_accuracy=None, all_is_work=None)

        images = split(image)
        accuracy, is_work, all_accuracy, all_is_work = predict_multiple(images)

        if sys.flags.debug:
            self.queue.put(('multiple', images, all_accuracy, all_is_work))

        return work_object_detection_pb2.Detections(
            status=True, error=None,
            accuracy=accuracy, is_work=is_work,
            all_accuracy=all_accuracy, all_is_work=all_is_work)


def run_server():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    work_object_detection_pb2_grpc.add_WorkObjectDetectionServicer_to_server(
        WorkObjectDetectionServicer(), server
    )
    server.add_insecure_port(f'[::]:{SERVER_PORT}')
    server.start()
    print(f'Listening [::]:{SERVER_PORT}')
    server.wait_for_termination()
    return


if __name__ == "__main__":
    run_server()
