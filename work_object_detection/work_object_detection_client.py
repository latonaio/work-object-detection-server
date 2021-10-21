import base64
import datetime
import glob

import cv2
import grpc

import work_object_detection_pb2
import work_object_detection_pb2_grpc


SERVER_HOST = '127.0.X.X'
# SERVER_PORT = 50051
SERVER_PORT = 50052

IMAGES_PATTERN = './dataset/test/**/*.jpg'


def mock_streaming_images():
    image_paths = glob.glob(IMAGES_PATTERN)
    image_paths.sort()
    print(f"glob.glob({IMAGES_PATTERN}) => {len(image_paths)} image paths.")

    for image_path in image_paths:
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (1920, 1200))

        result, encoded_image = cv2.imencode('.jpg', image)
        if not result:
            raise RuntimeError("cv2.imencode('.jpg', image) => False")
        base64_image = base64.b64encode(encoded_image)
        yield base64_image, timestamp
    return


class WorkObjectDetectionClient():
    def __init__(self):
        self.channel = None
        self.stub = None

    def __enter__(self):
        self.channel = grpc.insecure_channel(f'{SERVER_HOST}:{SERVER_PORT}')
        self.stub = work_object_detection_pb2_grpc.WorkObjectDetectionStub(self.channel)
        return self

    def __exit__(self, exc_type, exe_value, traceback):
        if self.channel is not None:
            self.channel.close()
        self.channel = None
        self.stub = None

        if exc_type is not None:
            raise RuntimeError(exe_value)
        return

    def Predict(self, image, date, debug=False):
        if self.stub is None:
            raise RuntimeError('Not open channel.')

        request_data = work_object_detection_pb2.Image(image=image, date=date, debug=True)
        detection = self.stub.Predict(request_data)
        return detection

    def SplitAndPredict(self, image, date, debug=False):
        if self.stub is None:
            raise RuntimeError('Not open channel.')

        request_data = work_object_detection_pb2.Image(image=image, date=date, debug=True)
        detections = self.stub.SplitAndPredict(request_data)
        return detections


def test_Predict():
    with WorkObjectDetectionClient() as client:
        for image, date in mock_streaming_images():
            detection = client.Predict(image=image, date=date, debug=True)
            print(detection)
    return


def test_SplitAndPredict():
    with WorkObjectDetectionClient() as client:
        for image, date in mock_streaming_images():
            detections = client.SplitAndPredict(image=image, date=date, debug=True)
            print(detections.status, detections.error)
            print(detections.accuracy, detections.all_accuracy)
            print(detections.is_work, detections.all_is_work)
    return


if __name__ == "__main__":
    # test_Predict()
    test_SplitAndPredict()
