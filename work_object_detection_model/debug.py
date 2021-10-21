import glob
import os
import shutil
import numpy as np
import cv2

from work_object_detection_model import image_utils
from work_object_detection_model.model import Model


# weights_path = f'./weights/2020-05-18T18:12:22.168327-epoch=02-val_loss=0.01/variables/variables'
# weights_path = f'./weights/2020-05-24T18:37:32.481663-epoch=10-val_loss=0.03/variables/variables'
weights_path = f'./weights/2020-05-24T19:30:23.098970-epoch=10-val_loss=0.00/variables/variables'
model = Model()
model.compile()
model.load_weights(weights_path)

TMP_DIR = './tmp'
shutil.rmtree(TMP_DIR, ignore_errors=True)
os.makedirs(TMP_DIR, exist_ok=True)


def main():
    images_dir = './dataset/test/not_work'
    # images_dir = './dataset/test/work'
    image_paths = glob.glob(f'{images_dir}/*.jpg')
    image_paths.sort()

    for image_path in image_paths:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images = image_utils.to_input_shape(image)
        # images = images.astype('float32')
        images = images / 255
        accuracy, labels = model.predict_plus(images)
        accuracy = float(accuracy[0])
        label = bool(labels[0])

        x = int(image.shape[0] * 0.1)
        y = int(image.shape[1] * 0.1)
        text = f"accuracy={accuracy}, label={label}"
        image = image_utils.put_text(image, x, y, text)
        image_name = os.path.basename(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_utils.save(f'./tmp/{image_name}.jpg', image)
        print(f'{image_name}: {text}')

    return


if __name__ == "__main__":
    main()
