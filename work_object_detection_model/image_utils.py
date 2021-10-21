import os

import cv2
import numpy as np

from work_object_detection_model import const


def to_input_shape(image):
    image = cv2.resize(image, (const.IMAGE_HEIGHT, const.IMAGE_WIDTH))
    images = image[np.newaxis, :]
    return images


def put_text(image, x, y, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_color = (255, 255, 255)
    thickness = 2
    text_width, text_height = cv2.getTextSize(
        text, font, font_scale, thickness)[0]
    # cv2.putText(
    #     image, text,
    #     (x - int(text_width/2), y + int(text_height/2)),
    #     font, font_scale, font_color, thickness)
    cv2.putText(
        image, text,
        (x, y + int(text_height/2)),
        font, font_scale, font_color, thickness)
    return image


def save(image_path, image):
    cv2.imwrite(image_path, image)
    return
