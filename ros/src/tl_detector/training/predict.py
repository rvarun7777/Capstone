import os
from keras.preprocessing.image import ImageDataGenerator
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from keras.utils import np_utils
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint
import cv2
import numpy as np
import glob
import os
import model as traffic_light_model

from PIL import Image
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array


def get_images_with_labels_udacity():
    # red_image_file = glob.glob("udacity_data/*/red/*.jpg")
    # yellow_image_file = glob.glob("udacity_data/*/yellow/*.jpg")
    # green_image_file = glob.glob("udacity_data/*/green/*.jpg")
    red_image_file = glob.glob("data/udacity_data/bag_dump_just_traffic_light/red/*.jpg")
    yellow_image_file = glob.glob("data/udacity_data/bag_dump_just_traffic_light/yellow/*.jpg")
    green_image_file = glob.glob("data/udacity_data/bag_dump_just_traffic_light/green/*.jpg")
    nolight_image_file = glob.glob("data/udacity_data/bag_dump_just_traffic_light/nolight/*.jpg")

    red_images,red_labels = process_image_with_label(red_image_file,color = "red")
    yellow_images,yellow_labels = process_image_with_label(yellow_image_file,color = "yellow")
    green_images,green_labels = process_image_with_label(green_image_file,color = "green")
    off_images,off_labels = process_image_with_label(nolight_image_file,color = "off")

    images = np.concatenate((red_images,yellow_images, green_images, off_images))
    labels = np.concatenate((red_labels, yellow_labels, green_labels, off_labels))

    return np.array(images),np.array(labels)

udacity_images, udacity_labels = get_images_with_labels_udacity()

label_count = 4
# One-hot encode the labels
udacity_labels_oh = np_utils.to_categorical(udacity_labels, label_count)
pred_udacity = model.predict_classes(udacity_images, verbose = 0)
score_udacity = model.evaluate(udacity_images,udacity_labels_oh)
