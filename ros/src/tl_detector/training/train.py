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

batch_size = 64
nb_epoch = 50

def process_image_with_label(image_file_names, color = "red"):
    image_array = []
    labels = []
    desired_size = (128,128)
    color_label = {"red":0, "yellow":1, "green":2, "unknown":3}
    label = color_label.get(color)

    for image_file_name in image_file_names:
        image = cv2.imread(image_file_name)
        resized_image = cv2.resize(image, desired_size, cv2.INTER_LINEAR)
        resized_image = resized_image.astype('float32')/255
        image_array.append(resized_image)
        labels.append(label)
    return np.array(image_array),np.array(labels)

def get_images_with_labels(data_dir):
    images = []
    labels = []

    red_images, red_labels = process_image_with_label(glob.glob(os.path.join(data_dir,"*_red.png")), color = "red")
    yellow_images, yellow_labels = process_image_with_label(glob.glob(os.path.join(data_dir,"*_yellow.png")), color = "yellow")
    green_images, green_labels = process_image_with_label(glob.glob(os.path.join(data_dir,"*_green.png")), color = "green")
    off_images, off_labels = process_image_with_label(glob.glob(os.path.join(data_dir,"*_unknown.png")), color = "unknown")

    images = np.concatenate((red_images,yellow_images, green_images, off_images))
    labels = np.concatenate((red_labels, yellow_labels, green_labels, off_labels))
    return np.array(images),np.array(labels)

# Concatenate training data with additional training data
train_images = np.concatenate((train_images, additional_images))
train_labels = np.concatenate((train_labels, additional_labels))

# One-hot encode the labels
NUM_CLASSES = 4
train_labels_oh = np_utils.to_categorical(train_labels, NUM_CLASSES)
additional_labels_oh = np_utils.to_categorical(additional_labels,  NUM_CLASSES)
test_labels_oh = np_utils.to_categorical(test_labels,  NUM_CLASSES)

# Train, Validation split
X_train, X_val, y_train, y_val = train_test_split(
                train_images,train_labels_oh,train_size = 0.75, random_state = 402)
assert(X_train.shape[0] == y_train.shape[0])
assert(X_val.shape[0] == y_val.shape[0])

model = traffic_light_model.get_model()

checkpointer =  ModelCheckpoint(filepath = 'light_classifier_model.hdf5',
                              verbose = 1, save_best_only = True)
history = model.fit(X_train, y_train,
                    batch_size=batch_size, epochs=nb_epoch,
                    verbose=1, validation_data=(X_val, y_val),callbacks = [checkpointer])
#model.save('light_classifier_model.h5')
model_json = model.to_json()
with open("light_classifier.json", "w") as json_file:
    json_file.write(model_json)
    
# serialize weights to HDF5
model.save_weights("light_classifier_model.hdf5")
print("Saved model to disk")
