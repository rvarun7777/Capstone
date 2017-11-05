from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, MaxPooling2D, Conv2D,GlobalAveragePooling2D,Activation

def get_model():

    model = Sequential()
    # Layer 1
    model.add(Conv2D(16, (2, 2), padding = 'same', input_shape =train_images[0].shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(0.2))

    # Layer 2
    model.add(Conv2D(32, (2, 2), padding = 'same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(0.2))

    # Layer 3
    model.add(Conv2D(64, (2, 2), padding = 'same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(0.2))
    model.add(GlobalAveragePooling2D())

    # Fully connected layers
    model.add(Dense(30))
    model.add(Dropout(0.2))
    model.add(Dense(10))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

    return model
