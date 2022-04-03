import requests as requests
import base64
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import json
from PIL import Image
from io import BytesIO
import sched, time

IMG_SIZE = 200
IMAGE_PATH = "img.jpg"
URL = 'http://api-voltron.herokuapp.com/api/'
img_id = 2


def send_analyse_result(score):
    response_post = requests.post(URL + 'images_process', data={'status': score, 'image_id': _id})
    if (response_post.status_code == 200):
        print("The request was a success!")

    elif (response_post.status_code == 404):
        print("Result not found!")


def build_model():
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    # x = layers.experimental.preprocessing.Rescaling(1. / 255)(inputs)
    x = layers.Conv2D(16, 3, activation="relu", padding="same")(inputs)
    x = layers.Conv2D(16, 3, activation="relu", padding="same")(x)
    x = layers.MaxPool2D()(x)

    x = layers.SeparableConv2D(32, 3, activation="relu", padding="same")(x)
    x = layers.SeparableConv2D(32, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D()(x)
    x = layers.SeparableConv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.SeparableConv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D()(x)

    x = layers.SeparableConv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.SeparableConv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.SeparableConv2D(256, 3, activation="relu", padding="same")(x)
    x = layers.SeparableConv2D(256, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Flatten()(x)

    x = layers.Dense(512, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.7)(x)

    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(64, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def analyse_picture():
    model = build_model()
    model.load_weights("best_model.h5")
    img = keras.preprocessing.image.load_img(
        IMAGE_PATH, target_size=(IMG_SIZE, IMG_SIZE)
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis

    predictions = model.predict(img_array)
    score = predictions[0]
    print(
        "This image is %.2f percent Normal and %.2f percent Sick."
        % (100 * (1 - score), 100 * score)
    )
    send_analyse_result(score)


start_time = time.time()
_id = '0000017'
while True:
    print ('here we go again')
    response = requests.get(URL + '/images/lasted')
    if response.status_code == 200:
        print("The request was a success!")
        body = json.loads((response.content))
        print(body['base_64'])
        base64_img = body['base_64']
        if _id != body['_id']:
            _id = body['_id']
            im = Image.open(BytesIO(base64.b64decode(base64_img)))
            im.save(IMAGE_PATH, 'PNG')
            analyse_picture()
    elif response.status_code == 404:
        print("Result not found!")
    time.sleep(300.0 - ((time.time() - start_time) % 60.0))
