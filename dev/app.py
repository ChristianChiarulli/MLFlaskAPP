from flask import Flask, render_template, request, jsonify
import base64
import numpy as np
import io
from PIL import Image
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import tensorflow as tf
import matplotlib.pyplot as plt


app = Flask(__name__)

#  APP_ROOT = os.path.dirname(os.path.abspath(__file__))
app.config['TEMPLATES_AUTO_RELOAD'] = True


@app.route('/')
def index():
    return render_template('home.html')


@app.route('/test')
def test():
    return render_template('test.html')


@app.route('/layout')
def layout():
    return render_template('layout.html')


def get_model():
    global model, graph
    model = load_model('VGG16_cats_and_dogs.h5')
    print(" * Model loaded!")
    graph = tf.get_default_graph()


def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    return image


print(" * Loading Keras model...")
get_model()


@app.route("/predict", methods=["POST"])
def predict():
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    with graph.as_default():
        image = Image.open(io.BytesIO(decoded))
        processed_image = preprocess_image(image, target_size=(224, 224))
        prediction = model.predict(processed_image).tolist()
        processed_image = np.squeeze(processed_image, axis=0)
        plt.imsave('static/images/processed.jpg', processed_image)
        data = open("static/images/processed.jpg", "rb").read()

    response = {
        'prediction': {
            'processed_image_url': f'data:image/jpeg;base64,{base64.b64encode(data).decode("utf-8")}',
            'dog': prediction[0][0],
            'cat': prediction[0][1]
        }
    }
    response = jsonify(response)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

# @app.route("/upload", methods=['POST'])
# def upload():
    # target = os.path.join(APP_ROOT, 'static/images')
    # print(target)

    # if not os.path.isdir(target):
        # os.mkdir(target)

    # for file in request.files.getlist("file"):
        # print(file)
        # filename = file.filename
        # destination = "/".join([target, filename])
        # print(destination)
        # file.save(destination)
        # retval = []
        # retval = {'filename': filename}
    # return jsonify(retval)


if __name__ == '__main__':
    app.run(debug=True)
