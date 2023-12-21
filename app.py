# some utilities
import os
import numpy as np
from util import base64_to_pil
from fearture_extract import predict_image

# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
import tensorflow.keras as keras
import keras.utils as image
from keras.applications import vgg19

#tensorflow
# import tensorflow as tf
# from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

# tf.keras.applications.MobileNetV2(
#     input_shape=None,
#     alpha=1.0,
#     include_top=True,
#     weights="imagenet",
#     input_tensor=None,
#     pooling=None,
#     classes=1000,
#     classifier_activation="softmax",
# )


# Variables 
# Change them if you are using custom model or pretrained model with saved weigths
Model_json = ".json"
Model_weigths = ".h5"


# Declare a flask app
app = Flask(__name__)

def get_ImageClassifierModel():
    # model = MobileNetV2(weights='imagenet')

    # Loading the pretrained model
    # model_json = open(Model_json, 'r')
    # loaded_model_json = model_json.read()
    # model_json.close()
    # model = model_from_json(loaded_model_json)
    # model.load_weights(Model_weigths)
    
    # model_path = 'D:\\doan\\fruit_recog_model.h5'

    # # # Load the MobileNetV2 model from the .h5 file
    model = keras.models.load_model('D:\\dataset\\train\\vgg19.h5')

    return model  
    


def model_predict(img, model):
    '''
    Prediction Function for model.
    Arguments: 
        img: is address to image
        model : image classification model
    '''
    CLASSES = ['Blazer', 'Blouse', 'Body', 'Dress', 'Hat', 'Hoodie', 'Longsleeve', 'Not sure', 'Other', 'Outwear', 'Pants', 'Polo', 'Shirt', 'Shoes', 'Shorts', 'Skip', 'Skirt', 'Top', 'T-Shirt', 'Undershirt']
    img = img.resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = vgg19.preprocess_input(x)
    
    return CLASSES[np.argmax(model.predict(x))]
    # # Preprocessing the image
    # x = image.img_to_array(img)
    # # x = np.true_divide(x, 255)
    # x = np.expand_dims(x, axis=0)

    # # Be careful how your trained model deals with the input
    # # otherwise, it won't make correct prediction!
    # x = preprocess_input(x, mode='tf')

    # preds = model.predict(x)
    # return preds


@app.route('/', methods=['GET'])
def index():
    '''
    Render the main page
    '''
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    '''
    predict function to predict the image
    Api hits this function when someone clicks submit.
    '''
    if request.method == 'POST':
        img = base64_to_pil(request.json)
        model = get_ImageClassifierModel()

        preds = model_predict(img, model)
        
        print(preds)
        
        name_products = predict_image(img, preds)
        print(name_products)
        
        # Serialize the result, you can add additional fields
        #return jsonify(result=result, probability=pred_proba)
        return jsonify(result=name_products, probability=preds)
    return None


if __name__ == '__main__':
    app.run('192.168.3.12', port=5000)


# from flask import Flask, request, jsonify, render_template
# from keras.models import load_model
# from keras.preprocessing import image
# from keras.applications.imagenet_utils import preprocess_input
# import numpy as np
# from PIL import Image

# from util import base64_to_pil

# app = Flask(__name__)

# # Load pre-trained CNN model
# model_path = 'D:\\doan\\fruit_recog_model.h5'
# model = load_model(model_path)

# def process_image_from_pil(pil_image):
#     # Resize hình ảnh nếu cần thiết
#     pil_image = pil_image.resize((64, 64))
    
#     # Chuyển hình ảnh sang mảng numpy
#     img_array = image.img_to_array(pil_image)
    
#     # Thêm chiều mới để phù hợp với đầu vào của mô hình
#     img_array = np.expand_dims(img_array, axis=0)
    
#     # Tiền xử lý hình ảnh
#     img_array = preprocess_input(img_array)
    
#     return img_array

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/predict', methods=['GET', 'POST'])
# def predict():
#     if request.method == 'POST':
#         img = base64_to_pil(request.json)

#         # Tiền xử lý hình ảnh từ dữ liệu PIL
#         img_array = process_image_from_pil(img)

#         # Dự đoán
#         predictions = model.predict(img_array)

#         # Lấy nhãn của lớp có xác suất cao nhất
#         predicted_class = np.argmax(predictions)

#         # Đọc file nhãn (labels.txt) để có tên của lớp
#         with open('labels.txt', 'r') as file:
#             labels = file.readlines()
        
#         result = labels[predicted_class].strip()

#         return jsonify(result=result)


# if __name__ == '__main__':
#     app.run('192.168.3.12', port=5000)
