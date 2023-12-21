from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import tensorflow.keras as keras
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained VGG19 model
base_model = VGG19(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)

# model = keras.models.load_model('D:\\dataset\\train\\vgg19.h5')
# Extract features for two images
img_path1 = "D:\\doan\\picture\\Hoodie\\Hoodie 1.jpg"
img_path2 = "D:\\doan\\figma\\ao2.jpg"

# img1 = image.load_img(img_path1, target_size=(224, 224))
# img_array1 = image.img_to_array(img1)
# img_array1 = np.expand_dims(img_array1, axis=0)
# img_array1 = preprocess_input(img_array1)
# features1 = model.predict(img_array1)

def predict_image(img, label):
    folder_path = "D:\\doan\\picture\\"+label
    similarities = {}
    
    img1 = img.resize((224, 224))
    img_array1 = image.img_to_array(img1)
    img_array1 = np.expand_dims(img_array1, axis=0)
    img_array1 = preprocess_input(img_array1)
    features1 = model.predict(img_array1)
    
    
    # Duyệt qua tất cả các file ảnh trong thư mục
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            # Đường dẫn đầy đủ đến ảnh
            img_path = os.path.join(folder_path, filename)
            
            # Load và tiền xử lý ảnh
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            
            features2 = model.predict(img_array)
            
            # Giải mã và in ra kết quả
            features1 = features1.reshape((1, -1))
            features2 = features2.reshape((1, -1))
            # tính độ tương đồng
            similarity = cosine_similarity(features1, features2)
            
            similarities[filename] = similarity[0, 0]
            
            # print("Cosine Similarity with images ", filename, ": ", similarity[0, 0])
    
    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

    resultImage = []
    for i, (filename, similarity) in enumerate(sorted_similarities[:5]):
        resultImage.append(filename.split(".")[0])
    return resultImage
    