import onnxruntime as ort
import urllib.request
import numpy as np
import cv2
from PIL import Image

# Load the model
model_path = 'googlenet-12-int8.onnx'
session = ort.InferenceSession(model_path)

#Get the Class labels
url = 'https://raw.githubusercontent.com/HoldenCaulfieldRye/caffe/master/data/ilsvrc12/synset_words.txt'
urllib.request.urlretrieve(url, 'synset_words.txt')

# Get the input name and shape of the model
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape

# Pre-processing function for ImageNet models using numpy
def preprocess(img):
    # Preprocessing required on the images for inference with mxnet gluon
    # The function takes loaded image and returns processed tensor
    img = cv2.resize(img, (input_shape[3], input_shape[2])).astype(np.float32)
    img[:, :, 0] -= 123.68
    img[:, :, 1] -= 116.779
    img[:, :, 2] -= 103.939
    img[:,:,[0,1,2]] = img[:,:,[2,1,0]]
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img

def get_image(path):
    # Check if the path is a URL or a file path
    if path.startswith('http'):
        # Download the image from the URL
        with urllib.request.urlopen(path) as url:
            img = np.asarray(bytearray(url.read()), dtype="uint8")
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    else:
        # Read the image from the file
        img = cv2.imread(path, cv2.IMREAD_COLOR)
    return img

def predict(path):
    img = get_image(path)
    input_data = preprocess(img)
    output = session.run(None, {input_name: input_data})
    prob = output[0].squeeze()
    class_labels = np.loadtxt('synset_words.txt', str, delimiter='\t')
    top_k = np.argsort(prob)[::-1][:5]
    for i in top_k:
        print(f'{class_labels[i]}: {prob[i]:.2f}')
    return class_labels[top_k[0]], list(zip(class_labels[top_k], prob[top_k]))

# Call the predict function with the path to an image or a URL
path = 'sample_images/biker.jpg'
predicted_class, top_classes = predict(path)
print(f'The predicted class is: {predicted_class}')
print(f'Top 5 classes: {top_classes}')