import onnxruntime as ort
import numpy as np
import imageio
from PIL import Image

# get the graph input name and shape
import onnx
model = onnx.load('googlenet-12-int8.onnx')
graph = model.graph
print(graph.input)

# Load the model
model_path = 'googlenet-12-int8.onnx'
session = ort.InferenceSession(model_path)

# Get the input name and shape of the model
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape

# Pre-processing function for ImageNet models using numpy
def preprocess(img):
    # Preprocessing required on the images for inference with mxnet gluon
    # The function takes loaded image and returns processed tensor
    img = np.array(Image.fromarray(img).resize((input_shape[3], input_shape[2]))).astype(np.float32)
    img[:, :, 0] -= 123.68
    img[:, :, 1] -= 116.779
    img[:, :, 2] -= 103.939
    img[:,:,[0,1,2]] = img[:,:,[2,1,0]]
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img

def get_image(path): #Using path to image, return the RGB load image
    img = imageio.imread(path, pilmode='RGB')
    return img

def predict(path):
    img = get_image(path)
    input_data = preprocess(img)
    output = session.run(None, {input_name: input_data})
    prob = output[0].squeeze()
    a = np.argsort(prob)[::-1]
    return a

# Load and preprocess the image
path = "./uploads/car.jpg"
img = get_image(path)
input_data = preprocess(img)

# Run inference
output = session.run(None, {input_name: input_data})

# The output of the model is a list of arrays, one for each output node of the model
# You can access the output of the first node like this:
output_data = output[0]

# Do something with the output
#print(output_data)

path = "./uploads/car.jpg"
predicted_classes = predict(path)
print(predicted_classes)