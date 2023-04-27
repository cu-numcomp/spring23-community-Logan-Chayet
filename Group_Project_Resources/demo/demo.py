import collections
import os
import sys

import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from PIL import Image

sys.path.append('losses')
import losses

# Load the image file into a variable
true_image = Image.open('/Users/nicolasperrault/Desktop/Results/SingleGearSmallMovement/true_frame_middle.png')
interoplated_image = Image.open('/Users/nicolasperrault/Desktop/Results/SingleGearSmallMovement/output_middle.png')


tensor_image = img_to_array(true_image)
tensor_image = tf.convert_to_tensor(tensor_image)
example = {'y': tensor_image}


interoplated_image = img_to_array(interoplated_image)
interpolated_tensor_image = tf.convert_to_tensor(interoplated_image)
prediction = {'image': interpolated_tensor_image}


vgg_weights = '/Users/nicolasperrault/Desktop/spring23-community-Logan-Chayet/LossCalculation/frame-interpolation/pretrained_models/vgg/imagenet-vgg-verydeep-19.mat'

loss = losses.vgg_loss(example, prediction, vgg_weights)
print(loss.numpy())