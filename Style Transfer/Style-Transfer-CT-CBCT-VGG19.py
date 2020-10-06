# CBCT to CT
# Using Style Transfer to Reduce Noise/Image Artifacts
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

import IPython.display as display

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import PIL.Image
import time
import SimpleITK as sitk


def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    tensor = np.squeeze(tensor, axis = 0)
    #tensor = np.squeeze(tensor, axis = 2) no squeezing because its RGB now
    return PIL.Image.fromarray(tensor)

#Original Function
'''def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)
'''

'''Visualize the input
Define a function loading a 3D grayscale image, converting it to RGB Image 
slices of 512x512 and cropp horizontally and vertically by given constant'''


#Wenn .png, .jpg schon vorhanden, 
def load_img_png(path_to_img):
  max_dim = 512
  img = tf.io.read_file(path_to_img)
  img = tf.image.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)

  shape = tf.cast(tf.shape(img)[:-1], tf.float32)
  long_dim = max(shape)
  scale = max_dim / long_dim

  new_shape = tf.cast(shape * scale, tf.int32)
  
  img = tf.image.resize(img, new_shape) 
  return img
#Converting grayscale to fake 3Channel-RGB doesnt work, image gets colored

# Create a simple function to display an image:
def imshow(image, title=None):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)
    #entfernt die 1. Achse
        plt.imshow(image)
    if title:
        plt.title(title)
        
# Since this is a float image, define a function to keep the pixel values between 0 and 1:
def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


# Using local images
# content image as the originally measured CBCT image, style image is the desired CT image

content_image = load_img_png('/home/creim/Desktop/TrainModel/Train Data 4D/Registrated 4D CBCT Bin 0/Reg-CBCT-Img-Bin0-50.png')
style_image = load_img_png('/home/creim/Desktop/TrainModel/Train Data 4D/4D CT Data/CT-image-50.png')


# Load a VGG19 without the classification head, and list the layer names
vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
print(vgg.summary())
#!!! JUST WORK FOR RGB Color coded IMAGES, Input MUST have 3 Channels

# Showing the layers of the VGG Net
print('Layers of VGG Net:')
for layer in vgg.layers:
    print(layer.name)


# Choose intermediate layers from the network to represent the style and content of the image:
# Also interesting to see how the result changes if the layers of interest for content and style are changed
# List of Content layers where we will pull our feature maps
content_layers = ['block5_conv2']

# List of Style layer of interest
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)


'''Build the model
The networks in tf.keras.applications are designed so you can easily extract
the intermediate layer values using the Keras functional API.

To define a model using the functional API, specify the inputs and outputs:

model = Model(inputs, outputs)

This following function builds a VGG19 model that returns a list of
intermediate! layer outputs:'''


def vgg_layers(layer_names):
    """ Creates a vgg model that returns a list of intermediate output values.
    Load our model. Load pretrained VGG, trained on imagenet data 
    (which are RGB pictures of random objects, better training at CT data!!)"""
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in layer_names]

    model = tf.keras.Model([vgg.input], outputs)
    return model


style_extractor = vgg_layers(style_layers)
style_outputs = style_extractor(style_image)


# Calculate style
def gram_matrix(input_tensor):
    '''Scalar product of two "style-tensors", where a large value represents a 
    good agreement of both styles, and a small value -> 0 a bad agreement'''
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result/(num_locations)


'''Extract style and content
Build a model that returns the style and content tensors.'''
'''When called on an image, this model returns the gram matrix (style) of
the style_layers and content of the content_layers:'''
class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        "Expects float input in [0,1]"
        inputs = inputs*255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                      outputs[self.num_style_layers:])

        style_outputs = [gram_matrix(style_output)
                     for style_output in style_outputs]

        content_dict = {content_name:value
                     for content_name, value
                     in zip(self.content_layers, content_outputs)}

        style_dict = {style_name: value
                     for style_name, value
                     in zip(self.style_layers, style_outputs)}
        return {'content': content_dict, 'style': style_dict}


extractor = StyleContentModel(style_layers, content_layers)

results = extractor(tf.constant(content_image))

style_results = results['style']

# Show statistics of used Layers
print('Statistics of used layers')
print('Styles:')
for name, output in sorted(results['style'].items()):
    print("  ", name)
    print("    shape: ", output.numpy().shape)
    print("    min: ", output.numpy().min())
    print("    max: ", output.numpy().max())
    print("    mean: ", output.numpy().mean())
    print()

print("Contents:")
for name, output in sorted(results['content'].items()):
    print("  ", name)
    print("    shape: ", output.numpy().shape)
    print("    min: ", output.numpy().min())
    print("    max: ", output.numpy().max())
    print("    mean: ", output.numpy().mean())

# Set your style and content target values:
style_targets = extractor(style_image)['style']
content_targets = extractor(content_image)['content']


# Parameters to adjust 
# Create an optimizer. The paper recommends LBFGS, but Adam works okay, too:
opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

style_weight = 1e-2   # default value: 1e-2
content_weight = 1e4 # default value: 1e4 
total_variation_weight= 30 # default value: 30


# Loss Function
def style_content_loss(outputs):
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2)
                           for name in style_outputs.keys()])
    style_loss *= style_weight / num_style_layers

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2)
                             for name in content_outputs.keys()])
    content_loss *= content_weight / num_content_layers
    loss = style_loss + content_loss
    return loss



def high_pass_x_y(image):
    x_var = image[:, :, 1:, :] - image[:, :, :-1, :]
    y_var = image[:, 1:, :, :] - image[:, :-1, :, :]

    return x_var, y_var

'''This will be shown in the subplots below
for the High Frequency artifacts of the style image, thus we initialize it here
'''

x_deltas, y_deltas = high_pass_x_y(content_image)

'''
print('High Frequency Artifacts')
plt.figure(figsize=(14, 10))
plt.subplot(2, 2, 1)
imshow(2*y_deltas+0.5, "Horizontal Deltas: Original")

plt.subplot(2, 2, 2)
imshow(2*x_deltas+0.5, "Vertical Deltas: Original")

x_deltas, y_deltas = high_pass_x_y(image)

plt.subplot(2, 2, 3)
imshow(2*y_deltas+0.5, "Horizontal Deltas: Styled")

plt.subplot(2, 2, 4)
imshow(2*x_deltas+0.5, "Vertical Deltas: Styled")

print('Therefore we need an additional loss factor: Total Variation Loss, which is included using the tensorflow TV function')
'''

# Shows that the high frequency artefacts of the image increase in the 'styled' version of the image
# Therefore we need the TV loss
'''Total variation loss
One downside to this basic implementation is that it produces a lot of
high frequency artifacts. Decrease these using an explicit regularization
term on the high frequency components of the image. In style transfer, this is
often called the total variation loss:

But tensorflow offers a TV loss implementation:
tf.image.total_variation(image).numpy()
Which is included in the code below
'''

#Initialize the optimization variable:
image = tf.Variable(content_image)

@tf.function()
def train_step(image):
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = style_content_loss(outputs)
        loss += total_variation_weight*tf.image.total_variation(image)

    grad = tape.gradient(loss, image)
    opt.apply_gradients([(grad, image)])
    image.assign(clip_0_1(image))


'''
Transfer Quality ? 
Comparing the styled image to the style target by comparing using different
error methods: pixelwise - MSE, NCC 
               anatomically - SSIM, DICE'''

#Evaluate the Mean Squared Error comparing Stylized image and Style Target Image
def MSE(img, style_img):
    return np.square(np.subtract(img,style_img)).mean(axis=None) 


#Evaluate the Normalized Cross Correlation:
#As defined here: c'_{av}[k] = sum_n a[n] conj(v[n+k])
def NCC(img, style_image):
    #Normalize inputs first
    img = (img-np.amin(img))/(np.amax(img)-np.amin(img))
    style_image = (style_image-np.amin(style_image))/(np.amax(style_image)-np.amin(style_image))
    ncc = np.correlate(img, style_image, 'full')
    return ncc



# Run optimization!!:
start = time.time()

epochs = 15
steps_per_epoch = 50

step = 0
for n in range(epochs):
    for m in range(steps_per_epoch):
        step += 1
        train_step(image)
        print(".", end='')
    display.clear_output(wait=True)
    display.display(tensor_to_image(image))
    print("Train step: {}".format(step))
end = time.time()


print("Total time: {:.1f}".format(end-start))

print('Parameters used:')
print('Initial Style Weight: ', style_weight)
print('Initial Content Weight: ', content_weight)
print('Total Variation Weight: ', total_variation_weight)

'''#Should print out the Mean Square Error of the stylized image to the style target image...
print('MSE: ', MSE(image,style_image)) 
possible reason why this doesnt work is that image is defined as tf.variable 
which maybe cant be used for array like computation
print('NCC:', NCC(image, style_image))
'''

plt.subplot(1, 2, 1)
imshow(content_image, 'Content Image')

plt.subplot(1, 2, 3)
imshow(style_image, 'Style Image')

plt.subplot(1, 3, 2)
imshow(image, 'Styled Image')

'Finally saving the result'
file_name = 'CT-styled-CBCT-image-orginal-parametersVGG19.png'
tensor_to_image(image).save(file_name)



try:
    from google.colab import files
except ImportError:
     pass
else:
    files.download(file_name)