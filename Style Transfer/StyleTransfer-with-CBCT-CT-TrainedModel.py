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
from skimage.measure import compare_ssim


mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False


def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    tensor = np.squeeze(tensor, axis = 0)
    tensor = np.squeeze(tensor, axis = 2)
    return PIL.Image.fromarray(tensor)

#Original Code
'''def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)
'''

# Wenn .png, .jpg schon vorhanden - Lädt Bild als tf Array!!
def load_img_png(path_to_img):
  max_dim = 512
  img = tf.io.read_file(path_to_img)
  img = tf.image.decode_image(img, channels=1)
  img = tf.image.convert_image_dtype(img, tf.float32)

  shape = tf.cast(tf.shape(img)[:-1], tf.float32)
  long_dim = max(shape)
  scale = max_dim / long_dim

  new_shape = tf.cast(shape * scale, tf.int32)

  img = tf.image.resize(img, new_shape)
  img = img[tf.newaxis, :]
  return img


# Create a simple function to display an image:
def imshow(image, title=None):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)
    #entfernt die 1. Achse
    #for grayscale image auch 3. Achse entfernen
    image = tf.squeeze(image, axis=2)
    plt.imshow(image, cmap='gray')
    if title:
        plt.title(title)

    
# Since this is a float image, define a function to keep the pixel values between 0 and 1:
def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


# Using local images
# content image as the oringinally measured CBCT image, style image is the desired CT image
content_image = load_img_png('/home/creim/Desktop/CycleGAN-Tensorflow-2/output4-new parameters/cbct2ct/example-cbct2ct-cycleGAN-translation/iter-000004200-cbct2ct.png')
style_image = load_img_png('/home/creim/Desktop/CycleGAN-Tensorflow-2/output4-new parameters/cbct2ct/example-cbct2ct-cycleGAN-translation/iter-000004200-target-ct.png')

content_image_before_GAN =load_img_png('/home/creim/Desktop/CycleGAN-Tensorflow-2/output4-new parameters/cbct2ct/example-cbct2ct-cycleGAN-translation/iter-000004200-orginal-cbct.png')

'''plt.imsave('Content-img.png', content_image)
plt.imsave('Style_img.png', style_image)
'''


# Load a VGG19 without the classification head, and list the layer names
#INSTEAD: LOAD PRETRAINED MODEL ON C/CBCT CLASSIFICATION HERE !!!
pretrained_model = tf.keras.models.load_model('/home/creim/Desktop/Style Transfer/CT-CBCT-ClassifierCNN-on-registrated-Data-deep.h5')

#Excluding the top classification layer:
pretrained_model.pop()

#Pretrained Model consists of following layers:
print('Layers of pretrained CNN:')

print(pretrained_model.summary())

# Choose intermediate layers from the network to represent the style and content of the image:
# Also interesting to see how the result changes if the layers of interest for content and style are changed
# List of Content layers where we will pull our feature maps
'''For VGG: content_layers = ['block5_conv2'] #Conv Layer 14
For VGG: style_layers = ['block1_conv1', 
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1'] #Conv Layers 1,3,5,9,13
'''
#For pretrained model choose:
content_layers = ['conv2d_227']
style_layers = ['conv2d_220','conv2d_222','conv2d_224','conv2d_226']

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)


'''Build the model
The networks in tf.keras.applications are designed so you can easily extract
the intermediate layer values using the Keras functional API.

To define a model using the functional API, specify the inputs and outputs:

model = Model(inputs, outputs)

This following function builds a VGG19 model that returns a list of
intermediate! layer outputs:'''

'''
def vgg_layers(layer_names):
    """ Creates a vgg model that returns a list of intermediate output values.
    Load our model. Load pretrained VGG, trained on imagenet data 
    (which are RGB pictures of random objects, better training at CT data!!)"""
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in layer_names]

    model = tf.keras.Model([vgg.input], outputs)
    return model
'''
def pretrained_layers(layer_names):
    
    pretrained_model = tf.keras.models.load_model('/home/creim/Desktop/Style Transfer/CT-CBCT-ClassifierCNN-on-registrated-Data-deep.h5')
    pretrained_model.pop()
    pretrained_model.trainable = False
    outputs = [pretrained_model.get_layer(name).output for name in layer_names]
    model = tf.keras.Model([pretrained_model.input], outputs)
    return model
#Error: input and filter must have same depth: gray (1) vs. rgb (3)
#need different model with only 1 filter per layer
    
style_extractor = pretrained_layers(style_layers)
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
Build a model that returns the style and content tensors.
When called on an image, this model returns the gram matrix (style) of
the style_layers and content of the content_layers:'''
class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.pretrained_model = pretrained_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.pretrained_model.trainable = False

    def call(self, inputs):
        "Expects float input in [0,1]"
        inputs = inputs*255.0
        #preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.pretrained_model(inputs)
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
#default values were: learning_rate=0.02, beta_1=0.99, epsilon=1e-1
#learning rates of lower that .00 make now sense -> cant get out of local minima
#higher than 0.05 also bad

style_weight = 10000       # default value: 1e-2 -> weight for target CT style 
content_weight = 1      # default value: 1e4 -> weight for CBCT content
total_variation_weight = 0.5  #Should encourage Smoothness, a little smoothness needed.
                              #otherwise it tends to blurr


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


'''
def high_pass_x_y(image):
    x_var = image[:, :, 1:, :] - image[:, :, :-1, :]
    y_var = image[:, 1:, :, :] - image[:, :-1, :, :]

    return x_var, y_var

Initialize the optimization variable: This will be shown in the subplots below
 for the High Frequency artifacts of the style image, thus we initialize it here

x_deltas, y_deltas = high_pass_x_y(content_image)

print('High Frequency Artifacts')
plt.figure(figsize=(14, 10))
plt.subplot(2, 2, 1)
imshow(clip_0_1(2*y_deltas+0.5), "Horizontal Deltas: Original")

plt.subplot(2, 2, 2)
imshow(clip_0_1(2*x_deltas+0.5), "Vertical Deltas: Original")

x_deltas, y_deltas = high_pass_x_y(image)

plt.subplot(2, 2, 3)
imshow(clip_0_1(2*y_deltas+0.5), "Horizontal Deltas: Styled")

plt.subplot(2, 2, 4)
imshow(clip_0_1(2*x_deltas+0.5), "Vertical Deltas: Styled")


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
Which is included in the code below'''

#Optimization Variable - containing the Content Information
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


'''Comparing the styled image to the style target by comparing using different
error methods: pixelwise - MSE, NCC 
               anatomically - SSIM, DICE'''
               
def MSE(target_img,styled_img):
    '''Computing the Mean Squared Error - Using the tf math package'''
    if len(styled_img.shape) > 3:
        styled_img = tf.squeeze(styled_img, axis = 0)
    if len(target_img.shape) > 3:
        target_img = tf.squeeze(target_img, axis = 0)
    error = tf.math.square(tf.math.subtract(target_img, styled_img))
    se = tf.math.reduce_sum(error)
    mse = se/tf.size(error, out_type=tf.dtypes.float32)
    return np.float(mse) #Converting tf.type to np.float for printing'''

    
def NCC(target_img, styled_img):
    '''Computing the Normalized Cross correlation of two images
    which is mathematically the cosine of angle between 
    flattened out image arrays as 1D-vectors'''
    flat_target = tf.reshape(target_img, [-1])
    flat_styled = tf.reshape(styled_img, [-1])
    flat_target = np.array(flat_target)
    flat_styled = np.array(flat_styled)
    dot_product = np.dot(flat_target, flat_styled)
    norm_product = np.linalg.norm(flat_target) * np.linalg.norm(flat_styled)
    ncc = dot_product / norm_product
    return ncc

def SSIM(target_img, styled_img):
    '''Computing the Structural Similarity Index SSIM 
    using the skimage.metrics package'''
    if len(styled_img.shape) > 3:
        styled_img = tf.squeeze(styled_img, axis = 0)
    if len(target_img.shape) > 3:
        target_img = tf.squeeze(target_img, axis = 0)
    target_img = np.array(target_img)
    styled_img = np.array(styled_img)
    ssim = compare_ssim(target_img, styled_img, multichannel=True)
    return ssim   


# Run optimization!!:
start = time.time()

epochs = 15
steps_per_epoch = 100

print('MSE before StyleTransfer: ', MSE(style_image, content_image))
print('NCC before Style Transfer: ', NCC(style_image, content_image))
print('SSIM before Style Transfer: ', SSIM(style_image, content_image))

step = 0
for n in range(epochs):
    for m in range(steps_per_epoch):
        step += 1
        train_step(image)
        print(".", end='')
    display.clear_output(wait=True)
    display.display(tensor_to_image(image))     
    print("Train step: {}".format(step))
    print('MSE after Style Transfer: ', MSE(style_image, image))
    print('NCC after Style Transfer: ', NCC(style_image, image))

end = time.time()

print("Total time: {:.1f}".format(end-start))

print('Parameters used:')
print('Initial Style Weight: ', style_weight)
print('Initial Content Weight: ', content_weight)
print('Total Variation Weight: ', total_variation_weight)
print('MSE before GAN: ', MSE(style_image, content_image_before_GAN))
print('MSE before StyleTransfer: ', MSE(style_image, content_image))
print('MSE after Style Transfer: ', MSE(style_image, image))
print('NCC before GAN: ', NCC(style_image, content_image_before_GAN))
print('NCC before Style Transfer: ', NCC(style_image, content_image))
print('NCC after Style Transfer: ', NCC(style_image, image))
print('SSIM before GAN: ', SSIM(style_image, content_image_before_GAN))
print('SSIM before Style Transfer: ', SSIM(style_image, content_image))
print('SSIM after Style Transfer: ', SSIM(style_image, image))


'Showing the Images together'
plt.subplot(1, 4, 1)
imshow(content_image_before_GAN, 'original cbct')

plt.subplot(1, 4, 2)
imshow(content_image, 'after GAN')

plt.subplot(1, 4, 4)
imshow(image, 'after Style Transfer')

plt.subplot(1, 4, 3)
imshow(style_image, 'Reference CT')


'Finally saving the result'
file_name = '/home/creim/Desktop/Style Transfer/CT styled CBCT Images/pretranslated-now-transfered-cbct2ct.png'
tensor_to_image(image).save(file_name)

try:
    from google.colab import files
except ImportError:
     pass
else:
    files.download(file_name)