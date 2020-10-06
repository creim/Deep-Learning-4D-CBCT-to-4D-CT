'pix2pix tensorflow'
#code from https://www.tensorflow.org/tutorials/generative/pix2pix
import tensorflow as tf 

import os
import time
from matplotlib import pyplot as plt
from IPython import display
import numpy as np

#Need install tensorboard - via pip install -q -U tensorboard
4
#dataset - facades
#_URL = 'https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/facades.tar.gz'

'''path_to_zip = tf.keras.utils.get_file('facades.tar.gz',
                                      origin=_URL,
                                      extract=True)'''
#path do dataset instead
#PATH = os.path.join(os.path.dirname(path_to_zip), 'facades/')
PATH =  '/home/creim/Desktop/pix2pix-tensorflow-official/datasets/Train-Thorax-5-ohne-Test-mit-wirklich-ohne/'
#PATH = '/home/creim/Desktop/pix2pix-tensorflow/datasets/facades/'

#Parameters
BUFFER_SIZE = 400
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256

EPOCHS = 200
LAMBDA = 200

#Loading labelled images, that are half/half, and declare them as input and real

def load(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_png(image, channels=1)
    
    w = tf.shape(image)[1]
    
    #This is done to half the image from the dataset, in real and input image
    w = w // 2 
    real_image = image[:, w:, :]
    input_image = image[:, :w, :]
    
    #And making sure its float32 type
    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)

    return input_image, real_image
    

#Showing one input image
inp, re = load(PATH+'train/25.png')


# casting to int for matplotlib to show the image
plt.figure()
plt.imshow(np.squeeze(inp, axis=2), cmap='gray')
plt.figure()
plt.imshow(np.squeeze(re, axis=2), cmap='gray')


def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(input_image, [height, width],
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width],
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return input_image, real_image

'''
def random_crop(input_image, real_image):
    cropped_input = tf.image.random_crop(input_image, size= [IMG_HEIGHT, IMG_WIDTH, 1])
    cropped_real = tf.image.random_crop(real_image, size= [IMG_HEIGHT, IMG_WIDTH, 1])
    return cropped_input, cropped_real
'''

def random_crop(input_image, real_image):
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(
        stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 1])
    
    return cropped_image[0], cropped_image[1]
    
    #normalizing the images to [-1,1]
def normalize(input_image, real_image):
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1

    return input_image, real_image
                      
@tf.function()
def random_jitter(input_image, real_image):

    #resizing to bigger size
    input_image, real_image = resize(input_image, real_image, 286, 286)
    
    #randomly cropping to 256x256
    input_image, real_image = random_crop(input_image, real_image)
    
    #random mirroring
    #if tf.random.uniform(()) > 0.5:
    input_image = tf.image.flip_left_right(input_image)
    real_image = tf.image.flip_left_right(real_image)
    input_image = tf.image.random_flip_up_down(input_image)
    real_image = tf.image.random_flip_up_down(real_image)
    
    return input_image, real_image

@tf.function()
def rot(input_image, real_image, k=1):
    rot_input = tf.image.rot90(input_image, k)
    rot_real = tf.image.rot90(real_image, k)
    return rot_input, rot_real
    
    
'''As you can see in the images below that they are going through 
random jittering. Random jittering as described in the paper is to:
Resize an image to bigger height and width
Randomly crop to the target size
Randomly flip the image horizontally'''


#plotting the random jittered images
plt.figure(figsize=(6,6))
for i in range(4):
    rj_inp, rj_re = random_jitter(inp, re)
    plt.subplot(2,2,i+1)
    plt.imshow(np.squeeze(rj_inp, axis=2), cmap='gray')
    plt.axis('off')
plt.show()

def load_image_train(image_file):
    input_image, real_image = load(image_file)
    input_image, real_image = random_jitter(input_image, real_image)
    input_image, real_image = rot(input_image, real_image)
    input_image, real_image = normalize(input_image, real_image)
    
    return input_image, real_image

def load_image_test(image_file):
    input_image, real_image = load(image_file)
    input_image, real_image = resize(input_image, real_image,
                                     IMG_HEIGHT, IMG_WIDTH)
    input_image, real_image = normalize(input_image, real_image)
    
    return input_image, real_image

#Input Pipeline
train_dataset = tf.data.Dataset.list_files(PATH+'train/*.png')
train_dataset = train_dataset.map(load_image_train,
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)

train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)

test_dataset = tf.data.Dataset.list_files(PATH+'test/*.png')
test_dataset = test_dataset.map(load_image_test)
test_dataset = test_dataset.batch(BATCH_SIZE)

'''Build the Generator

The architecture of the generator is a modified U-Net.
Each block in the encoder is (Conv -> Batchnorm -> Leaky ReLU)
Each block in the decoder is (Transposed Conv -> Batchnorm -> Dropout(applied to the first 3 blocks) -> ReLU)
There are skip connections between the encoder and decoder (as in U-Net).'''

OUTPUT_CHANNELS = 1

def downsample(filters, size, apply_batchnorm=True):
    
    initializer = tf.random_normal_initializer(0., 0.02)
    
    result = tf.keras.Sequential()
    result.add(
            tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                   kernel_initializer=initializer, use_bias=False))
    
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
    
    result.add(tf.keras.layers.LeakyReLU())
    
    return result

down_model = downsample(3, 4)
down_result = down_model(tf.expand_dims(inp, 0))
print(down_result.shape)

#Should be (1,128,128,3)

def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    
    result = tf.keras.Sequential()
    result.add(
            tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            use_bias=False))
    
    result.add(tf.keras.layers.BatchNormalization())
    
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))
    
    result.add(tf.keras.layers.ReLU())
    
    return result

up_model = upsample(3,4)
up_result = up_model(down_result)
print (up_result.shape)


def Generator():
  inputs = tf.keras.layers.Input(shape=[IMG_WIDTH,IMG_HEIGHT,1])

  down_stack = [
    downsample(64, 4, apply_batchnorm=False), # (bs, 128, 128, 64)
    downsample(128, 4), # (bs, 64, 64, 128)
    downsample(256, 4), # (bs, 32, 32, 256)
    downsample(512, 4), # (bs, 16, 16, 512)
    downsample(512, 4), # (bs, 8, 8, 512)
    downsample(512, 4), # (bs, 4, 4, 512)
    downsample(512, 4), # (bs, 2, 2, 512)
    downsample(512, 4), # (bs, 1, 1, 512)
  ]

  up_stack = [
    upsample(512, 4, apply_dropout=True), # (bs, 2, 2, 1024)
    upsample(512, 4, apply_dropout=True), # (bs, 4, 4, 1024)
    upsample(512, 4, apply_dropout=True), # (bs, 8, 8, 1024)
    upsample(512, 4), # (bs, 16, 16, 1024)
    upsample(256, 4), # (bs, 32, 32, 512)
    upsample(128, 4), # (bs, 64, 64, 256)
    upsample(64, 4), # (bs, 128, 128, 128)
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh') # (bs, 256, 256, 3)

  x = inputs

  # Downsampling through the model
  skips = []
  for down in down_stack:
    x = down(x)
    skips.append(x)

  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    x = tf.keras.layers.Concatenate()([x, skip])

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)

generator = Generator()

#Plotting a scheme of the generator model
#tf.keras.utils.plot_model(generator, show_shapes=True, dpi=64)

#Plotting the generator output
gen_output = generator(inp[tf.newaxis,...], training=False)
plt.imshow(np.squeeze(gen_output[0,...], axis=2), cmap='gray')

'''Generator loss
- It is a sigmoid cross entropy loss of the generated images and an array of ones.

- The paper also includes L1 loss which is MAE (mean absolute error) between 
the generated image and the target image.

- This allows the generated image to become structurally similar to the target image.

- The formula to calculate the total generator loss is 
generator loss = gan_loss + LAMBDA * l1_loss, 
where LAMBDA = 100. This value was decided by the authors of the paper.'''



def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    # mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    
    total_gen_loss = gan_loss + (LAMBDA *l1_loss)
    
    return total_gen_loss, gan_loss, l1_loss

'''Build the Discriminator

-> The Discriminator is a PatchGAN.
-> Each block in the discriminator is (Conv -> BatchNorm -> Leaky ReLU)
-> The shape of the output after the last layer is (batch_size, 30, 30, 1)
-> Each 30x30 patch of the output classifies a 70x70 portion of the input image (such an architecture is called a PatchGAN).
-> Discriminator receives 2 inputs.
    -> Input image and the target image, which it should classify as real.
    -> Input image and the generated image (output of generator), which it should classify as fake.
    -> We concatenate these 2 inputs together in the code (tf.concat([inp, tar], axis=-1))
'''

def Discriminator():
  initializer = tf.random_normal_initializer(0., 0.02)

  inp = tf.keras.layers.Input(shape=[IMG_WIDTH, IMG_HEIGHT, 1], name='input_image')
  tar = tf.keras.layers.Input(shape=[IMG_WIDTH, IMG_HEIGHT, 1], name='target_image')

  x = tf.keras.layers.concatenate([inp, tar]) # (bs, 256, 256, channels*2)

  down1 = downsample(64, 4, False)(x) # (bs, 128, 128, 64)
  down2 = downsample(128, 4)(down1) # (bs, 64, 64, 128)
  down3 = downsample(256, 4)(down2) # (bs, 32, 32, 256)

  zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3) # (bs, 34, 34, 256)
  conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1) # (bs, 31, 31, 512)

  batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

  leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

  zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) # (bs, 33, 33, 512)

  last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2) # (bs, 30, 30, 1)

  return tf.keras.Model(inputs=[inp, tar], outputs=last)

discriminator = Discriminator()

#Plotting the discriminator model
#tf.keras.utils.plot_model(discriminator, show_tapes=True, dpi=64)

disc_out = discriminator([inp[tf.newaxis,...], gen_output], training=False)
plt.imshow(disc_out[0,...,-1], vmin=-20, cmap='RdBu_r')
plt.colorbar()

'''Discriminator loss       
-> The discriminator loss function takes 2 inputs; real images, generated images
-> real_loss is a sigmoid cross entropy loss of the real images and 
an array of ones(since these are the real images)
-> generated_loss is a sigmoid cross entropy loss of the generated images and
an array of zeros(since these are the fake images)
-> Then the total_loss is the sum of real_loss and the generated_loss'''

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    
    total_disc_loss = real_loss + generated_loss
    
    return total_disc_loss

#Define the Optimizers and Checkpoint-saver
generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

checkpoint_dir = PATH+'/training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

'''Generate Images

Write a function to plot some images during training.

-> We pass images from the test dataset to the generator.
-> The generator will then translate the input image into the output.
-> Last step is to plot the predictions and voila!'''

def generate_images(model, test_input, tar):
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15,15))
    
    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']
    
    for i in range(3):
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(np.squeeze(display_list[i]), cmap='gray')
        plt.axis('off')
        plt.savefig(PATH+'/Results/testing_samples/'+str(title[i])+'.png')
        #plt.show()
    
#Image generated directly in training loop

    
#Training
'''
-> For each example input generate an output.
-> The discriminator receives the input_image and the generated image as the first input. The second input is the input_image and the target_image.
-> Next, we calculate the generator and the discriminator loss.
-> Then, we calculate the gradients of loss with respect to both the generator and the discriminator variables(inputs) and apply those to the optimizer.
-> Then log the losses to TensorBoard.'''



import datetime
log_dir="logs/"

summary_writer = tf.summary.create_file_writer(
        log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%H%S"))

@tf.function
def train_step(input_image, target, epoch):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)
        
        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)
        
        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
        
    generator_gradients = gen_tape.gradient(gen_total_loss,
                                            generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                                 discriminator.trainable_variables)
    
    generator_optimizer.apply_gradients(zip(generator_gradients,
                                            generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                discriminator.trainable_variables))
    
    with summary_writer.as_default():
        tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
        tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
        tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
        tf.summary.scalar('disc_loss', disc_loss, step=epoch)

'''The actual training loop:
-> Iterates over the number of epochs.
-> On each epoch it clears the display, and runs generate_images to show it's progress.
-> On each epoch it iterates over the training dataset, printing a '.' for each example.
-> It saves a checkpoint every 20 epochs.'''

def fit(train_ds, epochs, test_ds):
    for epoch in range(epochs):
        start = time.time()
        
        display.clear_output(wait=True)
        
        for example_input, example_target in test_ds.take(1):
            prediction = generator(example_input, training=True)
            plt.figure(figsize=(15,15))
    
            display_list = [example_input[0], example_target[0], prediction[0]]
            title = ['Input Image', 'Ground Truth', 'Predicted Image']
    
            for i in range(3):
                plt.figure(figsize=(8,8))
                plt.imshow(np.squeeze(display_list[i]), cmap='gray')
                plt.axis('off')
                plt.savefig(PATH+'/Results/training_samples/'+str(epoch)+str(title[i])+'.png')
                #plt.show()
                plt.close


        print("Epoch: ", epoch)
        
        # Train
        for n, (input_image, target) in train_ds.enumerate():
          print(".")
          if (n+1) % 100 == 0:
            print()
          train_step(input_image, target, epoch)
        print()
        
        # saving (checkpoint) the model every 20 epochs
        if (epoch + 1) % 20 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)
        
        print('Time taken for epoch {} is {} sec\n' .format(epoch +1, time.time()-start))
    checkpoint.save(file_prefix = checkpoint_prefix)
   
'''To monitor the training progress on TensorBoard launch it
via following code lines 
%load_ext tensorboard
%tensorboard --logdir {log_dir}'''

#Run training loop via
#fit(train_dataset, EPOCHS, test_dataset)


#Restore the latest checkpoint and test


# restoring the latest checkpoint in checkpoint_dir
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

#Generate using test dataset

# Run the trained model on a few examples from the test dataset

for test_input, test_target in test_dataset:
            prediction = generator(test_input, training=False)
            plt.figure(figsize=(15,15))
    
            display_list = [test_input[0], test_target[0], prediction[0]]
            title = ['Input Image', 'Ground Truth', 'Predicted Image']
            
            iterator = 0
            
            for i in range(3):
                plt.figure(figsize=(8,8))
                plt.imshow(np.squeeze(display_list[i]), cmap='gray')
                plt.axis('off')
                plt.savefig(PATH+'/Results/testing_samples/'+str(iterator)+str(title[i])+'.png')
                #plt.show()
                plt.close
                iterator += 1
  
  
