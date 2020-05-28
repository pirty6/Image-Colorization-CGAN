import time
import os
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import tensorflow as tf

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, Input, Dropout, Concatenate
from tensorflow.keras.layers import Flatten, Dense, Activation
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
from skimage.io import imsave
from skimage import img_as_ubyte

# Path where the files are located.
PATH = './gdrive/My Drive/CGAN/'
DATASET_PATH = PATH + 'Images/'
TRAIN_PATH = DATASET_PATH + 'Train/'
TEST_PATH = './gdrive/My Drive/CGAN/Images/Test_4/'
RESULTS_PATH = PATH + 'Test_2/'
WEIGHTS_PATH = PATH + 'temp/generated_epoch4.hdf5'

# Constants
BATCH_SIZE = 32
EPOCHS = 100
NO_IMAGES = 9294
ROWS = 224
COLS = 224
CHANNELS = 2
IMAGE_SHAPE = (ROWS, COLS, CHANNELS)
G_IMAGE_SHAPE = (ROWS, COLS, 3)
D_INPUT_SHAPE = (ROWS, COLS, 2)
GF = 64
DF = 64


# Construct the generator model
def construct_generator(image_shape, gf):
  mod = Sequential()
  # Encoder, Uses ResNet50 as transfer learning and removes the last layers from the classifier
  mod.add(ResNet50(include_top=False, weights='imagenet', input_shape=(224,224,3)))
  # Decoder
  mod.add(Conv2D(filters=256, kernel_size=(3,3),
                            activation='relu', padding='same'))
  mod.add(Conv2D(filters=128, kernel_size=(3,3),
                          activation='relu', padding='same'))
  mod.add(UpSampling2D((2, 2)))
  mod.add(Conv2D(filters=64, kernel_size=(3,3),
                          activation='relu', padding='same'))
  mod.add(UpSampling2D((2, 2)))
  mod.add(Conv2D(filters=32, kernel_size=(3,3),
                          activation='relu', padding='same'))
  mod.add(UpSampling2D((2, 2)))
  mod.add(Conv2D(filters=16, kernel_size=(3,3),
                          activation='relu', padding='same'))
  mod.add(UpSampling2D((2, 2)))
  mod.add(Conv2D(filters=2, kernel_size=(3, 3),
                          activation='tanh', padding='same'))
  mod.add(UpSampling2D((2, 2)))
  mod.summary()
  # ResNet = layer[0] is not going to be trainable
  mod.layers[0].trainable = False
  mod.compile(optimizer='Adam', loss='mse', metrics=['accuracy'])
  return mod

# Construct the PatchGan discriminator model
def construct_discriminator(image_shape, df):
  def conv2d(layer_input, filters, strides, f_size=4, bn=True):
    d = Conv2D(filters, kernel_size=f_size, strides=strides, padding='same')(layer_input)
    d = LeakyReLU(alpha=0.2)(d)
    if bn:
      d = BatchNormalization(momentum=0.8)(d)
    return d

  img_A = Input(image_shape)
  img_B = Input((ROWS, COLS, 3))

  d0 = Concatenate(axis=-1)([img_A, img_B])

  d1 = conv2d(d0, df, 2, bn=False)
  d2 = conv2d(d1, df * 2, 2)
  d3 = conv2d(d2, df * 4, 2)
  d4 = conv2d(d3, df * 8, 1) 

  validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)
  model = Model([img_A, img_B], validity)
  model.summary()
  optimizer = Adam(0.0002, 0.5)
  model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
  return model


# Uses keras datagenerator to get the images from the source folder and resize them to 
# images size 224x224
def load_dataset(dataset_path, batch_size, image_shape)
 data = ImageDataGenerator(rescale = 1. / 255)
 train_data = data.flow_from_directory(dataset_path,
                                       target_size = (image_shape[0], image_shape[1]),
                                       batch_size = batch_size, 
                                       class_mode = None,
                                       subset = 'training')
 return train_data

# Function that turns RGB images to the CIELab color space
def to_LAB(train_data):
  X = []
  Y = []
  for i, img in enumerate(train_data):
    # Turn image to lab
    lab = rgb2lab(img)
    # Get the L layer
    x = lab[:,:,0]
    # Stacks the L layer to create a 3 channel image to satisfy ResNet50 requirements
    x = gray2rgb(x)
    # Append the L layer to the X/lightness vector
    X.append(x)
    # Append the a and b layers to the Y vector
    Y.append(lab[:,:,1:] / 128)
  X = np.array(X)
  print(X.shape)
  Y = np.array(Y)
  return X, Y

def train(batch_size, epochs, image_shape, path, dataset_path):
  # Construct the generator
  generator = construct_generator(image_shape, GF)
  # Construct the discriminator
  discriminator = construct_discriminator(image_shape, DF)
 
  # Defines the input layer for the AB image
  img_A = Input((ROWS, COLS, 2))

  # Defines the input layer for the L image
  img_B = Input((ROWS, COLS, 3))

  # Defines that the generator is going to be fed the img_B and create an AB
  # prediction
  fake_A = generator(img_B)

  discriminator.trainable = False

  # Defines that the discriminator is going to be fed an AB predictions alongside
  # with the real AB image and it will return the validity of the prediction
  valid = discriminator([fake_A, img_B])

  # Defines the gan model and its inputs and outputs 
  gan = Model([img_A, img_B], [valid, fake_A])
  optimizer =Adam(0.0002, 0.5)
  gan.compile(loss=['mse', 'mae'], loss_weights=[1,100], optimizer=optimizer, metrics=None)

  # Loads the dataset
  dataset = load_dataset(dataset_path, batch_size, image_shape)

  # Get the number of batches relative to the no of images
  number_of_batches = int(NO_IMAGES / batch_size)

  adversarial_loss = np.empty(shape=1)
  discriminator_loss = np.empty(shape=1)
  batches = np.empty(shape=1)

  plt.ion()

  current_batch = 0
  for epoch in range(89, epochs):
    print("Epoch " + str(epoch + 1) + "/" + str(epochs) + ":")
    for batch_number in range(number_of_batches):
      start_time = time.time()

      # Get RGB images from dataset
      rgb_images = dataset.next()

      #img_B     img_A
      X_train_L, X_train_AB = to_LAB(rgb_images)

      # Get the current batch size, because the last batch size is not always the
      # same size as the others
      current_batch_size = rgb_images.shape[0]

      # get the size of a half batch
      half_batch_size = (int)(current_batch_size / 2)

      # Generate Images
      generated_colorized_images = generator.predict(X_train_L[:half_batch_size])

      # Create the labels
      # real_labels = np.ones(half_batch_size)
      # fake_labels = np.zeros(half_batch_size)
      real_labels = np.ones((half_batch_size,) + (28, 28, 1))
      fake_labels = np.zeros((half_batch_size,) + (28,28,1))
      # real_labels = (np.ones(half_batch_size) -
      #                 np.random.random_sample(half_batch_size) * 0.2) 
      # fake_labels = np.random.random_sample(half_batch_size) * 0.2

      # Train Discriminator
      discriminator.trainable = True
      
      d_loss = discriminator.train_on_batch([X_train_AB[:half_batch_size], X_train_L[:half_batch_size]], real_labels)
      d_loss += discriminator.train_on_batch([generated_colorized_images, X_train_L[:half_batch_size]], fake_labels)
      discriminator_loss = np.append(discriminator_loss, d_loss)

      #Train Generator
      discriminator.trainable = False
      real_labels = np.ones((current_batch_size,) + (28, 28, 1))
      # real_labels = np.ones(current_batch_size)
      # real_labels = (np.ones(current_batch_size) -
      #                 np.random.random_sample(current_batch_size) * 0.2)
      g_loss = gan.train_on_batch([X_train_AB, X_train_L], [real_labels, X_train_AB])

      # Get the adversarial loss
      adversarial_loss = np.append(adversarial_loss, g_loss)
      batches = np.append(batches, current_batch)

      # Each 50 batches show and save images
      print("Current Batch: ", (current_batch + 1))
      if((batch_number + 1) % 50 == 0 and current_batch_size == batch_size):
          print("Saving Images...")
          sample_images(epoch, batch_number, rgb_images, generator)
        
      if epoch + 1 == epochs and current_batch_size == batch_size:
          final_generated_images = generated_colorized_images

      time_elapsed = time.time() - start_time

      # Display and plot the results
      print("     Batch " + str(batch_number + 1) + "/" +
            str(number_of_batches) +
            " generator loss | discriminator loss : " +
            str(g_loss) + " | " + str(d_loss) + ' - batch took ' +
            str(time_elapsed) + ' s.')

      current_batch += 1

    # Save the model weights each 5 epochs
    if (epoch + 1) % 5 == 0:
        discriminator.trainable = True
        generator.save(PATH + 'temp/generated_epoch' + str(epoch) + '.hdf5')
        discriminator.save(PATH + 'temp/discriminator_epoch' +
                            str(epoch) + '.hdf5')

# Function that saves the predicted images
def sample_images(epoch, batch_i, images, generator):
  # Checks if the results path exist if not it is created
  os.makedirs(RESULTS_PATH, exist_ok=True)
  for idx, img in enumerate(images):
    # RGB image to LAB
    lab = rgb2lab(img)
    # Get L layer
    l = lab[:,:,0]
    # Stack the X vector to create a 3 channel image
    L = gray2rgb(l)
    # Reshape the image to fulfill the Generator's requirements
    L = L.reshape((1,224,224,3))
    # Get the AB prediction
    ab = generator.predict(L)
    # Transform from LAB to RGB
    ab = ab * 128
    cur = np.zeros((224,224,3))
    cur[:,:,0] = l
    cur[:,:,1:] = ab
    image = lab2rgb(cur)
    # Save image
    imsave(RESULTS_PATH + str(epoch) + '_' + str(batch_i) + '_' + str(idx) + '.jpg', img_as_ubyte(image))

# Function that loads the generators weights and predict a series of images
def test():
  generator = construct_generator(G_IMAGE_SHAPE, GF)
  generator.load_weights(PATH + 'temp/generated_epoch99.hdf5')
  dataset = load_dataset(TEST_PATH, 335, G_IMAGE_SHAPE)
  rgb_images = dataset.next()
  sample_images(0, 0, rgb_images, generator)


if __name__ == "__main__":
  #train(BATCH_SIZE, EPOCHS, IMAGE_SHAPE, PATH, TRAIN_PATH)
  test()
  print("Finished")
