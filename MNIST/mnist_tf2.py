# Developer Kinjal Dasgupta

import glob
import imageio
#import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time
import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Add a channels dimension
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(32)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

class MyModel(tf.keras.Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.conv1 = Conv2D(32, 3, activation='relu')
    self.flatten = Flatten()
    self.d1 = Dense(128, activation='relu')
    self.d2 = Dense(10)

  def call(self, x):
    x = self.conv1(x)
    x = self.flatten(x)
    x = self.d1(x)
    return self.d2(x)

model = MyModel()
loss_ob = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
checkpoint_dir = '/Data3/'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

@tf.function
def train_step(images, labels):
  with tf.GradientTape() as tape:
    # training=True is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(images, training=True)
    loss = loss_ob(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  train_accuracy(labels, predictions)

@tf.function
def test_step(images, labels):
  # training=False is only needed if there are layers with different

  predictions = model(images, training=False)
  t_loss = loss_ob(labels, predictions)

  test_loss(t_loss)
  test_accuracy(labels, predictions)
  #train_accuracy(labels, predictions)


EPOCHS = 50
def train(dataset, epochs):
  for epoch in range(epochs):
    start = time.time()

    for images, labels in train_ds:
      #train_step(images,labels)

    for test_images, test_labels in test_ds:
    	test_step(test_images, test_labels)
    visual = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
     #Save the model every 15 epochs
    if (epoch + 1) % 15 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)
      print('checkpoint_saved')

    print(visual.format(epoch + 1, train_loss.result(), train_accuracy.result(), test_loss.result(), test_accuracy.result()))


checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
train(train_ds, EPOCHS)

#checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
