from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten,Activation,Add
from keras.layers import Conv2D, MaxPooling2D,Dropout
from keras import optimizers
from keras.datasets import cifar10
from keras import backend as K
from keras.layers import Input, Dense
from keras.models import Model

batch_size = 128
num_classes = 10
epochs =50
#load the data 
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print('x_test shape:',x_test.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
x_train=x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
#building the cnn network
x = Input(shape=(32,32,3))
c1=Conv2D(32, (5, 5), padding='same')(x)
r1=Activation('relu')(c1)
m1=MaxPooling2D((2,2))(r1)
d1=Dropout(0.25)(m1)
c2=Conv2D(64, (3, 3), padding='same')(d1)
r2=Activation('relu')(c2)
d2=Dropout(0.25)(r2)
c3=Conv2D(64,(3,3),padding='same')(d2)
r3=Activation('relu')(c3)
s1=keras.layers.add([c2,r3])
c4=Conv2D(64,(3,3),padding='same')(r3)
r4=Activation('relu')(c4)
c5=Conv2D(64,(3,3),padding='same')(r4)
r5=Activation('relu')(c5)
s2=keras.layers.add([c4,r5])
m3=MaxPooling2D((2,2))(r5)
d2=Dropout(0.25)(m3)
f1=Flatten()(d2)
d1=Dense(512, activation='relu')(f1)
predictions = Dense(10, activation='softmax')(d1)
model = Model(inputs=x, outputs=predictions)
model.compile(loss=keras.losses.categorical_crossentropy,optimizer='sgd',metrics=['accuracy'])#model complie & testing
model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
#save model and weights
model_json=model.to_json()
with open("modelcifar10_cnn.json","W") as json file:
    json_file.write(model_json)
    model.save_weights("modelcifar10_cnn.h5")
    print("saved model to disk")
