
from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import createOperators as cro
np.random.seed(1337)  # for reproducibility
import cv2
from keras.datasets import mnist
import math
from keras.models import Sequential
from keras.layers.core import Dense, Activation,Dropout
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils

'''
    Jednostavna vestacka neuronska mreza za MNIST dataset
'''
def display_image(image, color= False):
    if color:
        plt.imshow(image)
    else:
        plt.imshow(image, 'gray')
def matrix_to_vector(image):
    return image.flatten()  
def invert(image):
    return 255-image
def resize_region(region):
    #img = res_img(w,h) 
    #img[10:10+h+1,10:10+w+1]=  region
    resized = cv2.resize(region,(28,28), interpolation = cv2.INTER_CUBIC)
    return resized
def dilate(image):
    kernel = np.ones((3,3)) # strukturni element 3x3 blok
    return cv2.dilate(image, kernel, iterations=1)
def scale_to_range(image):
    return image / 255    
def image_bin(image_gs):
    ret,image_bin = cv2.threshold(image_gs, 127, 255, cv2.THRESH_BINARY)
    return image_bin
def prepare_for_ann(regions):
    ready_for_ann = []
    i = 0;
    for region in regions:
        i+=1
        print(i)
        reg_bin = image_bin(region)
        x,y,w,h = crop(reg_bin)
        reg = reg_bin[y:y+h+1,x:x+w+1]
        ready_for_ann.append(scale_to_range(resize_region(reg)))
    return np.array(ready_for_ann)   

def prepare_for_ann_add(regions):
    ready_for_ann = []
    for region in regions:
        ready_for_ann.append(scale_to_range(region))
    return np.array(ready_for_ann)        
        
def image_gray(image):
    img = cv2.GaussianBlur(image,(3,3),0)
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
def gety(img):#y kordinata slike
    x = 0    
    for i in range(0,27):
        for y in range(0,27):
            if( img[i][y]>0):
                x = i
                return x;
         
def getx(img): #x koridnata slike
    j = 0
    for i in range(0,27):
        for y in range(0,27):
            if( img[y][i]>0):
                j = i
                return j;
def geth(img,hy):#visina
    for i in range(27,0,-1):
        for y in range(27,0,-1):        
            if(img[i][y]>0):
                return i-hy
def getw(img,hw):#sirina
    for i in range(27,0,-1):
        for y in range(27,0,-1):        
            if(img[y][i]>0):
                return i-hw
def crop(img):#crop image
    x = getx(img)
    y = gety(img)
    w = getw(img,x)
    h = geth(img,y)
    return x,y,w,h
            
# broj primeraka za SGD
batch_size = 128

# broj izlaza (klasa) - 10 cifara
nb_classes = 14

# broj epoha
nb_epoch = 10



# podaci: skup za obucavanje (60k uzoraka) i skup za validaciju/testiranje (10k uzoraka)
(X_train, y_train), (X_test, y_test) = mnist.load_data()
#xaa = prepare_for_ann(X_train)
X_train = prepare_for_ann(X_train)
X_test = prepare_for_ann(X_test)

X_train1,y_train1,X_test1,y_test1 = cro.getForTrain()
X_train1 = prepare_for_ann_add(X_train1)
X_test1 = prepare_for_ann_add(X_test1)

X_train = np.concatenate((X_train, X_train1), axis = 0)
y_train = np.concatenate((y_train, y_train1), axis = 0)
X_test = np.concatenate((X_test, X_test1), axis = 0)
y_test = np.concatenate((y_test,y_test1), axis = 0)





# reshape iz matrice 28x28 u vektor sa 784 elemenata
X_train = X_train.reshape(90938, 784)
X_test = X_test.reshape(16200, 784)



# pretvaranje u float zbog narednog koraka
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()
model.add(Dense(512, input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(14))
model.add(Activation('softmax'))



rms = RMSprop()

model.compile(loss='categorical_crossentropy', optimizer=rms)

# obucavanje neuronske mreze
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, validation_data=(X_test, Y_test))

# nakon obucavanje testiranje
score = model.evaluate(X_test, Y_test, show_accuracy=True)
print('Test score:', score[0])
print('Test accuracy:', score[1])

model.save_weights('model_weight1.hdf5',overwrite=True)
