# -*- coding: utf-8 -*-
#import potrebnih biblioteka
import cv2
import collections
import numpy as np
import scipy as sc
import pylab
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# k-means
from sklearn.cluster import KMeans

# keras
from keras.models import Sequential
from keras.layers.core import Dense,Activation,Dropout
from keras.optimizers import SGD,RMSprop
from keras.utils import np_utils

import matplotlib.pylab as pylab
pylab.rcParams['figure.figsize'] = 16, 12 # za prikaz većih slika i plotova, zakomentarisati ako nije potrebno


batch_size = 128



def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
def image_gray(image):
    #img = cv2.GaussianBlur(image,(3,3),0)
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
def image_bin(image_gs):
    ret,image_bin = cv2.threshold(image_gs, 180, 255, cv2.THRESH_BINARY)
    return image_bin
def image_bin_adaptive(image_gs):
    image_bin = cv2.adaptiveThreshold(image_gs, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 35, 10)
    return image_bin
def invert(image):
    return 255-image
def display_image(image, color= False):
    if color:
        plt.imshow(image)
    else:
        plt.imshow(image, 'gray')
def dilate(image):
    kernel = np.ones((3,3)) # strukturni element 3x3 blok
    return cv2.dilate(image, kernel, iterations=1)
def erode(image):
    kernel = np.ones((3,3)) # strukturni element 3x3 blok
    return cv2.erode(image, kernel, iterations=1)

def resize_region(region):

    resized = cv2.resize(region,(28,28), interpolation = cv2.INTER_CUBIC)
    return resized
    
def scale_to_range(image):
    return image / 255
def matrix_to_vector(image):
    return image.flatten()
def prepare_for_ann(regions):
    ready_for_ann = []
    for region in regions:
        ready_for_ann.append(matrix_to_vector(scale_to_range(region)))
    return ready_for_ann
def convert_output(outputs):
    return np.eye(len(outputs))
def winner(output):
    return max(enumerate(output), key=lambda x: x[1])[0]

def remove_noise(binary_image):
    ret_val = erode(dilate(erode(erode(binary_image))))
    ret_val = invert(ret_val)

    return ret_val

def select_roi(image_orig, image_bin):

    img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = []
    props = []
    sizes = []
    for contour in contours: 
        x,y,w,h = cv2.boundingRect(contour)
        if(w>5 and h>5):
            region = image_bin[y:y+h+1,x:x+w+1]
            props.append((x, w*h, resize_region(region)))
            cv2.rectangle(image_orig,(x,y),(x+w,y+h),(0,255,0),2)
    
    props = sorted(props)   
    sizes = [size for x,size,region in props]
    regions = [region for x,size,region in props]     
    return image_orig, regions, sizes
    



def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def display_result(results, alphabet, validation):
    stringres = []
  
    for idx,r in enumerate(results):
        if(len(stringres) == 0):
            if(is_number(alphabet[winner(results[idx+1])])):
                stringres.append(alphabet[winner(r)])
            else:
                stringres.append(alphabet[winner(r)]+".0")
        else:
            win =alphabet[winner(r)]  
            if(win == '.'):
                if(stringres[-1]=="."):
                    stringres[-1]="/"
                else:
                   stringres.append(".")                 
            else:   
                if(stringres[-1]=="."):
                    stringres[-1]="*"
                    
                if(win == "+" or win =="-"):  
                    stringres.append(win)
                else:
                    if(len(results)-1 > idx):
                        if(is_number(alphabet[winner(results[idx+1])])):
                            stringres.append(win)
                        else:
                            stringres.append(win+".0")
                    else:
                         stringres.append(win+".0")
    inpt = ''.join(stringres)
    print "Izraz: " +validation
    print"Prepoznao: " + inpt
    print "Poklapanje: " + `inpt == validation`

    return inpt
    
      
def createModel():
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
    # kompajliranje modela (Theano) - optimizacija svih matematickih izraza
    model.compile(loss='categorical_crossentropy', optimizer=rms)
    return model


def prepareAnn(shapes):
    ret =np.asarray(shapes)
    m,n,e = ret.shape
    ret = ret.reshape(m, 784)
    ret = ret.astype("float32")
    ret /= 255  
    return ret
    
def getExpressions():
    p1= "6.0+5.0-2.0/1.0"
    p2 = "9.0+4.0+3.0-8.0"
    p3 = "7.0-3.0+15.0"
    p4 = "8.0+9.0-16.0+5.0"
    p5 = "626.0+321.0*7.0/3.0"
    p6 = "252.0+83.0*6.0"
    p7 = "23.0-8.0+6.0*5.0-633.0"
    p8 = "369.0+825.0"
    p9 = "245.0/734.0"
    p10 = "4.0*8.0+7.0-4.0"
    p11 = "5.0+16.0-438.0"    
    a = np.array([p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11])
    return a
   
def start():
    model = createModel()
    model.load_weights('model_weight1.hdf5')      
    alphabet = ['0','1','2','3','4','5','6','7','8','9','+','-','*','.']
    valarr = getExpressions();
   
    for idx in range(1,12):
        print "Slika broj " + `idx`
        path = "tests/"
        path +=(`idx`+".jpg")
        img_test = load_image(path)
        img_test_bin = remove_noise(image_bin_adaptive(image_gray(img_test)))
        sel_img_test, shapes, sizes = select_roi(img_test.copy(), img_test_bin)
        
        inputs = prepare_for_ann(shapes)
        results = model.predict(np.array(inputs, np.float32))
        dis_res = display_result(results, alphabet,valarr[idx-1])
        try:
            print "Resenje: " + `eval(dis_res)` + "\n"
        except SyntaxError:
            print "Neispravan izraz " + dis_res + ", pokusajte ponovo"
            
        
start()
