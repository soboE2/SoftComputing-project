# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 13:46:50 2016
plus = 10, minus = 11, puta =12,podeljeno 13
@author: SoxBox
"""

import cv2
import collections
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt


bin_n = 16


def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
def image_bin(image_gs):
    ret,image_bin = cv2.threshold(image_gs, 127, 255, cv2.THRESH_BINARY)
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

#Funkcionalnost implementirana u V2
def res_img(a,b):
    img = np.zeros([b+20,a+20,3],dtype=np.uint8)
    img.fill(255) 
    img = invert(image_bin_adaptive(image_gray(img)))
    return img
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
    ret_val = erode(binary_image)
    ##ret_val=dilate(binary_image)
    
    ret_val = invert(ret_val)
    return ret_val
##ivlacenje slika iz mnista
def get_images(training_set):
    """ Return a list containing the images from the MNIST data
    set. Each image is represented as a 2-d numpy array."""
    flattened_images = training_set[0]
    ##img =  np.reshape(flattened_images[0], (-1, 28)) 
    a,b,c = flattened_images.shape
    print a,b,c
    

    ##return  [image_gray(np.reshape(f, (-1, 28))) for f in flattened_images]
 


def select_roi(image_orig,image_bin, num):

    img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = []
    props = []
    sizes = []
    for contour in contours: 
        x,y,w,h = cv2.boundingRect(contour)
        if(w>13 and h>9 and (num == 10 or num ==12) ):
            region = image_bin[y:y+h+1,x:x+w+1]
            props.append( resize_region(region))
            cv2.rectangle(image_orig,(x,y),(x+w,y+h),(0,255,0),2)
            sizes.append(num)
        elif(num == 11 and w>10 and h>5):
            
            region = image_bin[y:y+h+1,x:x+w+1]
            props.append( resize_region(region))
            cv2.rectangle(image_orig,(x,y),(x+w,y+h),(0,255,0),2)
            sizes.append(num)
        elif(num == 13 and w>6 and h>5):
            
            region = image_bin[y:y+h+1,x:x+w+1]
            props.append(resize_region(region))
            cv2.rectangle(image_orig,(x,y),(x+w,y+h),(0,255,0),2)
            sizes.append(num)
              

       
    for idx in range(0, props.__len__()):
        for angle in range(-15,15):
            props.append(rotateImage(props[idx],angle))
            sizes.append(num)
            
    return props, sizes
    
def rotateImage(image, angle):
  image_center = tuple(np.array(image.shape)/2)
  rot_mat = cv2.getRotationMatrix2D(image_center,angle,1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape,flags=cv2.INTER_LINEAR)
  return result      

def getPlus():
      print('Forimram plus niz za obuku i testiranje')
      img = load_image('operators/plus.jpg') 
      img1 = load_image('operators/plus1.jpg') 
      img2 = load_image('operators/plus2.jpg') 
      img_test = load_image('operators/plus_test.jpg')
      img_bin = remove_noise(image_bin_adaptive(image_gray(img)))
      img_bin1 = remove_noise(image_bin_adaptive(image_gray(img1)))
      img_bin2 = remove_noise(image_bin_adaptive(image_gray(img2)))
      img_test1 = remove_noise(image_bin_adaptive(image_gray(img_test)))
      
      
      
      x_train = []
      y_train = []
      x_test = []
      y_test = []
    
      x_train,y_train = select_roi(img,img_bin,10) 
      x_train1,y_train1 = select_roi(img1,img_bin1,10) 
      x_train2,y_train2 = select_roi(img2,img_bin2,10) 
      x_train = np.concatenate((x_train,x_train1), axis = 0)
      x_train = np.concatenate((x_train,x_train2), axis = 0)
      y_train = np.concatenate((y_train,y_train1), axis = 0)
      y_train = np.concatenate((y_train,y_train2), axis = 0)
      x_test,y_test = select_roi(img_test,img_test1,10)
      return x_train,y_train,x_test,y_test

    
    
def getMinus():
      print('Forimram minus niz za obuku i testiranje')
      img = load_image('operators/minus.jpg') 
      img1 = load_image('operators/minus1.jpg') 
      img2 = load_image('operators/minus2.jpg') 
      img_test = load_image('operators/minus_test.jpg')
    
      img_bin = remove_noise(image_bin_adaptive(image_gray(img)))
      img_bin2 = remove_noise(image_bin_adaptive(image_gray(img2)))
      img_test1 = remove_noise(image_bin_adaptive(image_gray(img_test)))
      x_train = []
      y_train = []
      x_test = []
      y_test = []
      
      

      x_train,y_train = select_roi(img,img_bin,11)
      x_train2,y_train2 = select_roi(img2,img_bin2,11) 
      x_train = np.concatenate((x_train,x_train2), axis = 0)
      y_train = np.concatenate((y_train,y_train2), axis = 0)
      x_test,y_test = select_roi(img_test,img_test1,11)
      return x_train,y_train,x_test,y_test
      
    
def getPuta():
      print('Forimram puta niz za obuku i testiranje')
      img = load_image('operators/puta.jpg') 
      img1 = load_image('operators/puta1.jpg') 
      img2 = load_image('operators/puta2.jpg') 
      img_test = load_image('operators/puta_test.jpg')
    
      img_bin = remove_noise(image_bin_adaptive(image_gray(img)))
      img_bin1 = remove_noise(image_bin_adaptive(image_gray(img1)))
      img_bin2 = remove_noise(image_bin_adaptive(image_gray(img2)))
      img_test1 = remove_noise(image_bin_adaptive(image_gray(img_test)))
      x_train = []
      y_train = []
      x_test = []
      y_test = []
      
    
      x_train,y_train = select_roi(img,img_bin,12)
      x_train1,y_train1 = select_roi(img1,img_bin1,12) 
      x_train2,y_train2 = select_roi(img2,img_bin2,12) 
      x_train = np.concatenate((x_train,x_train1), axis = 0)
      x_train = np.concatenate((x_train,x_train2), axis = 0)
      y_train = np.concatenate((y_train,y_train1), axis = 0)
      y_train = np.concatenate((y_train,y_train2), axis = 0)
      x_test,y_test = select_roi(img_test,img_test1,12)
      
      return x_train,y_train,x_test,y_test
      
def getTacka():
      print('Forimram tacku niz za obuku i testiranje')
      img = load_image('operators/tacka.jpg') 
      img1 = load_image('operators/tacka1.jpg') 
      img2 = load_image('operators/tacka2.jpg') 
      img_test = load_image('operators/tacka_test.jpg')
    
      img_bin = remove_noise(image_bin_adaptive(image_gray(img)))
      img_bin1 = remove_noise(image_bin_adaptive(image_gray(img1)))
      img_bin2 = remove_noise(image_bin_adaptive(image_gray(img2)))
      img_test1 = remove_noise(image_bin_adaptive(image_gray(img_test)))
      x_train = []
      y_train = []
      x_test = []
      y_test = []
      
  
      x_train,y_train = select_roi(img,img_bin,13)
      x_train1,y_train1 = select_roi(img1,img_bin1,13) 
      x_train2,y_train2 = select_roi(img2,img_bin2,13) 
      x_train = np.concatenate((x_train,x_train1), axis = 0)
      x_train = np.concatenate((x_train,x_train2), axis = 0)
      y_train = np.concatenate((y_train,y_train1), axis = 0)
      y_train = np.concatenate((y_train,y_train2), axis = 0)
      x_test,y_test = select_roi(img_test,img_test1,13)
      return x_train,y_train,x_test,y_test
      
      

   
def getForTrain():
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    
    x=[]
    y=[]    
    
    x,y,xx,yy = getPlus()
    X_train = x
    y_train = y
    X_test = xx
    y_test = yy
    
    x,y,xx,yy = getMinus()
    X_train = np.concatenate((X_train,x), axis = 0)
    y_train = np.concatenate((y_train,y), axis = 0)
    X_test = np.concatenate((X_test,xx), axis = 0)
    y_test = np.concatenate((y_test,yy), axis = 0)      
      
      
      
    x,y,xx,yy = getPuta()
    X_train = np.concatenate((X_train,x), axis = 0)
    y_train = np.concatenate((y_train,y), axis = 0)
    X_test = np.concatenate((X_test,xx), axis = 0)
    y_test = np.concatenate((y_test,yy), axis = 0)  
    
    x,y,xx,yy = getTacka()
    X_train = np.concatenate((X_train,x), axis = 0)
    
    y_train = np.concatenate((y_train,y), axis = 0)
    X_test = np.concatenate((X_test,xx), axis = 0)
    y_test = np.concatenate((y_test,yy), axis = 0)  
 
 
    print (len(X_train), len(y_train),len(X_test))
    return X_train,y_train,X_test, y_test
getForTrain()