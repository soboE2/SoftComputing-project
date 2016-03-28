# -*- coding: utf-8 -*-
#import potrebnih biblioteka
import cv2
import collections
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt



# k-means
from sklearn.cluster import KMeans

# keras
from keras.models import Sequential
from keras.layers.core import Dense,Activation,Dropout
from keras.optimizers import SGD,RMSprop
from keras.utils import np_utils

import matplotlib.pylab as pylab
pylab.rcParams['figure.figsize'] = 16, 12 # za prikaz većih slika i plotova, zakomentarisati ako nije potrebno

# broj primeraka za SGD
batch_size = 128

# broj izlaza (klasa) - 10 cifara
nb_classes = 14
scale = 1
delta = 0
ddepth = cv2.CV_16S
# broj epoha
nb_epoch = 20
#Funkcionalnost implementirana u V1

myStack = []
topStack = 0
currentChar = 0
currentPos = 0

def priority(inpt):
    if((inpt == '+') | (inpt =='-') ):
        return 1
    elif ((inpt == '*') | (inpt == '/')):
        return 2
    else :
        return -1

def isStackEmpty():
    if (myStack.__len__() == 0):
        return True
    else:
        return False
         
def push(n):
    myStack.append(n)
 
 
def top():
    return myStack[-1]
 
 
def pop():
    if(~isStackEmpty()):
        znak = myStack[-1]
        myStack.pop()
        return znak

      
    
def inputsI(postfix,currentChar):
   
    znak = []
    broj = []
    ##enum Type tip=0;
    if (postfix[currentChar]=='.'):
        currentChar+=1

    if((postfix[currentChar]=='+') | (postfix[currentChar]=='-') | (postfix[currentChar]=='*') | (postfix[currentChar]=='/')):
        znak.append(postfix[currentChar])
        currentChar +=1
        return znak,currentChar
    else:
        
        while((postfix[currentChar] != '.') & (postfix[currentChar] != '+') & (postfix[currentChar] != '-')  
            & (postfix[currentChar]!='*') & (postfix[currentChar]!='/') ):
            broj.append(postfix[currentChar])
            currentChar+=1
            if(currentChar == postfix.__len__()):
                break

        return broj,currentChar
 

def infix2Postfix(infix,currentChar,currentPos,topStack):
    output=[None] * 100
    x =[]
    znak= []
    i=0
    r=0
    brojac=0
    x,currentChar=inputsI(infix,currentChar)
    while(brojac<infix.__len__()):
 
        if ((x[0]!='+') & (x[0]!='-') & (x[0]!='*') & (x[0]!='/') ):
            r=0
            for i in range(0, x.__len__()):
                output[currentPos]=x[i]
                currentPos+=1
                r+=1
     
            output[currentPos]='.'
            currentPos+=1
            brojac+=r

        else :
            while(isStackEmpty() == False & priority(x[0])<=priority(top())):
               znak=pop();
               output[currentPos]=znak
               currentPos+=1
               output[currentPos]='.'
               currentPos+=1
            push(x[0])
            brojac+=1
        if(currentChar < infix.__len__()):
            x,currentChar=inputsI(infix,currentChar)
     
    while(isStackEmpty() == False):
        znak=pop()
        output[currentPos]=znak
        currentPos +=1
        output[currentPos]='.'
        currentPos +=1
 
 
    currentPos +=1
    otp = []
    for r in range(0,output.__len__()):
        if(output[r]!=None):
            otp.append(output[r])
        else:
            break
    izlaz =  ''.join(otp)   
    return izlaz

 



def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
def image_gray(image):
    img = cv2.GaussianBlur(image,(3,3),0)
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
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
def resize_region(region,w,h):
    img = res_img(w,h) 
    img[10:10+h+1,10:10+w+1]=  region
    resized = cv2.resize(img,(28,28), interpolation = cv2.INTER_CUBIC)
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
    #ret_val = erode(binary_image)
    #ret_val=dilate(binary_image)
    
    ret_val = invert(binary_image)
    ret_val = dilate(ret_val)
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
 


def select_roi(image_orig, image_bin):
    '''Oznaciti regione od interesa na originalnoj slici. (ROI = regions of interest)
        Za svaki region napraviti posebnu sliku dimenzija 28 x 28. 
        Za označavanje regiona koristiti metodu cv2.boundingRect(contour).
        Kao povratnu vrednost vratiti originalnu sliku na kojoj su obeleženi regioni
        i niz slika koje predstavljaju regione sortirane po rastućoj vrednosti x ose
    '''
    img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = []
    props = []
    sizes = []
    i=0
    for contour in contours: 
        x,y,w,h = cv2.boundingRect(contour)
        region = image_bin[y:y+h+1,x:x+w+1]
        props.append((x, w*h, resize_region(region,w,h)))
        cv2.rectangle(image_orig,(x,y),(x+w,y+h),(0,255,0),2)
    
    props = sorted(props)
    
    #print [(y,x) for y,x,w,h,region in props]
    sizes = [size for x,size,region in props]
    regions = [region for x,size,region in props]
    invert_reg = []
    for r in regions :
        invert_reg.append(dilate(r))
        
    return image_orig, regions, sizes
    
def evalPostfix(exp):
    
    currentChar=0
    ##init()
    rez = None
    ''' x
    op1
    op2
    znak '''
    i=0
    tempRez=0
    while(currentChar <(exp.__len__()-1)):
        print('usp')
        x,currentChar=inputsI(exp,currentChar)

        if ((x[0]!='+') & (x[0]!='-') & (x[0]!='*') & (x[0]!='/') & (x[0]!='^')):
            inpt = ''.join(x)
            push(float(inpt));

        else :
            znak=x[0]
            if znak == '+':
                op2=pop()
                op1=pop()
                tempRez=op1+op2
                push(tempRez)
            elif znak == '-':
                op2=pop()
                op1=pop()
                tempRez =op1-op2
                push(tempRez)
            elif znak == '*':
                op2=pop()
                op1=pop()
                tempRez =op1*op2
                push(tempRez)
            elif znak == '/':
                op2=pop()
                op1=pop()
                tempRez =op1/op2
                push(tempRez)

    rez=pop()
    
    print('uspasda')
    if (isStackEmpty()):
        print rez
    else:
        print ("Ulaz nije odgovarajuci")





def display_result(outputs, alphabet, k_means):
    '''
    Funkcija određuje koja od grupa predstavlja razmak između reči, a koja između slova, i na osnovu
    toga formira string od elemenata pronađenih sa slike.
    Args:
        outputs: niz izlaza iz neuronske mreže.
        alphabet: niz karaktera koje je potrebno prepoznati
        kmeans: obučen kmeans objekat
    Return:
        Vraća formatiran string
    '''
    big_elements = {}
    small_elements = {}
    big_group = max(enumerate(k_means.cluster_centers_), key = lambda x: x[1])[0]
    print big_group
    for idx, output in enumerate(outputs[0:,:]):
        elem = alphabet[winner(output)]
        if (k_means.labels_[idx] == big_group):
            if elem not in big_elements:
                big_elements[elem] = 0
            big_elements[elem] += 1
        else:
            if elem not in small_elements:
                small_elements[elem] = 0
            small_elements[elem]  += 1
    result = ""
    for key, value in big_elements.iteritems():
        result += str(value) + " big " + str(key) + ("s\n" if (value > 1 and value != 0) else "\n")
    for key, value in small_elements.iteritems():
        result += str(value) + " small " + str(key) + ("s\n" if (value > 1 and value != 0) else "\n")
    return result
    
def display_result1(outputs, alphabet):

     for idx, output in enumerate(outputs[0:,:]):
         elem = alphabet[winner(output)]
         print elem

       
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
    
img_test = load_image('tests/6.jpg')
alphabet = alphabet = ['0','1','2','3','4','5','6','7','8','9','+','-','*','.']

img_test_bin = remove_noise(image_bin_adaptive(image_gray(img_test)))
sel_img_test, shapes, sizes = select_roi(img_test.copy(), img_test_bin)

display_image(img_test_bin)

inputs = prepareAnn(shapes)


#display_image(image_bin_adaptive(image_gray(img_test)))


model = createModel()
model.load_weights('model_weight1.hdf5')

results = model.predict(np.array(inputs, np.float32))
stringres = []

for r in results:
    stringres.append(alphabet[winner(r)])
inpt = ''.join(stringres)
print (inpt)
##print(infix2Postfix(inpt,currentChar,currentPos,topStack))

#evalPostfix(infix2Postfix(inpt,currentChar,currentPos,topStack))

