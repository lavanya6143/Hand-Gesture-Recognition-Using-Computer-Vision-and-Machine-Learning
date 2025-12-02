import cv2
import numpy as np
import pandas as pd
import math
import sys
import os
import tensorflow as tf 
from keras.models import load_model
import pyttsx3 
from wordsegment import load,segment
import enchant
import autocomplete
num=0
engine = pyttsx3.init() 
cap = cv2.VideoCapture(0)
img_width = 1024
img_height = 768
cap.set(cv2.CAP_PROP_FRAME_WIDTH, img_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, img_height)
ll=[]
def image_resize(image, height = 45, inter = cv2.INTER_AREA):
    resized = cv2.resize(image, (height,height), interpolation = inter)
    return resized

model = load_model('C:/Users/lavan/Downloads/B24 Project/Project/Hand Gestures/trained.h5')

# Check if the model has been loaded correctly
model.summary()

encoding_chart = pd.read_csv('C:/Users/lavan/Downloads/B24 Project/Project/Hand Gestures/label_encoded.csv')
encoding_values = encoding_chart['Encoded'].values
encoding_labels = encoding_chart['Label'].values
int_to_label = dict(zip(encoding_values,encoding_labels))

font = cv2.FONT_HERSHEY_DUPLEX

history = list()
counts = dict()
history_length = 15
threshold = 0.9

start = 200
end = 500
alpha = 0.4

sentence_raw = list()


color = (59, 185, 246)

load()
sente=""
disp=""
sent=""
while(True):
    ret, img = cap.read()
    img = cv2.flip(img,1)
    alpha_layer = img.copy()
    source = img.copy()

    crop_img = source[start:end, start:end]
    cv2.circle(alpha_layer, (int((start+end)/2),int((start+end)/2)), int((end - start)/2), color ,-1)
    cv2.addWeighted(alpha_layer, alpha, img, 1 - alpha,0, img)

    grey = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    resized = image_resize(crop_img)
    predicted = model.predict(np.array([resized]))

    predicted_char = int_to_label[np.argmax(predicted)]
    cv2.putText(img,"Legend:",(50,160),font,1,(255,255,255),1)
    cv2.putText(img,"s-Speech",(50,190),font,1,(255,255,255),1)
    cv2.putText(img,"a-append",(50,220),font,1,(255,255,255),1)
    cv2.putText(img,"p-space",(50,250),font,1,(255,255,255),1)
    cv2.putText(img,"x-Delete",(50,280),font,1,(255,255,255),1)
    cv2.putText(img,"X-Clear",(50,310),font,1,(255,255,255),1)
    
    
    if(len(history)>=history_length):
        keys = list(counts.keys())
        values = list(counts.values())
        arg = np.argmax(values)
        if(values[arg]>threshold*history_length):
            sentence_raw.append(keys[arg])
        counts.clear()
        history.clear()
    if(predicted_char != 'None'):
        history.append(predicted_char)
        if(predicted_char in counts):
            counts[predicted_char]+=1
        else:
            counts[predicted_char]=1
    
        textsize = cv2.getTextSize(str(predicted_char), font, 0.6, 1)[0]
        textX = int(start + ((end - start) - textsize[0])/2)
        textY = int(end - ((end - start) - textsize[1])/2)
    
        cv2.putText(img, str(predicted_char), (textX,textY),font,6,color,7)
    scribble = "".join(str(c) for c in sentence_raw)
    sentence = " ".join(segment(scribble))    

    k=cv2.waitKey(1)
    
    if k==ord('a'):
        sente+=predicted_char
    elif k==ord('p'):
        sente+=" "
        
    cv2.putText(img,sente,(80,50),font,1,(255,255,255),2)
    
    
    #k = cv2.waitKey(10)
    if k == ord('x'):
        sente=sente[:len(sente)-1]
    elif k==ord('X'):
        sente=""
    elif k==ord('s'):
        engine.say(sente) 

        engine.runAndWait()
   
    if k == 27:
        break
    
    d=enchant.Dict("en_US")
    try:
        disp=""
        sent1=sente.split()
        
        
        a=d.suggest(str(sent1[-1]))
        
       
        for cc,tt in enumerate(a):
            if cc==10:
                break
            disp+=str(cc)+str(".")+str(tt)+" "
            
     
            
    except:
        kkk=0
    cv2.putText(img,disp,(0,120),font,1,(255,255,255),1)
    cv2.putText(img,"Suggestions:",(0,90),font,1,(255,255,255),1)
    cv2.putText(img,"Text:",(0,50),font,1,(255,255,255),1)
    
    
    if k==48:
       sent=sente.split()
       if len(sent)>1:
           lastword=a[0]
        
           sente=""
           for gg in range(len(sent)-1):
               sente+=str(sent[gg])+" "
           sente+=str(a[0])
       else:
           sente=str(a[0])
    elif k==49:
        sent=sente.split()
        if len(sent)>1:
           lastword=a[1]
        
           sente=""
           for gg in range(len(sent)-1):
               sente+=str(sent[gg])+" "
           sente+=str(a[1])
        else:
           sente=str(a[1])
    elif k==50:
        sent=sente.split()
        if len(sent)>1:
           lastword=a[2]
        
           sente=""
           for gg in range(len(sent)-1):
               sente+=str(sent[gg])+" "
           sente+=str(a[2])
        else:
           sente=str(a[2])
    elif k==51:
        sent=sente.split()
        if len(sent)>1:
           lastword=a[0]
        
           sente=""
           for gg in range(len(sent)-1):
               sente+=str(sent[gg])+" "
           sente+=str(a[3])
        else:
           sente=str(a[3])
    elif k==52:
        sent=sente.split()
        if len(sent)>1:
           lastword=a[0]
        
           sente=""
           for gg in range(len(sent)-1):
               sente+=str(sent[gg])+" "
           sente+=str(a[4])
        else:
           sente=str(a[4])
    elif k==53:
        sent=sente.split()
        if len(sent)>1:
           lastword=a[0]
        
           sente=""
           for gg in range(len(sent)-1):
               sente+=str(sent[gg])+" "
           sente+=str(a[5])
        else:
           sente=str(a[5])
    elif k==54:
        sent=sente.split()
        if len(sent)>1:
           lastword=a[0]
        
           sente=""
           for gg in range(len(sent)-1):
               sente+=str(sent[gg])+" "
           sente+=str(a[6])
        else:
           sente=str(a[6])
    elif k==55:
        sent=sente.split()
        if len(sent)>1:
           lastword=a[0]
        
           sente=""
           for gg in range(len(sent)-1):
               sente+=str(sent[gg])+" "
           sente+=str(a[7])
        else:
           sente=str(a[7])
    elif k==56:
        sent=sente.split()
        if len(sent)>1:
           lastword=a[0]
        
           sente=""
           for gg in range(len(sent)-1):
               sente+=str(sent[gg])+" "
           sente+=str(a[8])
        else:
           sente=str(a[8])
    cv2.imshow('WebCam', img)
    #cv2.putText(img,disp,(70,70),font,1,(255,255,255),2)
    #print(a[-1])
    #cv2.putText(img,str(a[0]),(150,150),font,1,(255,255,255),2)
