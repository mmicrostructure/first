# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 12:38:10 2023

@author: 91916
"""

import keras
from PIL import Image, ImageOps
import numpy as np
import cv2
import os
from skimage.util import img_as_ubyte
from skimage.color import label2rgb
from numba import njit
from matplotlib import pyplot as plt
from keras.models import load_model





def pearlite_phase(a,b,c,d):        #a=Actual Image,b=Actual image RGB
    
    
    
    a2=cv2.resize(a,(512,512))
    test_img1 = np.resize(a2,(512,512,1))
    test_img = test_img1/255 
    test_img = np.expand_dims(test_img, axis=0)
    prediction =(c.predict(test_img))
    
    y_pred_argmax=np.argmax(prediction, axis=3)
    a1=y_pred_argmax.reshape(-1, y_pred_argmax.shape[-1])
    overlay=np.zeros((512,512,3),dtype=np.uint8)
    
    
    
    
    for i in range (0,512,1):
        for j in range (0,512,1):
            if(a1[i][j]!=9):
                overlay[i][j][:]=255,0,0

            else:
                overlay[i][j][:]=0,0,0
                
    
    resize_original=cv2.resize(overlay,d)
   
    phase=round((np.count_nonzero(a1==9)/(512*512)),2)
                    
    dst = cv2.addWeighted(b,1, overlay,0.3, 0)
    
    
    dst=cv2.resize(dst,(d))
    overlay=cv2.resize(overlay,(d))
    
    return overlay,dst,phase


    
def teachable_machine_classification(img, weights_file,m):
    
       
    

# Load the model
    model = load_model(weights_file,compile=False)
    image=img
    size=img.size
    image_array = np.asarray(image)
    image=img_as_ubyte(image_array)
    bcktorgb=cv2.resize(image,(512,512))
    bcktorgb=cv2.cvtColor(bcktorgb, cv2.COLOR_GRAY2RGB)
    if(m==0):
       per,mask,phase= pearlite_phase(image,bcktorgb,model,size)
       return per,mask,phase
       
        
        
        
   
 
