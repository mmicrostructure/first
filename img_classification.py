# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 01:17:13 2023

@author: 91916
"""



from PIL import Image, ImageOps
import numpy as np
import cv2
import os
from skimage.util import img_as_ubyte
from skimage.color import label2rgb
from numba import njit
from matplotlib import pyplot as plt
import keras
from keras.models import load_model





def pearlite_phase(a,b,c,d):        #a=Actual Image,b=Actual image RGB
    
    
    
    print("orignail shape",a.shape)
    a2=cv2.resize(a,(512,512))
    test_img1 = np.resize(a2,(512,512,1))
#    plt.imshow(test_img1,cmap="gray")
#    plt.show()
 
    #print(test_img1.dtype)
    test_img = test_img1/255 
    test_img = np.expand_dims(test_img, axis=0)
    
    prediction =(c.predict(test_img))
    #print(" 3 ",prediction.shape)
    
    y_pred_argmax=np.argmax(prediction, axis=3)
    a1=y_pred_argmax.reshape(-1, y_pred_argmax.shape[-1])

    overlay=np.zeros((512,512),dtype=np.uint8)
    
    
    
    
    for i in range (0,512,1):
        for j in range (0,512,1):
            if(a1[i][j]!=9):
                overlay[i][j]=255

            else:
                overlay[i][j]=0
                
    
    resize_original=cv2.resize(overlay,d)
    print("size of overlay", resize_original.shape)
   
    phase=round((np.count_nonzero(a1==9)/(512*512)),2)
    

    
    #resize_original=cv2.resize(a,d)
                     
    dst = cv2.addWeighted(a2,0.4, overlay,0.6, 0)
    
    dst=cv2.resize(dst,(d))
    overlay=cv2.resize(overlay,(d))
    
    return overlay,dst,phase



def polar_plot(theta,r) :   
    # Data
    #theta = [10, 30, 50, 70, 90, 110, 130, 150, 170]
    #r = [0.54, 0.50, 0.58, 0.06, 0.06, 0.08, 0.2, 0.4, 0.4]
    
    # Convert degrees to radians
    theta = np.deg2rad(theta)
    colors = [(222,222, 222),
              (255,0, 0), 
              (0,255,0), 
              (0, 0,255),  
              (128, 0,128),  
              (150, 75, 0),  
              (255,255, 0),   
              (255, 165, 0),  
              (0, 255, 255)]   
    
    
    
    
    # Create a figure and polar axes
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(5, 5))
    for t, value, color in zip(theta, r, colors):
        color = [c / 255.0 for c in color]  # Normalize RGB values to the [0, 1] range
        ax.bar(t, value, width=0.35, bottom=0.0, color=color, alpha=0.7)
    
    # Customize the polar plot
    ax.set_rlabel_position(90)
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.set_xticks(np.deg2rad([0, 20, 40, 60, 80, 100, 120, 140, 160]))
    ax.set_xticklabels(['0°', '20°', '40°', '60°', '80°','100°', '120°', '140°', '160°'])
    ax.set_xlabel("Orientation fraction", labelpad=12)
    ax.set_ylabel("Angular Orientation", labelpad=15)
    ax.tick_params(axis='y', labelsize=6)
    plt.title("Pearlite angular lamella orientation Plot")
    return fig

def orientation_map(a,b,c,d):
    
    a2=cv2.resize(a,(512,512))
    test_img1 = np.resize(a2,(512,512,1))
#   plt.imshow(test_img1,cmap="gray")
#   plt.show()
 
    #print(test_img1.dtype)
    test_img = test_img1/255 
    test_img = np.expand_dims(test_img, axis=0)
    
    prediction =(c.predict(test_img))
    #print(" 3 ",prediction.shape)
    
    y_pred_argmax=np.argmax( prediction, axis=3)
    a=y_pred_argmax.reshape(-1, y_pred_argmax.shape[-1])

    overlay=np.zeros((512,512,3),dtype=np.uint8)
    
       
    
    
    for i in range (0,512,1):
        for j in range (0,512,1):
            if(a[i][j]==0):
                overlay[i][j][:]=255,255,255
            if(a[i][j]==1):
                overlay[i][j][:]=255,0,0
            if(a[i][j]==2):
                overlay[i][j][:]=0,255,0
            if(a[i][j]==3):
                 overlay[i][j][:]=0,0,255
            if(a[i][j]==4):
                overlay[i][j][:]=128,0,128
            if(a[i][j]==5):
                overlay[i][j][:]=150,75,0
            if(a[i][j]==6):
                overlay[i][j][:]=255,255,0
            if(a[i][j]==7):
                overlay[i][j][:]=255,165,0
            if(a[i][j]==8):
                overlay[i][j][:]=0,255,255
            if(a[i][j]==9):
                overlay[i][j][:]=0,0,0
                
                
    
                
    
        
    unique_elements, counts_elements = np.unique(a, return_counts=True)
    print(unique_elements,"Heelo")
    print((counts_elements))
    
    #counts_elements=np.delete(counts_elements,[9])
    counts_elements=np.delete(counts_elements,np.where(unique_elements==9))
    unique_elements=np.delete(unique_elements,np.where(unique_elements==9))
    print((counts_elements))
    print((unique_elements))
    #np.delete(unique_elements,np.where(unique_elements==9))
    print('possible outcome', counts_elements)
    
    ang=(unique_elements*20)+10
    ang_fra=counts_elements/(sum(counts_elements))
    print(ang)
    print(ang_fra)
    
    
    angle=unique_elements[counts_elements.argmax()]
    
    angle=angle*20
    print("prefferend lamella",unique_elements[counts_elements.argmax()],angle)
    
    figure = polar_plot(ang,ang_fra)
    
    
    
         
    dst = cv2.addWeighted(b,1, overlay,0.5, 0)
    
    dst=cv2.resize(dst,(d))
    overlay=cv2.resize(overlay,(d))
    
    
    return overlay,dst,angle,figure
    
def bainite(a,b,c):
    
    
    test_img1 = np.resize(a,(512,512,1))
#    plt.imshow(test_img1,cmap="gray")
#    plt.show()
 
    #print(test_img1.dtype)
    test_img = test_img1/255 
    test_img = np.expand_dims(test_img, axis=0)
    
    prediction =(c.predict(test_img))
    #print(" 3 ",prediction.shape)
    a=np.squeeze(prediction, 0)
    a=np.squeeze(a, 2)
    a[a>0.5]=1
    a[a<=0.5]=0
    
    #print(a.shape,b.shape)
    dst = label2rgb(a, image=b, bg_label=0,alpha=0.3)
    dst = img_as_ubyte(dst)
    #dst = cv2.addWeighted(b,1, a,0.6, 0)
    #print(a.shape,b.shape)
    phase=round((np.count_nonzero(a))/(512*512),2)
    #print(phase)
    return a,dst,phase    




def preprocess(test_img1,size,img):
    
    bin_avg=cv2.resize(test_img1,(size[1],size[0]))
    res,bin_avg1=cv2.threshold(bin_avg, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    img_avg2=img.flatten()
    bin_avg2=bin_avg1.flatten()
    
    index=np.where(bin_avg2==0)[0]
    img_avg3 = np.delete(img_avg2, index)
    avg=img_avg3.mean()
    return avg
    
    



def prediction(test_img1,c):
    
     
    test_img2 = np.resize(test_img1,(512,512,1))
    #plt.imshow(test_img1,cmap="gray")
    #plt.show()
    #print(test_img1.dtype)
    test_img2 = test_img2/255 
    test_img2 = np.expand_dims(test_img2, axis=0)
        
    prediction_50=(c.predict(test_img2))
     
    prediction_50=np.squeeze(prediction_50, 0)
    prediction_50=np.squeeze(prediction_50, 2)
    prediction_50[prediction_50>0.5]=1
    prediction_50[prediction_50<=0.5]=0
    
    prediction_50=img_as_ubyte(prediction_50)
  
    return prediction_50



def postprocess(pred,avg,binary_img,final_img,size):
    
    final=np.copy(final_img)
    noise=cv2.subtract(pred,binary_img)
    #print("3",size[1])
    noise=cv2.resize(noise,(size[0],size[1]))
    #print("4",noise.shape)
    res1,noise=cv2.threshold(noise, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #print("5",noise.shape,final.shape)
    final[noise==255]=avg
    #print("6")
    return final 
    



def cleaninig(a,b,c,size):
    
    a= np.asarray(a)
    a=img_as_ubyte(a)
    test_img=cv2.resize(a,(512,512))
    #print("1")
    test_img=(np.rint((test_img-test_img.min())*(255/(test_img.max()-test_img.min())))).astype(np.uint8)
    #print("2")
    res,test_img1=cv2.threshold(test_img, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #print("3")
    avg=preprocess(test_img1,size,a)
    #print("4")
    pred=prediction(test_img1,c)
    #print("5")
    result=postprocess(pred,avg,test_img1,a,size)
    return result,pred


def teachable_machine_classification(img, weights_file,m):
    
       
    

# Load the model
    model = load_model(weights_file,compile=False)
    image=img
    #print("Intial Size=",image.size)
    size=img.size
    #test_img1= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #image = ImageOps.fit(image, (512,512), Image.ANTIALIAS)
    #print("after Size=",image.size)
    #print("Intial Size=",img.size)
    
    #test_img1=cv2.resize(img,(512,512))

    #k=os.path.basename(files)
    image_array = np.asarray(image)
    image=img_as_ubyte(image_array)
    #print("before",image.shape)
    bcktorgb=cv2.resize(image,(512,512))
    bcktorgb=cv2.cvtColor(bcktorgb, cv2.COLOR_GRAY2RGB)
    #bcktorgb=cv2.resize(bcktorgb,(512,512))
    
    #print("RGB",bcktorgb.shape)
    
    
   # plt.imshow(test_img1,cmap="gray")
   # plt.show()
    #test_img1 = np.resize(image,(512,512,1))
#    plt.imshow(test_img1,cmap="gray")
#    plt.show()
 
    #print(test_img1.dtype)
    #test_img = test_img1/255 
    #test_img = np.expand_dims(test_img, axis=0)
    
    #prediction =(model.predict(test_img))
    
        
    
    
    if(m==1):
        
        
       orientation,mask,angle,figure=orientation_map(image,bcktorgb,model,size)
       return mask, orientation,angle,figure
        
        
    elif(m==0):
       per,mask,phase= pearlite_phase(image,bcktorgb,model,size)
       return per,mask,phase
   
    elif(m==2):
        bai,mask,phase= bainite(image,bcktorgb,model)
        return bai,mask,phase
    elif(m==3):
    #    print("final szie",img.size)
        micro,binary= cleaninig(img,bcktorgb,model,size)
        return micro,binary
        
        
        
        
   
 
