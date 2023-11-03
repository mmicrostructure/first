
import io
from PIL import Image as im
import streamlit as st
from img_classification import teachable_machine_classification
import cv2
from PIL import Image, ImageOps
import numpy as np
from streamlit_cropper import st_cropper
from keras.models import load_model
from io import BytesIO



model_adress_pear='trail-3_100ep_4_batch_extend_trial5.h5'
#model_adress_pear='C:/Users/91916/Desktop/training/paper 3/model/trail-2_50ep_4batch_10class.h5'
model_adress_bainit='C:/Users/91916/Desktop/training/ashwani work/Ashwani-0060.HDF5'

model_clean="C:/Users/91916/Desktop/training/single phase/threshold_trail1_50epoch.HDF5"


def crop_fun(image):
    cropped_img = st_cropper(image,aspect_ratio=None)
    return cropped_img


def pearlite_phase_processing(final_img):
    
    n=0
    st.write("")
    st.write("Quantifying...")
    
   # print(final_img.dtype)

    label,mask,phase = teachable_machine_classification(final_img, model_adress_pear,n)
    st.write("Ferrit phae- ",str(phase))
    st.image([label,mask] ,caption=['label image','overlayed image'],channels=' BGR',use_column_width=True)
   
    
    buffer = io.BytesIO() # make buffer memory to hold image 
    
    image_1 = im.fromarray(mask)
    image_1.save(buffer, format="PNG")# save image to buffer memory

    btn = st.download_button(label="Download image",data=buffer,file_name="pearlite phase fra-"+str(phase)+".png",mime="image/png")
    

def pearlite_orientation_processing(final_img):
    
    n=1
      
    st.write("")
    st.write("Quantifying...")
    mask,label,angle,figure = teachable_machine_classification(final_img, model_adress_pear,n)
    st.image([label,scale,mask] ,caption=['original image',"angle scale",'overlayed image'],channels=' BGR',use_column_width=True)
    
    st.pyplot(figure)
    
    

    buffer = io.BytesIO() # make buffer memory to hold image 
    image_1 = im.fromarray(mask)
    image_1.save(buffer, format="PNG")# save image to buffer memory

    btn = st.download_button(label="Download image",data=buffer,file_name="orientation map.png",mime="image/png")
        
  


def bainite(final_img):
    

    
    
    n=2
    st.write("")
    st.write("Quantifying...")

    label,mask,phase = teachable_machine_classification(final_img,model_adress_bainit,n)
    st.write("Bainite phase- ",str(phase))
    st.image([label,mask] ,caption=['label image','overlayed image'],channels=' BGR',use_column_width=True)
    
    buffer = io.BytesIO() # make buffer memory to hold image 

    image_1 = im.fromarray(mask)
    image_1.save(buffer, format="PNG")# save image to buffer memory

    btn = st.download_button(label="Download image",data=buffer,file_name="bainite fra-"+str(phase)+".png",mime="image/png")
        
  
def clean(final_img):
    n=3
    st.write("")
    st.write("Cleaning.. 🧹")
    
    micro,binary = teachable_machine_classification(final_img,model_clean,n)
    #st.write("Bainite phase- ",str(phase))
    st.image([micro,binary] ,caption=['Cleaned microstructure','binary microstrucutre'],channels=' BGR',use_column_width=True)
    
    buffer = io.BytesIO() # make buffer memory to hold image 

    image_1 = im.fromarray(binary)
    image_1.save(buffer, format="PNG")# save image to buffer memory

    btn = st.download_button(label="Download image",data=buffer,file_name="clean_micro.png",mime="image/png")
        
     
     

st.header("Phase quantification and micrograph cleaning using deep learning")
st.subheader("Work by N.Chaurasia, S.Sangal and S.K.Jha, IIT Kanpur")
#st.text("Upload a SEM micrographs for phase quantification")

dataset_name=st.sidebar.selectbox("select the option", ("None","Pearlite lamella orientation map","ferrite and pearlite phase quatification","Pearlite inter lamella spacing","Bainite quantification","Single phase microstructure cleaning"))
if dataset_name== "Pearlite lamella orientation map":
    
    scale=cv2.imread("scale.png")
    scale = cv2.cvtColor(scale, cv2.COLOR_BGR2RGB)
    
    uploaded_file = st.sidebar.file_uploader("Choose a Ferrite and Pearlite SEM micrograph ...")
    if uploaded_file is not None:

        image = Image.open(uploaded_file)
        image = ImageOps.grayscale(image)
        
        st.image(image, caption='Uploaded SEM micrograph.')
        crop=st.sidebar.selectbox("crop image", ("None","Full Image","Free Cropping"))
        
        if crop =='Free Cropping':
            cropped_img=crop_fun(image) 
            print("crop befor:",cropped_img.size)
            result=st.button("Click here to run")
            
            if result:
                
                #_ = cropped_img.thumbnail((512,512))
                st.image(cropped_img)
                #st.image(cropped_img ,caption="cropped image")        
                pearlite_orientation_processing(cropped_img)
        elif crop =='Full Image':
            result1=st.button("Click here to run")
            if result1:
                pearlite_orientation_processing(image)
                
elif dataset_name== "ferrite and pearlite phase quatification":
    
    uploaded_file = st.sidebar.file_uploader("Choose a Ferrite and Pearlite SEM micrograph ...")
    if uploaded_file is not None:

        image = Image.open(uploaded_file)
        image = ImageOps.grayscale(image)
        
        st.image(image, caption='Uploaded SEM micrograph.')
        crop=st.sidebar.selectbox("crop image", ("None","Full Image","Free Cropping"))
        
        if crop =='Free Cropping':
            cropped_img=crop_fun(image)
            print("croppinf size",cropped_img.size)
            result=st.button("Click here to run")
            
            if result:
                
               # _ = cropped_img.thumbnail((512,512))
                st.image(cropped_img)
                #st.image(cropped_img ,caption="cropped image") 
                pearlite_phase_processing(cropped_img)
        elif crop =='Full Image':
            result1=st.button("Click here to run")
            if result1:
                pearlite_phase_processing(image)
elif dataset_name== "Bainite quantification":
    uploaded_file = st.sidebar.file_uploader("Choose a Bainitic SEM micrograph ...")
    if uploaded_file is not None:

        image = Image.open(uploaded_file)
        image = ImageOps.grayscale(image)
        
        st.image(image, caption='Uploaded SEM micrograph.')
        crop=st.sidebar.selectbox("crop image", ("None","Full Image","Free Cropping"))
        
        if crop =='Free Cropping':
            cropped_img=crop_fun(image) 
            result=st.button("Click here to run")
            
            if result:
                
                _ = cropped_img.thumbnail((512,512))
                st.image(cropped_img)
                #st.image(cropped_img ,caption="cropped image")   
                bainite(cropped_img)
        elif crop =='Full Image':
            result1=st.button("Click here to run")
            if result1:
                bainite(image)
elif dataset_name== "Single phase microstructure cleaning":
    
    uploaded_file = st.sidebar.file_uploader("Choose a single phase optical micrograph ...")
    if uploaded_file is not None:

        image = Image.open(uploaded_file)
        image = ImageOps.grayscale(image)
        
        st.image(image, caption='Uploaded micrograph.')
        crop=st.sidebar.selectbox("crop image", ("None","Full Image","Crop scalebar"))
        
        if crop =='Crop scalebar':
            cropped_img=crop_fun(image) 
            result=st.button("Click here to run")
            
            if result:
                
                _ = cropped_img.thumbnail((512,512))
                st.image(cropped_img)
                #st.image(cropped_img ,caption="cropped image")   
                clean(cropped_img)
        elif crop =='Full Image':
            result1=st.button("Click here to run")
            if result1:
                clean(image)

elif dataset_name== "Pearlite inter lamella spacing":
    
    scale=cv2.imread("scale.png")
    scale = cv2.cvtColor(scale, cv2.COLOR_BGR2RGB)
    
    uploaded_file = st.sidebar.file_uploader("Choose a Ferrite and Pearlite SEM micrograph ...")
    if uploaded_file is not None:

        image = Image.open(uploaded_file)
        image = ImageOps.grayscale(image)
        
        st.image(image, caption='Uploaded SEM micrograph.')
        crop=st.sidebar.selectbox("crop image", ("None","Full Image","Free Cropping"))
        
        if crop =='Free Cropping':
            cropped_img=crop_fun(image) 
            print("crop befor:",cropped_img.size)
            result=st.button("Click here to run")
            
            if result:
                
                #_ = cropped_img.thumbnail((512,512))
                st.image(cropped_img)
                #st.image(cropped_img ,caption="cropped image")        
                pearlite_orientation_processing(cropped_img)
        elif crop =='Full Image':
            result1=st.button("Click here to run")
            if result1:
                pearlite_orientation_processing(image)
