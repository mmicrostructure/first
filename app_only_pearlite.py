# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 12:35:33 2023

@author: 91916
"""
import io
from PIL import Image as im
import streamlit as st

import cv2
from PIL import Image, ImageOps
import numpy as np
from streamlit_cropper import st_cropper
from keras.models import load_model
from io import BytesIO
from matplotlib import pyplot as plt


from img_classification_only_pearlite import teachable_machine_classification

model_adress_pear='C:/Users/91916/Desktop/training/paper 3/model/trail-3_100ep_4_batch_extend_trial5.h5'


def crop_fun(image):
    cropped_img = st_cropper(image,aspect_ratio=None)
    return cropped_img


def pearlite_phase_processing(final_img):
    
    n=0
    st.write("")
    st.write("Quantifying...")
    
   # print(final_img.dtype)

    label,mask,phase = teachable_machine_classification(final_img, model_adress_pear,n)
    st.write("Ferrit phase- ",str(phase))
    st.image([label,mask] ,caption=['label image','overlayed image'],channels=' BGR',use_column_width=True)
   
    
    buffer = io.BytesIO() # make buffer memory to hold image 
    
    image_1 = im.fromarray(mask)
    image_1.save(buffer, format="PNG")# save image to buffer memory
    
        # Sample data
    ferrite = phase
    pearlite = (1-phase)
    
    # Bar plot
    fig, ax = plt.subplots()
    percentages = [ferrite, pearlite]
    labels = ['Ferrite Phase Fraction', 'Pearlite Phase Fraction']
    colors = ['black', 'red']
    
    ax.bar(labels, percentages, color=colors)
    
    # Add labels and title
    ax.set_ylabel('Phase Fraction')
    ax.set_title('Pearlite Ferrite quatification')
    
    # Display the plot in Streamlit
    st.pyplot(fig)

    btn = st.download_button(label="Download image",data=buffer,file_name="Ferrite phase fraction-"+str(phase)+".png",mime="image/png")
    

st.header("Phase quantification and micrograph cleaning using deep learning")
st.subheader("Work by N.Chaurasia, S.Sangal and S.K.Jha")
#st.text("Upload a SEM micrographs for phase quantification")


with st.sidebar:
    
   dataset_name=st.sidebar.selectbox("select the option", ("None","ferrite and pearlite phase quatification"))


tab1, tab2, tab3 = st.tabs(["phase quatification", "paper", "contact"])



with tab2:
    st.header("Paper Link")
    st.subheader("https://www.sciencedirect.com/science/article/pii/S2589152923001308")
        


    
        
    
with  tab1:
   
    if dataset_name== "ferrite and pearlite phase quatification":
        
        uploaded_file = st.sidebar.file_uploader("Choose a Ferrite and Pearlite SEM micrograph ...")
        if uploaded_file is not None:
    
            image = Image.open(uploaded_file)
            image = ImageOps.grayscale(image)
            
            st.image(image, caption='Uploaded SEM micrograph.')
            crop=st.sidebar.selectbox("CROP IMAGE", ("None","Full Image","Free Cropping"))
            
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
    

with tab3:
    st.header("Contact")
    st.subheader("Prof. Sandeep Sangal and Shikhar K Jha")
    
    col1,col2,col3= st.columns(3,gap="small")
    with col1:
        sangal=cv2.imread("C:/Users/91916/Desktop/training/paper 3/GUI/trials-2/images/sangal_1.png",0)
        st.image(sangal,caption="Prof.Sandeep Sangal")
        st.text("Email-sangals@iitk.ac.in")
    with col2:
        shikhar=cv2.imread("C:/Users/91916/Desktop/training/paper 3/GUI/trials-2/images/shikhar.png",0)
        st.image(shikhar,caption="Prof. Shikhar K. Jha ")
        st.text("Email-skjha@iitk.ac.in")
        
    with col3:
        
        
        nikhil=cv2.imread("C:/Users/91916/Desktop/training/paper 3/GUI/trials-2/images/nikhil_1.png",0)
        st.image(nikhil,caption="Ph.D Student Nikhil Chaurasia")
        st.subheader("Linkedin-www.linkedin.com/in/nikhil-chaurasia")
        st.text("Email-nikc@iitk.ac.in")
        text_paragraphs = ("I'm deeply involved in advancing AI-based characterization techniques\n\n""specifically concentrating on micrographs.\n\n"" My current focus involves employing deep learning for the quantification of pearlite lamellae,\n\n ""microstructure cleaning, and fractography quantification.")

        #st.text("I'm deeply involved in advancing AI-based characterization techniques,/n/n"" specifically concentrating on micrographs./n/n"" My current focus involves employing deep learning for the quantification of pearlite lamellae, microstructure cleaning, and fractography quantification.")
        st.text(text_paragraphs)  