import streamlit as st
import pickle
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

modela = tf.keras.models.load_model("C:/Users/satya/Downloads/mdl_wt_ra.hdf5")
modelg = tf.keras.models.load_model("C:/Users/satya/Downloads/mdl_wt_rg.hdf5")
import tensorflow
import pandas as pd
st.title("Covid Severity Prediction System")

#modelresage=pickle.load(open("C:\\Users\\satya\\Downloads\\age.pkl",'rb'))

#gen_model=pickle.load(open('./gender.pkl','rb'))

def get_severity(gender_classes,age_classes):
  s=[]
  for i in range(0, len(age_classes)):
    if age_classes[i]==0 and gender_classes[i]==0:     #female 0-18
      severity=0.001
      s.append(severity)
    elif age_classes[i]==0 and gender_classes[i]==1:    #male 0-18
      severity=0.0015
      s.append(severity)
    elif age_classes[i]==1 and gender_classes[i]==0:
      severity=0.0048
      s.append(severity)
    elif age_classes[i]==1 and gender_classes[i]==1:
      severity=0.0085
      s.append(severity)
    elif age_classes[i]==2 and gender_classes[i]==0:
      severity=0.45
      s.append(severity)
    elif age_classes[i]==2 and gender_classes[i]==1:
      severity=0.59
      s.append(severity)
    elif age_classes[i]==3 and gender_classes[i]==0:
      severity=0.53
      s.append(severity)
    else:
      severity=0.39
      s.append(severity)
  return s

uploaded_file = st.file_uploader("Upload an image of a Human and click on 'Predict Age, Gender and Severity for COVID' button")

if uploaded_file is not None:
     # To read file as bytes:
     bytes_data = uploaded_file.getvalue()
     st.image(bytes_data)

def predict(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype = np.uint8)
    img= cv2.imdecode(file_bytes,1)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img =  cv2.resize(img,(48,48)) # (200,200)--->(48,48)
    x_data = np.array(img)
    gender=modelg.predict(np.array([ img ]))
    age=modela.predict(np.array([ img ]))
    gender_classes = gender.argmax(axis=-1)
    age_classes = age.argmax(axis=-1)
    return gender_classes,age_classes

if st.button('Predict Age, Gender and Severity for COVID'):
    gender_classes,age_classes=predict(uploaded_file)
    sev=get_severity(gender_classes,age_classes)
    st.write("Gender")
    if gender_classes == 0:
        st.write('Female')
    if gender_classes == 1:
        st.write('Male')
    st.write("Age")
    if age_classes ==0:
        st.write('0-18')
    if age_classes ==1:
        st.write('19-30')
    if age_classes ==2:
        st.write('31-80')
    if age_classes ==3:
        st.write('>80')
    st.write(sev[0])



