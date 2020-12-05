# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 15:33:20 2020

@author: Bhanu
"""

import pandas as pd
import streamlit as st
from joblib import load

st.write("""
         
# Diabetes Prediction App        
 Input the details in order to know whether you are having th diabetes or not.
        
         
""")

st.sidebar.header("Input parameters")
def user_input_features():
    Pregnancies = st.sidebar.slider('Pregnancies', 0, 13, 1)
    Glucose=st.sidebar.slider('Glucose', 56.0, 198.0, 56.0)
    BloodPressure=st.sidebar.slider('BloodPressure',24.0, 110.0,24.0)
    SkinThickness=st.sidebar.slider('SkinThickness', 7.0, 63.0, 7.0)
    Insulin=st.sidebar.slider('Insulin', 14.0, 846.0, 14.0)
    BMI=st.sidebar.slider("BMI",0.0,68.0,0.0) 
    DiabetesPedigreeFunction=st.sidebar.slider("DiabetesPedigreeFunction",0.085000,2.420000,0.085000)
    Age=st.sidebar.slider("Age",13,90,13) 

   
    data = {'Pregnancies': Pregnancies,
            "Glucose":Glucose,
             "BloodPressure":BloodPressure,
             "SkinThickness":SkinThickness,
             "Insulin":Insulin,
             "BMI":BMI,
             "DiabetesPedigreeFunction":DiabetesPedigreeFunction,
             "Age":Age
             
             
            }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()
dfa=df.values
st.subheader('User Input parameters')
st.write(df)

diabetesdf=load("diabetesweights")


prediction=diabetesdf.predict_proba(dfa)
dia=prediction.tolist()
st.write(prediction)
if dia[0][1]==1:
    st.write("You have a high chance of diabetes")
    st.write(prediction)

else:
    st.write("You having diabetes are very less but this app alone can't be 100 % sure.")
