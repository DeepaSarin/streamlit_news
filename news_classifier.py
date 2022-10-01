# -*- coding: utf-8 -*-
"""news_classifier.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1SWvGp7P-ayXejeYaa_d1ZMhpvL7Kd45F
"""
import pandas as pd
import streamlit as st
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
# import warnings
# warnings.filterwarnings("ignore")
# from PIL import Image

pickle_in = open("news_classifier.pkl","rb")
classifier = pickle.load(pickle_in)
pickle_vect = open("tfidf_vect.pkl","rb")
tf_vect = pickle.load(pickle_vect)

def predict_news_category(News,classifier):
    #Tfidf_vect = TfidfVectorizer(max_features=5000)
    #Tfidf_vect.fit([News])
    
    News_tr=tf_vect.transform(News)
    prediction=((classifier.predict(News_tr))[0])
    print(prediction)
    return prediction  
    
def Input_Output():
    st.title("News Category Prediction")
    st.image("https://www.google.com/imgres?imgurl=https%3A%2F%2Fi.pinimg.com%2F736x%2F80%2F0d%2F23%2F800d23c71a4fa03a3280402193978a8d--latest-jokes-website-software.jpg&imgrefurl=https%3A%2F%2Fwww.pinterest.com%2Ftechcruiser%2F&tbnid=tIL3MKDaEsrwoM&vet=12ahUKEwiWg5X9kL36AhUOktgFHWaMA74QMyghegUIARCvAg..i&docid=vDjLB-YVZRL1OM&w=351&h=199&q=online%20news%20images&ved=2ahUKEwiWg5X9kL36AhUOktgFHWaMA74QMyghegUIARCvAg", width=600)
    
    st.markdown("You are using Streamlit...",unsafe_allow_html=True)
    st.markdown("Coal,Petroleum,Agriculture...",unsafe_allow_html=True)
                
    news  = [st.text_input("Enter News to be classified " , " ")]
    
    
    
    result = ""
    if st.button("Click here to Predict"):
        result = predict_news_category(news,classifier)
        st.balloons()     
    st.success('The output is {}'.format(result))
   
if __name__ ==  '__main__':
    Input_Output()
