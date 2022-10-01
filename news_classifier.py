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
    st.image("https://thumbs.dreamstime.com/b/news-newspapers-folded-stacked-word-wooden-block-puzzle-dice-concept-newspaper-media-press-release-42301371.jpg", width=200)
    
    st.markdown("You are using Streamlit...",unsafe_allow_html=True)
    st.markdown("Coal,Petroleum,Agriculture,Construction,Steel and Iron and Maritime are the categories inclueded in training the model",unsafe_allow_html=True)
                
    news  = [st.text_input("Enter News to be classified " , " ")]
    
    
    
    result = ""
    if st.button("Click here to Predict"):
        result = predict_news_category(news,classifier)
        st.balloons()
    
    if result==1 :
       result='agriculture' 
    elif result==2 :
       result='coal'
    elif result==3 :
       result='construction'
    elif result==4:
        result='fertilizer'
    elif result==5 :
       result='iron ore and steel'        
    elif result==6 :
       result='maritime'
    else :
       result='petroleum'
    result=result+1
    st.success('The output is {}'.format(result))
   
if __name__ ==  '__main__':
    Input_Output()
