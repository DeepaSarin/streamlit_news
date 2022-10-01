import pandas as pd
import streamlit as st
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
# import warnings
# warnings.filterwarnings("ignore")
# from PIL import Image

pickle_in = open("news_sentiment.pkl","rb")
classifier = pickle.load(pickle_in)
pickle_vect = open("tfidf_vect.pkl","rb")
tf_vect = pickle.load(pickle_vect)

def predict_news_sentiment(News,classifier):
    #Tfidf_vect = TfidfVectorizer(max_features=5000)
    #Tfidf_vect.fit([News])
    
    News_tr=tf_vect.transform(News)
    prediction=((classifier.predict(News_tr))[0])
    print(prediction)
    return prediction  
    
def Input_Output():
    st.title("Customer Sentiment Analysis")
    st.image("https://www.freecodecamp.org/news/content/images/size/w2000/2020/09/wall-5.jpeg", width=200)
    
    
    st.markdown("A value for Customer Sentiment will be predicted using Random Forest Regressor (-1 being highly negative, +1 being highly positive and 0 a neutral News ",unsafe_allow_html=True)
    st.markdown("Use Cases for Sentiment Analysis include :1.Customer Satisfaction Analysis 2.Analyse Customer Service Issues 3.Plan Product Improvements to name a few ",unsafe_allow_html=True)
                
    news  = [st.text_input("Enter Customer Review to be analyzed for Sentiment " , " ")]
           
    result = 0.0
    if st.button("Click here to Predict"):
        result =float( predict_news_sentiment(news,classifier))
        st.balloons()
    
    
    st.success('The Sentiment Score Predicted for this Review Article is ',result)
    if result >0 :
        st.markdown("This is a Positive Review")
    elif result == 0:
        st.markdown("This is a neutral Review")
    else:
        st.markdown("This is a Negative Review")
if __name__ ==  '__main__':
    Input_Output()
