import streamlit as st
import base64
import pandas as pd
import random


reviews = pd.read_csv(r'Movie_images\movie_dataset_svm.csv')
option = st.sidebar.selectbox('Select', ('Select', 'Must Watch', 'Average','Evaluation'))

if option is 'Select':
    st.write("Best Movies For You!")
    st.write("Please Select the Option")


if option is 'Evaluation':
    st.image("heatmap_svm.png", width=800, caption="Heat Map")
    st.image("wordcloud_positive_svm.png", width=800, caption="Word Cloud")


movies = reviews.loc[reviews['Status'] == option, 'Movie']
Image = reviews.loc[reviews['Status'] == option, 'image']
Image_list = Image.tolist()
list_movie = movies.tolist()

# https://discuss.streamlit.io/t/grid-of-images-with-the-same-height/10668
try:
    idx = 0
    for _ in range(len(Image_list) - 1):

        cols = st.beta_columns(3)

        if idx < len(Image_list):
            cols[0].image(Image_list[idx], width=150, caption=list_movie[idx])
        idx += 1


        if idx < len(Image_list):
            cols[1].image(Image_list[idx], width=150, caption=list_movie[idx])
        idx += 1
       

        if idx < len(Image_list):
            cols[2].image(Image_list[idx], width=150, caption=list_movie[idx])
            idx = idx +1
            
        else:
            break
except Exception as e:
        print(e)


