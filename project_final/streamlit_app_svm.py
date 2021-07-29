import streamlit as st
import base64
import pandas as pd
import random


reviews = pd.read_csv(r'Movie_images\movie_dataset_svm.csv')
option = st.sidebar.selectbox('Select', ('Select', 'Must Watch', 'Average'))

if option is 'Select':
    st.write("Best Movies For You!")
    st.write("Please Select the Option")
st.write('You selected:', option)
movies = reviews.loc[reviews['Status'] == option, 'Movie']
Image = reviews.loc[reviews['Status'] == option, 'image']
Image_list = Image.tolist()
list_movie = movies.tolist()
try:
    idx = 0
    for _ in range(len(Image_list) - 1):

        cols = st.beta_columns(3)
        print(Image_list[idx])
        if idx < len(Image_list):
            cols[0].image(Image_list[idx], width=150, caption=list_movie[idx])
        idx += 1
        print(Image_list[idx])

        if idx < len(Image_list):
            cols[1].image(Image_list[idx], width=150, caption=list_movie[idx])
        idx += 1
        print(Image_list[idx])

        if idx < len(Image_list):
            cols[2].image(Image_list[idx], width=150, caption=list_movie[idx])
            idx = idx +1
            print(Image_list[idx])
        else:
            break
except Exception as e:
        print(e)
# button_action = st.sidebar.button("Evaluation")
# if button_action:

evalution = st.sidebar.button('Evaluation')
if evalution:
    cols = st.beta_columns(2)
    cols[0].image("heatmap_svm.png", width=500, caption="Bar Chat")
    cols[1].image("bar_chat.png", width=500, caption="Heat Map")