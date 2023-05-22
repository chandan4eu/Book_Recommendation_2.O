'''
Author: Chandan kumar
Email: chandan4eu@gmail.com
Date: 17 may 2023
'''

import pickle
import streamlit as st
import numpy as np
import re
import pickle
import operator
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from scipy.sparse import csr_matrix



st.header('Book Recommendation System 2.O')
model = pickle.load(open('Pickles/modelbook.pkl', 'rb'))
book_names = pickle.load(open('Pickles/book_names.pkl', 'rb'))
pickel_off1= open(r"Pickles/final_rating.pkl","rb")
final_rating = pd.read_pickle(pickel_off1)

pickle_off = open(r"Pickles/book_pivot.pkl","rb")
book_pivot = pd.read_pickle(pickle_off)


def fetch_poster(suggestion):
    book_name = []
    ids_index = []
    poster_url = []

    for book_id in suggestion:
        book_name.append(book_pivot.index[book_id])

    for name in book_name[0]:
        ids = np.where(final_rating['title'] == name)[0][0]
        ids_index.append(ids)

    for idx in ids_index:
        url = final_rating.iloc[idx]['image_url']
        poster_url.append(url)

    return poster_url


def recommend_book(book_name):
    books_list = []
    book_id_temp = np.where(book_pivot.index == book_name)
    book_id=[]
    distance=[]
    suggestion=[]
    if(len(book_id_temp[0])>0):
        book_id= np.where(book_pivot.index == book_name)[0][0]
        distance, suggestion = model.kneighbors(book_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=6)
    else:
        exit()
    #distance, suggestion = model.kneighbors(book_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=6)

    poster_url = fetch_poster(suggestion)

    for i in range(len(suggestion)):
        books = book_pivot.index[suggestion[i]]
        for j in books:
            books_list.append(j)
    return books_list, poster_url

selected_books = st.selectbox(
    "Type or select a book from the dropdown",
    book_names
)

if st.button('Show Recommendation'):
    recommended_books, poster_url = recommend_book(selected_books)
    col1, col2, col3, col4, col5 = st.columns(5)
    if(len(recommended_books) >1):
        with col1:
            st.text(recommended_books[1])
            st.image(poster_url[1])
    if (len(recommended_books) > 2):
        with col2:
            st.text(recommended_books[2])
            st.image(poster_url[2])
    if (len(recommended_books) > 3):
        with col3:
            st.text(recommended_books[3])
            st.image(poster_url[3])
    if (len(recommended_books) > 4):
        with col4:
            st.text(recommended_books[4])
            st.image(poster_url[4])
    if (len(recommended_books) > 5):
        with col5:
            st.text(recommended_books[5])
            st.image(poster_url[5])