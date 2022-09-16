import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer  # To vectorize the movie names
import re  # To clean text
from sklearn.metrics.pairwise import cosine_similarity  # To calculate similarity
import streamlit as st
from zipfile import ZipFile
import os
pd.set_option('display.width', 1000)

#
header = st.container()
dataset = st.container()
features = st.container()

with header:
    st.title('Heyya!!!, You Should Watch This!!!')
    st.text(
        'Want Something to watch ? \nI Can Help You.')

path = 'data.zip'




@st.cache(allow_output_mutation=True)
def extractation(x):
    zip_data = ZipFile(x)
    et = {}
    with zip_data:
        for idx, file in enumerate(zip_data.namelist()):
            et['data_' + str(idx)] = zip_data.extract(file)
    return et

paths = extractation(path)
#
@st.cache(allow_output_mutation=True)
def load1(x):
    movies_data = pd.read_parquet(x)
    return movies_data

# print(load1(et['data_0']))
#

# movies_data = pd.read_parquet(et['data_0'])
movies_data = load1(paths['data_0'])
# movies_data = movies_data[~movies_data.genres.str.contains(r'\(')]

def clean_title(x):  # Create a function
    return re.sub('[^\w ]', '', x)  # This code removes anything except numbers,letters and blanks


movies_data['clean_title'] = movies_data.title.apply(lambda x: clean_title(x.strip()))  # Use the function to clean the title text in each row.

vec = TfidfVectorizer(ngram_range=(1, 2))  # Vectorizer converts the test into numpy
# arrays, it takes single words and word pairs into consideration
vec_data = vec.fit_transform(movies_data.clean_title)  # Transform the cleaned text column


def search(query):
    query = clean_title(query)  # Clean the variable passed in the function
    query = vec.transform([query])  # Vectorize the variable   **  Only transform **
    similarity = cosine_similarity(query, vec_data).flatten()  # Calculate the  similarity score
    # locs = np.append(np.argpartition(similarity,-10)[-10:],np.argmax(similarity))
    locs = np.argsort(similarity)[-10:]  # Find 10 indices with the highest score
    movies = movies_data.iloc[locs][::-1].drop_duplicates()  # Pass the indices in the movie data frame and create a new data frame.
    return movies

@st.cache(allow_output_mutation=True)
def load(url):
    ratings = pd.read_parquet(url)
    return ratings.iloc[:,1:]

ratings = load(paths['data_1'])


@st.cache
def recommendation(movie_id):
    ## Get userIds of people who liked the movie registered with the specified movie id. We can assume those users are similar users. I will refer this group as similar users to make things clear.
    similar_users = ratings[(ratings.movieId==movie_id) & (ratings.rating>4)]['userId'].unique()
    ## Collect the Ids of the other movies that similar people liked. Assume that similar people generally like similar movies.
    recs = ratings[(ratings.userId.isin(similar_users)==True) & (ratings.rating>4)]['movieId']
    ## Calculate which movie is liked how many times by similar users and divide it to the total number of the group. It shows us the percentage of people who like the movie
    recs = recs.value_counts() / len(similar_users)
    ## Filter the movies that are liked by at least %10 of the group.
    recs = recs[recs > 0.1]
    ## The data that show all users who liked the movies that the at least % 10 of the similar users also liked.
    all_=ratings[(ratings.movieId.isin(recs.index)==True) & (ratings.rating>4)]
    ## Calculate the ratio of the total population who liked the movies that the similar users liked.
    all_recs=all_['movieId'].value_counts()/len(all_['userId'].unique())
    ## Concatenate the ratio tables to see the comparison
    combined_recs = pd.concat([recs,all_recs],axis=1)
    ## Rename columns
    combined_recs.columns = ['similar','all']
    ## To calculate the score we use the percentages. If a movie is liked by similar people but not popular among the total population, it is assumed to be a better recommendation, because recommendation, in its nature, is valuable when the asker do not know about the movie. So we take the raio between the score among the similar people and the total population; the score is amplified when divided.
    combined_recs['score'] = 2*combined_recs['similar'] + combined_recs['all']
    ## Sort the data frame by score
    combined_recs=combined_recs.sort_values('score',ascending=False)
    ## Merge scores and the movies data frames on movieId column, filter 3 columns and the first 10 rows.

    results = combined_recs.merge(movies_data,left_index=True,right_on='movieId')[['title','genres','score']]

    genr = results.genres.iloc[0].split('|')

    frame = results.genres.apply(lambda x: 1 if len([k for k in genr if k in x.split('|')])>0 else 0)

    idx = frame[frame==1].index

    return results[results.index.isin(idx)].head(20)


with features:
    title = st.text_input('You want to watch a similar movie to :', 'Hulk')
    results = search(title)
    movie_id = results.iloc[0]['movieId']
    rec = recommendation(movie_id)
    st.dataframe(rec)  # Same as st.write(df)

# @st.cache
# def convert_df(df):
#     # IMPORTANT: Cache the conversion to prevent computation on every rerun
#     return df.to_csv().encode('utf-8')
#
#
# csv = convert_df(movies_data)
#
# st.download_button(
#     label="Download Movies as CSV",
#     data=csv,
#     file_name='movies_data.csv',
#     mime='text/csv',
# )
#
#
# csv1 = convert_df(movies_data)
#
# st.download_button(
#     label="Download Ratings as CSV",
#     data=csv1,
#     file_name='ratings_data.csv',
#     mime='text/csv',
# )
