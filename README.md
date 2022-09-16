# Movie Recommendation System

![1_t98V5s6uNKVNEde5ZYQemw](https://user-images.githubusercontent.com/105684729/189542347-0a1bd076-d0c1-4d4e-9edb-428bb09b6a64.jpeg)

An algorithm using cosine similarity to make movie recommendations based on user ratings.

The data is composed of 62424 different movies and the rating scores assigned to them by users. The rating scores range between 0.5 to 5.0 increasing by 0.5 for each step. 

## Logic of the algorithm
The users who rated the queried movie highly are assumed to have similar taste. Just as it is in real life, the algorithm 
asks this group of people about the other movies they liked the most and the first 10 collected movie names are given as recommendations.


## Roadmap

- Load and Preprocess the data

- Visualize the results
 
- Deploy the algorithm


## Libraries 

- Scikit-learn 
- Pickle
- Pandas
- Streamlit
- NumPy

The algorithm is deployed on streamlit API.

[You can access it here](https://yusufgulcan-movie-recommend-movie-recom-a2r5id.streamlitapp.com/)

## Further Improvements
The data size could be increased to be able to select recommendations from a wider pool. 
The rating sensitivity could also be increased for more precise recommendations. 

