import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# 🛠 Set Streamlit page config
st.set_page_config(page_title="Tamil Movie Recommender", layout="wide")

# 🎬 Load the data
@st.cache_data
def load_data():
    df = pd.read_csv("Tamil_movies_dataset.csv")
    return df

df = load_data()

st.title("🎥 Tamil Movie Recommender (IMDb Style)")
st.write("Pick a Tamil movie or enter a number to get movie recommendations based on genre, rating, and other features.")

# 📊 Normalize features
features = ['Rating', 'Hero_Rating', 'movie_rating', 'content_rating']
scaler = MinMaxScaler()
df_scaled = df.copy()
df_scaled[features] = scaler.fit_transform(df[features])

# 🎯 Recommendation function
def recommend_movies(movie_input, top_n=5):
    if isinstance(movie_input, str):
        movie_input = movie_input.strip().lower()
        if movie_input not in df['MovieName'].str.lower().values:
            return pd.DataFrame(), f"❌ Movie '{movie_input}' not found."
        idx = df[df['MovieName'].str.lower() == movie_input].index[0]
    else:
        if movie_input >= len(df):
            return pd.DataFrame(), f"❌ Invalid index: {movie_input}"
        idx = movie_input

    # Compute similarity
    sim = cosine_similarity([df_scaled.loc[idx, features]], df_scaled[features])[0]
    sim_scores = list(enumerate(sim))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_indices = [i for i, score in sim_scores if i != idx][:top_n]
    result = df.loc[top_indices].reset_index(drop=True)
    return result, ""

# 🎛️ User input
col1, col2 = st.columns([2, 1])
with col1:
    movie_input = st.text_input("Enter a movie name or index (0 to {}):".format(len(df) - 1), "Aadukalam")

with col2:
    top_n = st.slider("Number of Recommendations", 1, 10, 5)

# 🔍 Recommend movies
if movie_input:
    try:
        movie_input_eval = int(movie_input)
    except:
        movie_input_eval = movie_input

    recommended_df, error = recommend_movies(movie_input_eval, top_n)

    if error:
        st.error(error)
    else:
        st.subheader("🎬 Recommended Movies")
        st.dataframe(recommended_df[['MovieName', 'Genre', 'Year', 'Rating', 'Hero_Rating', 'content_rating']])
