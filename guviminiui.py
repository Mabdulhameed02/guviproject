import streamlit as st
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# Load the dataset
@st.cache_data
def load_data():
    file_path = "swiggy_chennai_data.csv"
    df = pd.read_csv(file_path)
    df_cleaned = df.drop_duplicates(subset=['restaurant']).reset_index(drop=True)
    df_cleaned['subcity'].fillna("Unknown", inplace=True)
    df_cleaned['menu'].fillna("Unknown", inplace=True)
    df_cleaned['item'].fillna("Unknown", inplace=True)
    df_cleaned['veg_or_non_veg'].fillna("Unknown", inplace=True)
    df_cleaned['rating count'] = df_cleaned['rating count'].replace("Too Few Ratings", "0")
    df_cleaned['rating count'] = pd.to_numeric(df_cleaned['rating count'], errors='coerce').fillna(0)
    df_cleaned['price'].fillna(df_cleaned['price'].median(), inplace=True)
    return df_cleaned

def preprocess_data(df):
    df_reduced = df[['restaurant', 'rating', 'cost', 'cuisine', 'veg_or_non_veg']].copy()
    df_reduced['cuisine'] = df_reduced['cuisine'].astype('category').cat.codes
    df_reduced['veg_or_non_veg'] = df_reduced['veg_or_non_veg'].astype('category').cat.codes
    df_reduced['rating'] = (df_reduced['rating'] - df_reduced['rating'].min()) / (df_reduced['rating'].max() - df_reduced['rating'].min())
    df_reduced['cost'] = (df_reduced['cost'] - df_reduced['cost'].min()) / (df_reduced['cost'].max() - df_reduced['cost'].min())
    return df_reduced

def train_knn(df):
    features = df[['rating', 'cost', 'cuisine', 'veg_or_non_veg']]
    knn = NearestNeighbors(n_neighbors=6, metric='euclidean')
    knn.fit(features)
    return knn, features

def recommend_knn(restaurant_name, df, knn, features, num_recommendations=5):
    if restaurant_name not in df['restaurant'].values:
        return ["Restaurant not found."]
    idx = df[df['restaurant'] == restaurant_name].index[0]
    distances, indices = knn.kneighbors([features.iloc[idx]])
    return df.iloc[indices[0][1:num_recommendations+1]]['restaurant'].tolist()

# Streamlit UI
st.title("Swiggy Restaurant Recommendation System")

df = load_data()
df_reduced = preprocess_data(df)
knn, features = train_knn(df_reduced)

restaurant_list = df['restaurant'].unique().tolist()
selected_restaurant = st.selectbox("Select a restaurant:", restaurant_list)

if st.button("Recommend Similar Restaurants"):
    recommendations = recommend_knn(selected_restaurant, df_reduced, knn, features)
    st.write("### Recommended Restaurants:")
    for res in recommendations:
        st.write(f"- {res}")
