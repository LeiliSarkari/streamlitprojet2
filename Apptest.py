#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import NearestNeighbors

# Fonction pour construire l'URL complète de l'image
def construct_image_url(file_path, file_size='w500'):
    base_url = "https://image.tmdb.org/t/p/"
    if file_path:
        return f"{base_url}{file_size}{file_path}"
    return None

# Lire le fichier Excel
file_path = r'C:\Users\wissa\Downloads\projet2\merged_df.xls'
df = pd.read_csv(file_path)  # Corrigez si le fichier est un Excel

# Convertir release_date en format datetime
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')

# Encodage one-hot des genres
df = pd.get_dummies(df, columns=["genres"], prefix=["genres"])

# Création d'une colonne 'all_genres' pour une représentation simplifiée des genres
df['all_genres'] = df.apply(lambda row: ', '.join([col.replace("genres_", "") for col in df.columns if col.startswith("genres_") and row[col] == 1]), axis=1)

# Colonnes numériques à normaliser
numeric_columns = ['averageRating', 'numVotes', 'popularity', 'runtime', 'vote_average']

# Normalisation des colonnes numériques
scaler = MinMaxScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# Préparation des données pour le modèle
features = df.columns.difference([
    'tconst', 'title', 'cast', 'posters', 'release_date', 
    'overview', 'production_countries', 'tagline', 
    'original_language', 'spoken_languages', 'all_genres'
])
X = df[features]

# Entraîner le modèle KNN avec une distance de similarité différente (cosinus)
knn_model = NearestNeighbors(n_neighbors=10, metric='cosine')
knn_model.fit(X)

# Fonction pour recommander des films
def recommend_movies(movie_title):
    try:
        # Trouver l'index du film donné
        movie_index = df[df["title"].str.contains(movie_title, case=False, na=False)].index[0]
        
        # Trouver les plus proches voisins
        _, indices = knn_model.kneighbors([X.iloc[movie_index]])
        
        # Indices des films recommandés (en excluant le premier, qui est le film lui-même)
        recommended_movies_index = indices[0][1:]
        
        recommendations = []
        for index in recommended_movies_index:
            title = df["title"].iloc[index]
            release_date = df["release_date"].iloc[index]
            poster = df["posters"].iloc[index]
            overview = df["overview"].iloc[index]
            genres = df["all_genres"].iloc[index]
            
            # Construction de l'URL complète pour les posters
            image_path = construct_image_url(poster)
            
            recommendations.append({
                "title": title,
                "release_date": release_date,
                "genres": genres,
                "poster": image_path,
                "overview": overview
            })
        return recommendations
    except IndexError:
        st.error(f"Le film '{movie_title}' n'a pas été trouvé dans la base de données.")
        return []
    except Exception as e:
        st.error(f"Une erreur est survenue : {e}")
        return []

# Interface utilisateur avec Streamlit
st.title("RetroCiné Creuse")

# Input de l'utilisateur pour le titre du film
movie_title_input = st.text_input("Entrez le titre du film pour obtenir des recommandations :")

# Bouton pour lancer la recherche
if st.button("Rechercher"):
    if movie_title_input:
        recommendations = recommend_movies(movie_title_input)
        if recommendations:
            st.write(f"Recommandations pour le film '{movie_title_input}':")
            for rec in recommendations:
                st.subheader(rec["title"])
                if rec["poster"]:
                    try:
                        st.image(rec["poster"], width=150)
                    except Exception as e:
                        st.warning(f"Impossible de charger l'image : {e}")
                st.write(f"Date de sortie : {rec['release_date'].strftime('%Y-%m-%d') if pd.notnull(rec['release_date']) else 'N/A'}")
                st.write(f"Genres : {rec['genres']}")
                st.write(f"Description : {rec['overview']}")
        else:
            st.write("Aucune recommandation trouvée. Vérifiez le titre du film.")
    else:
        st.write("Veuillez entrer un titre de film.")


# In[ ]:




