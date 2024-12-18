import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Titre de l'application
st.title("Application d'analyse de données interactives")

# SECTION 1 : Chargement ou téléchargement de données
st.header("1. Chargement des données")
st.markdown("**Option 1 :** Utilisez une base de données par défaut (Iris).")
st.markdown("**Option 2 :** Téléchargez vos propres données au format CSV.")

# Base de données par défaut (Iris)
iris = load_iris(as_frame=True)
default_data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
default_data['species'] = iris.target
default_data['species'] = default_data['species'].map(lambda x: iris.target_names[x])

# Téléchargement des données par défaut
if st.checkbox("Télécharger la base de données Iris (CSV)"):
    csv_data = default_data.to_csv(index=False).encode('utf-8')
    st.download_button("Télécharger Iris", csv_data, "iris.csv")

# Téléchargement de fichier par l'utilisateur
uploaded_file = st.file_uploader("Téléchargez un fichier CSV", type="csv")
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.success("Fichier chargé avec succès !")
else:
    data = default_data
    st.info("Base de données par défaut utilisée.")

st.write("Aperçu des données :", data.head())

# SECTION 2 : Statistiques descriptives et filtres
st.header("2. Analyse et manipulation des données")

if st.checkbox("Afficher des statistiques descriptives"):
    st.write(data.describe())

if st.checkbox("Filtrer par colonne spécifique"):
    column = st.selectbox("Choisissez une colonne :", data.columns)
    unique_values = data[column].unique()
    selected_value = st.selectbox(f"Choisissez une valeur de {column} :", unique_values)
    filtered_data = data[data[column] == selected_value]
    st.write(f"Données filtrées pour {column} = {selected_value} :", filtered_data)

# Export des données filtrées
if 'filtered_data' in locals() and not filtered_data.empty:
    st.download_button(
        "Télécharger les données filtrées",
        filtered_data.to_csv(index=False).encode('utf-8'),
        "filtered_data.csv"
    )

# SECTION 3 : Visualisation des données
st.header("3. Visualisation des données")

if st.checkbox("Afficher un pairplot (Seaborn)"):
    st.subheader("Pairplot des variables")
    fig = sns.pairplot(data, hue=data.columns[-1], markers=["o", "s", "D"])
    st.pyplot(fig)

if st.checkbox("Afficher un boxplot"):
    st.subheader("Boxplot d'une caractéristique")
    column_to_plot = st.selectbox("Choisissez une colonne pour l'axe des y :", data.columns[:-1])
    fig, ax = plt.subplots()
    sns.boxplot(data=data, x=data.columns[-1], y=column_to_plot, ax=ax)
    ax.set_title(f"Boxplot de {column_to_plot}")
    st.pyplot(fig)

if st.checkbox("Afficher un histogramme"):
    st.subheader("Histogramme d'une caractéristique")
    column_to_plot_hist = st.selectbox("Choisissez une colonne :", data.columns[:-1], key="histogram_column")
    fig, ax = plt.subplots()
    sns.histplot(data[column_to_plot_hist], kde=True, ax=ax)
    ax.set_title(f"Histogramme de {column_to_plot_hist}")
    st.pyplot(fig)

# SECTION 4 : Personnalisation avancée
st.header("4. Personnalisation avancée")
if st.checkbox("Personnaliser les couleurs et le style"):
    palette = st.color_picker("Choisissez une couleur pour les graphiques", "#3498db")
    style = st.selectbox("Choisissez un style de graphique :", ["darkgrid", "whitegrid", "dark", "white", "ticks"])
    sns.set_theme(style=style)
    st.info(f"Style appliqué : {style}, couleur : {palette}")
